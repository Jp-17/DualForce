"""
DualForce Autoregressive Inference Pipeline.

Generates talking head videos autoregressively using:
- Sliding window denoising (Diffusion Forcing)
- Dual-tower (video + struct) with bridge cross-attention
- Audio conditioning via HuBERT features
- Classifier-free guidance (CFG)

Key difference from MOVA inference:
- No full-sequence denoising: generates frame-by-frame (or chunk-by-chunk)
- Per-frame noise levels decrease from fully noisy → clean via sliding window
- 3D struct stream provides structural consistency across frames
"""

import gc
import html
import os
import re
from typing import Any, List, Optional, Tuple, Union

import ftfy
import torch
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm
from transformers.models import T5TokenizerFast, UMT5EncoderModel

from mova.diffusion.models import WanModel, WanStructModel, sinusoidal_embedding_1d
from mova.diffusion.models.interactionv2 import DualTowerConditionalBridge
from mova.diffusion.models.audio_conditioning import AudioConditioningModule, DualAdaLNZero
from mova.diffusion.schedulers.diffusion_forcing import DiffusionForcingScheduler
from mova.registry import DIFFUSION_PIPELINES


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def prompt_clean(text):
    text = re.sub(r"\s+", " ", basic_clean(text))
    return text.strip()


def retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents.")


class DualForceInference(DiffusionPipeline):
    """
    DualForce autoregressive inference pipeline.

    Generates talking head video frame-by-frame using Diffusion Forcing's
    sliding window denoising schedule. Each new frame starts fully noisy
    and is progressively denoised while attending to already-clean past frames.
    The dual-tower architecture runs video and 3D struct streams in parallel
    with bridge cross-attention for structural consistency.
    """

    model_cpu_offload_seq = "text_encoder->video_dit->video_vae"

    video_vae: AutoencoderKLWan
    text_encoder: UMT5EncoderModel
    tokenizer: T5TokenizerFast
    scheduler: DiffusionForcingScheduler
    video_dit: WanModel
    struct_dit: WanStructModel
    dual_tower_bridge: DualTowerConditionalBridge

    @register_to_config
    def __init__(
        self,
        video_vae: AutoencoderKLWan,
        text_encoder: UMT5EncoderModel,
        tokenizer: T5TokenizerFast,
        scheduler: DiffusionForcingScheduler,
        video_dit: WanModel,
        struct_dit: WanStructModel,
        dual_tower_bridge: DualTowerConditionalBridge,
        audio_conditioning: Optional[AudioConditioningModule] = None,
        video_adaln: Optional[DualAdaLNZero] = None,
        struct_adaln: Optional[DualAdaLNZero] = None,
    ):
        super().__init__()
        self.register_modules(
            video_vae=video_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            video_dit=video_dit,
            struct_dit=struct_dit,
            dual_tower_bridge=dual_tower_bridge,
        )
        self.audio_conditioning = audio_conditioning
        self.video_adaln = video_adaln
        self.struct_adaln = struct_adaln
        self.video_processor = VideoProcessor(vae_scale_factor=8)

    # ============================================================
    # Text Encoding
    # ============================================================

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        max_sequence_length: int = 512,
    ):
        """Encode text prompt using T5."""
        if isinstance(prompt, str):
            prompt = [prompt]

        prompt = [prompt_clean(p) for p in prompt]

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                text_input_ids, attention_mask=attention_mask
            ).last_hidden_state

        # Trim to actual lengths
        seq_lens = attention_mask.sum(dim=-1).tolist()
        prompt_embeds = [u[:seq_len] for u, seq_len in zip(prompt_embeds, seq_lens)]

        max_len = max(seq_lens)
        prompt_embeds = torch.stack([
            F.pad(u, (0, 0, 0, max_len - u.shape[0])) for u in prompt_embeds
        ])

        return prompt_embeds

    # ============================================================
    # Latent Helpers
    # ============================================================

    def _encode_first_frame(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode first frame image to latent space.

        Args:
            image: [B, C, H, W] first frame in [-1, 1]

        Returns:
            cond_latent: [B, 16, 1, H', W'] encoded first frame
            mask: [B, 4, 1, H', W'] conditioning mask (ones for first frame)
        """
        B = image.shape[0]
        image_5d = image.unsqueeze(2)  # [B, C, 1, H, W]

        device_type = image.device.type if isinstance(image.device, torch.device) else str(image.device).split(":")[0]

        with torch.no_grad():
            with torch.autocast(device_type, dtype=image.dtype):
                cond_latent = retrieve_latents(
                    self.video_vae.encode(image_5d), sample_mode="argmax"
                )  # [B, 16, 1, H', W']

        # Normalize
        mean = getattr(self.video_vae.config, "latents_mean", None)
        std = getattr(self.video_vae.config, "latents_std", None)
        if mean is not None and std is not None:
            mean = torch.tensor(mean, device=cond_latent.device, dtype=cond_latent.dtype).view(1, -1, 1, 1, 1)
            std = torch.tensor(std, device=cond_latent.device, dtype=cond_latent.dtype).view(1, -1, 1, 1, 1)
            cond_latent = (cond_latent - mean) / std

        H_l, W_l = cond_latent.shape[3], cond_latent.shape[4]
        mask = torch.ones(B, 4, 1, H_l, W_l, device=cond_latent.device, dtype=cond_latent.dtype)

        return cond_latent, mask

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Denormalize video latents for VAE decoding."""
        mean = getattr(self.video_vae.config, "latents_mean", None)
        std = getattr(self.video_vae.config, "latents_std", None)
        if mean is not None and std is not None:
            mean = torch.tensor(mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
            std = torch.tensor(std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
            latents = latents * std + mean
        return latents

    # ============================================================
    # Dual-Tower Inference Forward
    # ============================================================

    def _dual_tower_forward(
        self,
        visual_x: torch.Tensor,        # [B, 36, T', H', W'] video input (noisy + cond)
        struct_x: torch.Tensor,         # [B, 128, T_struct] struct latents
        context: torch.Tensor,          # [B, seq_len, 4096] text embeddings
        video_timestep: torch.Tensor,   # [B] global timestep for video
        struct_timestep: torch.Tensor,  # [B] global timestep for struct
        video_fps: float = 25.0,
        sigma_v: Optional[torch.Tensor] = None,  # [B, T'] per-frame sigma for video
        sigma_s: Optional[torch.Tensor] = None,  # [B, T_struct] per-frame sigma for struct
        audio_features: Optional[torch.Tensor] = None,  # [B, T_audio, 1024]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run dual-tower inference forward pass.

        Returns:
            video_pred: [B, 16, T', H', W'] velocity prediction for video
            struct_pred: [B, 128, T_struct] velocity prediction for struct
        """
        model_dtype = torch.bfloat16

        # --- Timestep embeddings ---
        device_type = visual_x.device.type
        with torch.autocast(device_type, dtype=torch.float32):
            visual_t = self.video_dit.time_embedding(
                sinusoidal_embedding_1d(self.video_dit.freq_dim, video_timestep))
            visual_t_mod = self.video_dit.time_projection(visual_t).unflatten(1, (6, self.video_dit.dim))

            struct_t = self.struct_dit.time_embedding(
                sinusoidal_embedding_1d(self.struct_dit.freq_dim, struct_timestep))
            struct_t_mod = self.struct_dit.time_projection(struct_t).unflatten(1, (6, self.struct_dit.dim))

        visual_t = visual_t.to(model_dtype)
        visual_t_mod = visual_t_mod.to(model_dtype)
        struct_t = struct_t.to(model_dtype)
        struct_t_mod = struct_t_mod.to(model_dtype)

        # --- Text embeddings per tower ---
        visual_context = self.video_dit.text_embedding(context)
        struct_context = self.struct_dit.text_embedding(context)

        # --- Patchify video ---
        visual_x = visual_x.to(model_dtype)
        visual_x, (t, h, w) = self.video_dit.patchify(visual_x)
        grid_size = (t, h, w)

        # Video RoPE
        visual_freqs = tuple(freq.to(visual_x.device) for freq in self.video_dit.freqs)
        visual_freqs = torch.cat([
            visual_freqs[0][:t].view(t, 1, 1, -1).expand(t, h, w, -1),
            visual_freqs[1][:h].view(1, h, 1, -1).expand(t, h, w, -1),
            visual_freqs[2][:w].view(1, 1, w, -1).expand(t, h, w, -1)
        ], dim=-1).reshape(t * h * w, 1, -1).to(visual_x.device)

        # --- Tokenize struct ---
        struct_x = struct_x.to(model_dtype)
        struct_x, (f,) = self.struct_dit.tokenize(struct_x)

        # Struct RoPE
        struct_freqs = torch.cat([
            self.struct_dit.freqs[0][:f].view(f, -1).expand(f, -1),
            self.struct_dit.freqs[1][:f].view(f, -1).expand(f, -1),
            self.struct_dit.freqs[2][:f].view(f, -1).expand(f, -1),
        ], dim=-1).reshape(f, 1, -1).to(struct_x.device)

        # --- Per-frame AdaLN modulation ---
        if self.video_adaln is not None and sigma_v is not None:
            visual_t_mod = self.video_adaln(sigma_v, grid_size).to(model_dtype)
        if self.struct_adaln is not None and sigma_s is not None:
            struct_t_mod = self.struct_adaln(sigma_s, (f,)).to(model_dtype)

        # --- Causal grid sizes ---
        visual_causal_grid = grid_size if getattr(self.video_dit, 'causal_temporal', False) else None
        struct_causal_grid = (f, 1, 1) if getattr(self.struct_dit, 'causal_temporal', False) else None

        # --- Bridge cross-RoPE ---
        if self.dual_tower_bridge.apply_cross_rope:
            (visual_rope_cos_sin, struct_rope_cos_sin) = self.dual_tower_bridge.build_aligned_freqs(
                video_fps=video_fps,
                grid_size=grid_size,
                audio_steps=struct_x.shape[1],
                device=visual_x.device,
                dtype=visual_x.dtype,
            )
        else:
            visual_rope_cos_sin = None
            struct_rope_cos_sin = None

        # --- Forward through paired layers ---
        min_layers = min(len(self.video_dit.blocks), len(self.struct_dit.blocks))

        for layer_idx in range(min_layers):
            visual_block = self.video_dit.blocks[layer_idx]
            struct_block = self.struct_dit.blocks[layer_idx]

            # Bridge cross-attention
            if self.dual_tower_bridge.should_interact(layer_idx, 'a2v'):
                visual_x, struct_x = self.dual_tower_bridge(
                    layer_idx, visual_x, struct_x,
                    x_freqs=visual_rope_cos_sin,
                    y_freqs=struct_rope_cos_sin,
                    condition_scale=1.0,
                    video_grid_size=grid_size,
                )

            # Video DiT block
            visual_x = visual_block(
                visual_x, visual_context, visual_t_mod, visual_freqs,
                causal_grid_size=visual_causal_grid,
            )

            # Struct DiT block
            struct_x = struct_block(
                struct_x, struct_context, struct_t_mod, struct_freqs,
                causal_grid_size=struct_causal_grid,
            )

            # Audio conditioning (shallow layers only)
            if audio_features is not None and self.audio_conditioning is not None:
                visual_x = self.audio_conditioning.condition_video(layer_idx, visual_x, audio_features)
                struct_x = self.audio_conditioning.condition_struct(layer_idx, struct_x, audio_features)

        # Remaining video layers
        for layer_idx in range(min_layers, len(self.video_dit.blocks)):
            visual_x = self.video_dit.blocks[layer_idx](
                visual_x, visual_context, visual_t_mod, visual_freqs,
                causal_grid_size=visual_causal_grid,
            )

        # Remaining struct layers
        for layer_idx in range(min_layers, len(self.struct_dit.blocks)):
            struct_x = self.struct_dit.blocks[layer_idx](
                struct_x, struct_context, struct_t_mod, struct_freqs,
                causal_grid_size=struct_causal_grid,
            )

        # --- Output heads ---
        video_pred = self.video_dit.head(visual_x, visual_t)
        video_pred = self.video_dit.unpatchify(video_pred, grid_size)  # [B, 16, T', H', W']

        struct_pred = self.struct_dit.head(struct_x, struct_t)
        struct_pred = self.struct_dit.detokenize(struct_pred, (f,))  # [B, 128, T]

        return video_pred, struct_pred

    # ============================================================
    # Sliding Window Autoregressive Generation
    # ============================================================

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: torch.Tensor,
        struct_latents: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # Generation parameters
        num_frames: int = 49,
        height: int = 480,
        width: int = 720,
        video_fps: float = 25.0,
        # Denoising parameters
        num_inference_steps: int = 20,
        window_size: int = 8,
        window_stride: int = 4,
        cfg_scale: float = 5.0,
        # Scheduler
        sigma_max: float = 1.0,
        sigma_min: float = 0.0,
        # Output
        output_type: str = "pil",
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Generate talking head video autoregressively.

        The sliding window approach:
        1. Start with first frame encoded as condition
        2. Initialize a window of noisy latent frames
        3. For each denoising step in the window:
           - Run dual-tower forward (video + struct with bridge)
           - Apply flow matching step
           - Slide window forward by stride
        4. Once a frame exits the window fully denoised, it's finalized

        Args:
            prompt: Text description
            image: [B, C, H, W] first frame image in [-1, 1]
            struct_latents: [B, D=128, T] pre-extracted struct features (optional, generated if None)
            audio_features: [B, T_audio, 1024] HuBERT features (optional)
            negative_prompt: Negative prompt for CFG
            num_frames: Total number of video frames to generate
            height, width: Output resolution
            video_fps: Frame rate
            num_inference_steps: Denoising steps per frame within the window
            window_size: Number of frames denoised simultaneously
            window_stride: How many frames to slide forward each iteration
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
            sigma_max, sigma_min: Noise schedule bounds
            output_type: "pil", "pt", or "latent"
        """
        if device is None:
            device = self._execution_device

        B = image.shape[0]
        dtype = torch.bfloat16
        device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]

        do_cfg = cfg_scale > 1.0

        # Latent space dimensions
        vae_temporal_stride = 4
        vae_spatial_stride = 8
        num_latent_frames = (num_frames - 1) // vae_temporal_stride + 1
        H_l = height // vae_spatial_stride
        W_l = width // vae_spatial_stride
        C_latent = 16

        # Struct dimensions
        D_struct = 128
        T_struct = num_latent_frames  # One struct token per latent frame

        # ============================================================
        # 1. Encode text
        # ============================================================
        prompt_embeds = self._encode_prompt(prompt, device).to(dtype)

        if do_cfg:
            neg_prompt = negative_prompt or ""
            if isinstance(neg_prompt, str):
                neg_prompt = [neg_prompt] * B
            neg_prompt_embeds = self._encode_prompt(neg_prompt, device).to(dtype)

        # ============================================================
        # 2. Encode first frame
        # ============================================================
        cond_latent, mask = self._encode_first_frame(image.to(device=device, dtype=dtype))
        # cond_latent: [B, 16, 1, H_l, W_l]
        # mask: [B, 4, 1, H_l, W_l]

        # ============================================================
        # 3. Initialize buffers
        # ============================================================
        # Video latents: start with noise
        all_video_latents = randn_tensor(
            (B, C_latent, num_latent_frames, H_l, W_l),
            generator=generator, device=device, dtype=dtype,
        )

        # Struct latents: start from provided or noise
        if struct_latents is not None:
            all_struct_latents = struct_latents.to(device=device, dtype=dtype)
            generate_struct = False
        else:
            all_struct_latents = randn_tensor(
                (B, D_struct, T_struct),
                generator=generator, device=device, dtype=dtype,
            )
            generate_struct = True

        # Build denoising schedule
        denoise_schedule = torch.linspace(sigma_max, sigma_min, num_inference_steps + 1, device=device)

        # ============================================================
        # 4. Sliding window denoising
        # ============================================================
        num_windows = max(1, (num_latent_frames - 1 + window_stride - 1) // window_stride)
        pbar = tqdm(total=num_windows * num_inference_steps, desc="DualForce generation")

        for win_idx in range(num_windows):
            win_start = win_idx * window_stride
            win_end = min(win_start + window_size, num_latent_frames)
            win_len = win_end - win_start

            if win_start >= num_latent_frames:
                break

            # Extract window video latents
            win_video = all_video_latents[:, :, win_start:win_end].clone()

            # Extract window struct latents
            win_struct = all_struct_latents[:, :, win_start:win_end].clone()

            # Build first-frame condition for this window
            if win_start == 0:
                win_cond = torch.zeros(B, 16, win_len, H_l, W_l, device=device, dtype=dtype)
                win_cond[:, :, 0:1] = cond_latent
                win_mask = torch.zeros(B, 4, win_len, H_l, W_l, device=device, dtype=dtype)
                win_mask[:, :, 0:1] = mask
            else:
                win_cond = torch.zeros(B, 16, win_len, H_l, W_l, device=device, dtype=dtype)
                win_mask = torch.zeros(B, 4, win_len, H_l, W_l, device=device, dtype=dtype)

            # Extract audio features for this window (if provided)
            win_audio = None
            if audio_features is not None:
                # Audio features are at HuBERT framerate; approximate temporal alignment
                T_audio = audio_features.shape[1]
                ratio = T_audio / num_latent_frames
                a_start = int(win_start * ratio)
                a_end = int(win_end * ratio)
                a_end = min(a_end, T_audio)
                if a_start < T_audio:
                    win_audio = audio_features[:, a_start:a_end].to(device=device, dtype=dtype)

            # Denoise this window
            for step_idx in range(num_inference_steps):
                sigma_now = denoise_schedule[step_idx]
                sigma_next = denoise_schedule[step_idx + 1]

                # Per-frame sigma tensors (uniform within window for inference)
                sigma_v = torch.full((B, win_len), sigma_now.item(), device=device, dtype=dtype)
                sigma_s = torch.full((B, win_len), sigma_now.item(), device=device, dtype=dtype)

                # Build video DiT input: [noisy_video(16) + mask(4) + cond(16)] = 36 channels
                x_input = torch.cat([win_video, win_mask, win_cond], dim=1)

                # Timestep
                timestep_val = self.scheduler.sigma_to_timestep(sigma_now.unsqueeze(0)).item()
                video_timestep = torch.full((B,), timestep_val, device=device, dtype=dtype)
                struct_timestep = torch.full((B,), timestep_val, device=device, dtype=dtype)

                # Forward pass (conditional)
                video_pred, struct_pred = self._dual_tower_forward(
                    visual_x=x_input,
                    struct_x=win_struct if generate_struct else win_struct.clone(),
                    context=prompt_embeds,
                    video_timestep=video_timestep,
                    struct_timestep=struct_timestep,
                    video_fps=video_fps,
                    sigma_v=sigma_v,
                    sigma_s=sigma_s,
                    audio_features=win_audio,
                )

                # CFG: unconditional forward + guidance
                if do_cfg:
                    video_pred_uncond, struct_pred_uncond = self._dual_tower_forward(
                        visual_x=x_input,
                        struct_x=win_struct if generate_struct else win_struct.clone(),
                        context=neg_prompt_embeds,
                        video_timestep=video_timestep,
                        struct_timestep=struct_timestep,
                        video_fps=video_fps,
                        sigma_v=sigma_v,
                        sigma_s=sigma_s,
                        audio_features=None,  # No audio for uncond
                    )
                    video_pred = video_pred_uncond + cfg_scale * (video_pred - video_pred_uncond)
                    if generate_struct:
                        struct_pred = struct_pred_uncond + cfg_scale * (struct_pred - struct_pred_uncond)

                # Flow matching step: x_{t-1} = x_t + v_pred * (sigma_next - sigma_now)
                dt = sigma_next - sigma_now
                win_video = win_video + video_pred * dt

                if generate_struct:
                    win_struct = win_struct + struct_pred * dt

                pbar.update(1)

            # Write denoised window back to buffer
            all_video_latents[:, :, win_start:win_end] = win_video
            if generate_struct:
                all_struct_latents[:, :, win_start:win_end] = win_struct

        pbar.close()

        # ============================================================
        # 5. Decode video latents
        # ============================================================
        if output_type == "latent":
            return all_video_latents, all_struct_latents

        # Denormalize
        video_latents = self._denormalize_latents(all_video_latents)

        # Decode through VAE
        with torch.autocast(device_type, dtype=torch.bfloat16):
            video = self.video_vae.decode(video_latents).sample

        # Post-process
        if output_type == "pil":
            video = self.video_processor.postprocess_video(video, output_type="pil")
        elif output_type == "pt":
            video = (video + 1) / 2  # [-1, 1] -> [0, 1]

        return video


@DIFFUSION_PIPELINES.register_module()
def DualForceInference_from_pretrained(
    from_pretrained: str,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load DualForce inference pipeline from a saved checkpoint."""
    model = DualForceInference.from_pretrained(
        from_pretrained,
        torch_dtype=torch_dtype,
    )
    return model
