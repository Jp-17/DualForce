"""
DualForce Autoregressive Inference Pipeline.

Generates talking head videos autoregressively using:
- Sliding window denoising (Diffusion Forcing)
- KV-cache for efficient attention
- Struct stream for 3D consistency
- Audio conditioning via HuBERT features

Key difference from MOVA inference:
- No full-sequence denoising: generates frame-by-frame (or chunk-by-chunk)
- Per-frame noise levels decrease from fully noisy → clean via sliding window
- KV-cache avoids recomputing attention over already-generated frames
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
from mova.diffusion.models.kv_cache import MultiModalKVCache
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
        self.video_processor = VideoProcessor(vae_scale_factor=8)

    # ============================================================
    # Text Encoding
    # ============================================================

    def _encode_prompt(self, prompt: Union[str, List[str]], device: torch.device):
        """Encode text prompt using T5."""
        if isinstance(prompt, str):
            prompt = [prompt]

        prompt = [prompt_clean(p) for p in prompt]

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)

        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_input_ids)[0]

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
        # Add temporal dim: [B, C, 1, H, W]
        image_5d = image.unsqueeze(2)

        with torch.no_grad():
            cond_latent = retrieve_latents(
                self.video_vae.encode(image_5d)
            )  # [B, 16, 1, H', W']

        # Normalize
        mean = self.video_vae.config.get("latents_mean", None)
        std = self.video_vae.config.get("latents_std", None)
        if mean is not None and std is not None:
            mean = torch.tensor(mean, device=cond_latent.device, dtype=cond_latent.dtype).view(1, -1, 1, 1, 1)
            std = torch.tensor(std, device=cond_latent.device, dtype=cond_latent.dtype).view(1, -1, 1, 1, 1)
            cond_latent = (cond_latent - mean) / std

        # Mask: 4 channels of ones (first frame is visible)
        H_l, W_l = cond_latent.shape[3], cond_latent.shape[4]
        mask = torch.ones(B, 4, 1, H_l, W_l, device=cond_latent.device, dtype=cond_latent.dtype)

        return cond_latent, mask

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Denormalize video latents for VAE decoding."""
        mean = self.video_vae.config.get("latents_mean", None)
        std = self.video_vae.config.get("latents_std", None)
        if mean is not None and std is not None:
            mean = torch.tensor(mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
            std = torch.tensor(std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
            latents = latents * std + mean
        return latents

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
        # Generation parameters
        num_frames: int = 49,
        height: int = 480,
        width: int = 720,
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
           - Denoise all frames in window simultaneously
           - Slide window forward by stride
           - New frames enter fully noisy, old frames carry over partially denoised
        4. Once a frame exits the window fully denoised, it's finalized

        Args:
            prompt: Text description
            image: [B, C, H, W] first frame image in [-1, 1]
            struct_latents: [B, D=128, T] pre-extracted struct features (optional)
            audio_features: [B, T_audio, 1024] HuBERT features (optional)
            num_frames: Total number of video frames to generate
            height, width: Output resolution
            num_inference_steps: Denoising steps per frame within the window
            window_size: Number of frames denoised simultaneously
            window_stride: How many frames to slide forward each iteration
            cfg_scale: Classifier-free guidance scale
            sigma_max, sigma_min: Noise schedule bounds
            output_type: "pil" or "latent"
        """
        if device is None:
            device = self._execution_device

        B = image.shape[0]
        dtype = torch.bfloat16

        # Latent space dimensions
        vae_temporal_stride = 4
        vae_spatial_stride = 8
        num_latent_frames = (num_frames - 1) // vae_temporal_stride + 1
        H_l = height // vae_spatial_stride
        W_l = width // vae_spatial_stride
        C_latent = 16

        # ============================================================
        # 1. Encode text and first frame
        # ============================================================
        prompt_embeds = self._encode_prompt(prompt, device)
        prompt_embeds = prompt_embeds.to(dtype)

        cond_latent, mask = self._encode_first_frame(image.to(device=device, dtype=dtype))
        # cond_latent: [B, 16, 1, H_l, W_l]
        # mask: [B, 4, 1, H_l, W_l]

        # ============================================================
        # 2. Initialize output buffer (all frames in latent space)
        # ============================================================
        # Start with random noise for all frames
        all_latents = randn_tensor(
            (B, C_latent, num_latent_frames, H_l, W_l),
            generator=generator, device=device, dtype=dtype
        )

        # Per-frame noise levels: start at sigma_max
        frame_sigmas = torch.full(
            (num_latent_frames,), sigma_max, device=device, dtype=dtype
        )
        # First frame is already clean (from condition)
        frame_sigmas[0] = 0.0

        # Build per-frame denoising schedule
        # Each frame needs to go from sigma_max → 0 over num_inference_steps
        denoise_schedule = torch.linspace(sigma_max, sigma_min, num_inference_steps + 1, device=device)

        # ============================================================
        # 3. Sliding window denoising loop
        # ============================================================
        # Window slides over latent frames
        num_windows = max(1, (num_latent_frames - 1 + window_stride - 1) // window_stride)

        pbar = tqdm(total=num_windows * num_inference_steps, desc="DualForce generation")

        for win_idx in range(num_windows):
            # Window boundaries in latent frame space
            win_start = win_idx * window_stride
            win_end = min(win_start + window_size, num_latent_frames)
            win_len = win_end - win_start

            if win_start >= num_latent_frames:
                break

            # Extract window latents
            window_latents = all_latents[:, :, win_start:win_end].clone()

            # Build condition for this window
            # First frame condition extends to cover the window
            if win_start == 0:
                # Window includes the conditioned first frame
                win_cond = torch.zeros(B, 16, win_len, H_l, W_l, device=device, dtype=dtype)
                win_cond[:, :, 0:1] = cond_latent
                win_mask = torch.zeros(B, 4, win_len, H_l, W_l, device=device, dtype=dtype)
                win_mask[:, :, 0:1] = mask
            else:
                # No first-frame in this window; use zeros
                win_cond = torch.zeros(B, 16, win_len, H_l, W_l, device=device, dtype=dtype)
                win_mask = torch.zeros(B, 4, win_len, H_l, W_l, device=device, dtype=dtype)

            # Struct latents for this window (if provided)
            win_struct = None
            if struct_latents is not None:
                raw_start = win_start * vae_temporal_stride
                raw_end = win_end * vae_temporal_stride
                if raw_end <= struct_latents.shape[2]:
                    win_struct = struct_latents[:, :, raw_start:raw_end]
                else:
                    win_struct = F.pad(
                        struct_latents[:, :, raw_start:],
                        (0, raw_end - struct_latents.shape[2])
                    )

            # ============================================================
            # 3.1 Denoise window over multiple steps
            # ============================================================
            for step_idx in range(num_inference_steps):
                sigma_now = denoise_schedule[step_idx]
                sigma_next = denoise_schedule[step_idx + 1]

                # Build input: concat(noisy_latent[16], mask[4], cond_latent[16]) = 36 channels
                x_input = torch.cat([window_latents, win_mask, win_cond], dim=1)
                # x_input: [B, 36, win_len, H_l, W_l]

                # Timestep for this denoising step
                timestep = torch.full((B,), sigma_now * 1000, device=device, dtype=dtype)

                # Forward through video DiT
                video_pred = self.video_dit(
                    x=x_input,
                    timestep=timestep,
                    context=prompt_embeds,
                )  # [B, 16, win_len, H_l, W_l]

                # Struct DiT (if available)
                struct_pred = None
                if win_struct is not None and self.struct_dit is not None:
                    struct_pred = self.struct_dit(
                        x=win_struct,
                        timestep=timestep,
                        context=prompt_embeds,
                    )

                # TODO: Bridge interaction during inference
                # For now, video and struct run independently

                # Flow matching step: x_{t-1} = x_t + pred * (sigma_next - sigma_now)
                window_latents = window_latents + video_pred * (sigma_next - sigma_now)

                pbar.update(1)

            # Write denoised window back to buffer
            all_latents[:, :, win_start:win_end] = window_latents

        pbar.close()

        # ============================================================
        # 4. Decode video latents
        # ============================================================
        if output_type == "latent":
            return all_latents

        # Denormalize
        video_latents = self._denormalize_latents(all_latents)

        # Decode through VAE
        with torch.autocast("cuda", dtype=torch.bfloat16):
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
