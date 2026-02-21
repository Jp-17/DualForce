"""
DualForce Training Pipeline.

Adapts MOVATrain for DualForce with:
- Single video DiT (no video_dit_2)
- 3D structure stream replacing audio tower
- Per-frame independent noise (Diffusion Forcing)
- Multi-modal losses (video, struct, FLAME alignment, lip-sync)
- HuBERT audio conditioning (not as a diffusion modality)
"""

import re
import ftfy
import html
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Dict, List, Tuple
from functools import partial

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from diffusers.pipelines import DiffusionPipeline
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.configuration_utils import register_to_config
from transformers.models import UMT5EncoderModel, T5TokenizerFast

from mova.diffusion.models import WanModel, WanStructModel, sinusoidal_embedding_1d
from mova.diffusion.models.interactionv2 import DualTowerConditionalBridge
from mova.diffusion.schedulers.diffusion_forcing import DiffusionForcingScheduler

from mova.distributed.functional import (
    _sp_split_tensor, _sp_split_tensor_dim_0, _sp_all_gather_avg
)

from dataclasses import dataclass


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


@dataclass
class DualForceLossConfig:
    """Loss weights for DualForce multi-modal training."""
    video_weight: float = 1.0
    struct_weight: float = 0.5
    flame_weight: float = 0.1
    lip_sync_weight: float = 0.3


class DualForceTrain(DiffusionPipeline):
    """
    DualForce training pipeline.

    Components:
    - video_dit: MOVA-Lite video DiT (single, no two-stage)
    - struct_dit: 3D Structure DiT (replaces audio tower)
    - dual_tower_bridge: Video <-> 3D bidirectional attention
    - video_vae: Frozen video VAE for latent encoding
    - text_encoder: Frozen UMT5 for text conditioning
    - scheduler: DiffusionForcingScheduler with per-frame noise
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
        # DualForce specific
        loss_config: Optional[Dict] = None,
        # Gradient checkpointing
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
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

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        # Video VAE config
        self.height_division_factor = self.video_vae.spatial_compression_ratio * 2
        self.width_division_factor = self.video_vae.spatial_compression_ratio * 2

        # Loss config
        if loss_config is None:
            loss_config = {}
        self.loss_config = DualForceLossConfig(**loss_config)

    # ============================================================
    # Forward (called by AccelerateTrainer)
    # ============================================================

    def forward(self, **kwargs) -> dict:
        """Dispatch to training_step with proper argument mapping.

        The AccelerateTrainer calls model(video=..., audio=..., caption=..., ...)
        but DualForce's dataset provides video_latents, struct_latents, etc.
        This method bridges both conventions.
        """
        # Map from DualForceDataset collate_fn output
        if "video_latents" in kwargs:
            return self.training_step(
                video_latents=kwargs["video_latents"],
                struct_latents=kwargs["struct_latents"],
                audio_features=kwargs.get("audio_features"),
                caption=kwargs["caption"],
                first_frame=kwargs["first_frame"],
                flame_params=kwargs.get("flame_params"),
                cp_mesh=kwargs.get("cp_mesh"),
            )
        # Map from MOVA-style batch keys (fallback compatibility)
        elif "video" in kwargs:
            return self.training_step(
                video_latents=kwargs["video"],
                struct_latents=kwargs.get("audio", torch.zeros(1)),
                audio_features=kwargs.get("audio"),
                caption=kwargs["caption"],
                first_frame=kwargs["first_frame"],
                flame_params=kwargs.get("flame_params"),
                cp_mesh=kwargs.get("cp_mesh"),
            )
        else:
            raise ValueError(f"Unexpected kwargs: {list(kwargs.keys())}")

    # ============================================================
    # Text Encoding (same as MOVA)
    # ============================================================

    def _get_t5_prompt_embeds(
        self,
        prompt: List[str],
        device: torch.device = None,
        max_sequence_length: int = 512,
    ):
        device = device or self.device

        prompt = [prompt_clean(u) for u in prompt]

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

        prompt_embeds = self.text_encoder(
            text_input_ids, attention_mask=attention_mask
        ).last_hidden_state

        # Trim to actual lengths
        seq_lens = attention_mask.sum(dim=-1).tolist()
        prompt_embeds = [u[:seq_len] for u, seq_len in zip(prompt_embeds, seq_lens)]

        # Pad to max length
        max_len = max(seq_lens)
        prompt_embeds = torch.stack([
            F.pad(u, (0, 0, 0, max_len - u.shape[0])) for u in prompt_embeds
        ])

        return prompt_embeds

    # ============================================================
    # Video Latent Encoding
    # ============================================================

    def normalize_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalize video latents using VAE statistics."""
        mean = torch.tensor(
            self.video_vae.config.latents_mean,
            device=latents.device, dtype=latents.dtype
        ).view(1, -1, 1, 1, 1)
        std = torch.tensor(
            self.video_vae.config.latents_std,
            device=latents.device, dtype=latents.dtype
        ).view(1, -1, 1, 1, 1)
        return (latents - mean) / std

    # ============================================================
    # Dual Tower Forward Pass
    # ============================================================

    def forward_dual_tower(
        self,
        visual_x: torch.Tensor,        # [B, L_v, dim_v] patchified video tokens
        struct_x: torch.Tensor,         # [B, L_s, dim_s] struct tokens
        visual_context: torch.Tensor,   # [B, seq_len, dim_v] text embedding for video
        struct_context: torch.Tensor,   # [B, seq_len, dim_s] text embedding for struct
        visual_t_mod: torch.Tensor,     # [B, 6, dim_v] timestep modulation for video
        struct_t_mod: torch.Tensor,     # [B, 6, dim_s] timestep modulation for struct
        visual_freqs: torch.Tensor,     # [L_v, 1, head_dim] video RoPE
        struct_freqs: torch.Tensor,     # [L_s, 1, head_dim] struct RoPE
        grid_size: Tuple[int, int, int],  # (T, H, W) video grid
        video_fps: float,
        cp_mesh: Optional[DeviceMesh] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both towers with bridge cross-attention.

        Returns:
            visual_x: [B, L_v, dim_v] video hidden states
            struct_x: [B, L_s, dim_s] struct hidden states
        """
        min_layers = min(len(self.video_dit.blocks), len(self.struct_dit.blocks))

        # Precompute aligned RoPE for cross-modal attention
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

        # Context parallel setup
        sp_enabled = False
        if cp_mesh is not None:
            sp_rank = cp_mesh.get_local_rank()
            sp_size = cp_mesh.size()
            sp_group = cp_mesh.get_group()
            visual_x, _, visual_pad_len, _ = _sp_split_tensor(visual_x, sp_size=sp_size, sp_rank=sp_rank)
            struct_x, _, struct_pad_len, _ = _sp_split_tensor(struct_x, sp_size=sp_size, sp_rank=sp_rank)
            visual_freqs, _, _, _ = _sp_split_tensor_dim_0(visual_freqs, sp_size=sp_size, sp_rank=sp_rank)
            struct_freqs, _, _, _ = _sp_split_tensor_dim_0(struct_freqs, sp_size=sp_size, sp_rank=sp_rank)
            if visual_rope_cos_sin is not None:
                visual_rope_cos_sin = [
                    _sp_split_tensor(rcs, sp_size=sp_size, sp_rank=sp_rank)[0]
                    for rcs in visual_rope_cos_sin
                ]
            if struct_rope_cos_sin is not None:
                struct_rope_cos_sin = [
                    _sp_split_tensor(rcs, sp_size=sp_size, sp_rank=sp_rank)[0]
                    for rcs in struct_rope_cos_sin
                ]
            if len(visual_t_mod.shape) == 4:
                visual_t_mod, _, _, _ = _sp_split_tensor(visual_t_mod, sp_size=sp_size, sp_rank=sp_rank)
            sp_enabled = True

        def _make_custom_forward(module):
            def _fn(*inputs):
                return module(*inputs)
            return _fn

        # Determine causal grid sizes for each tower
        visual_causal_grid = grid_size if getattr(self.video_dit, 'causal_temporal', False) else None
        # Struct tokens are 1D temporal: treat as (num_tokens, 1, 1) for block-causal mask
        struct_causal_grid = (struct_x.shape[1], 1, 1) if getattr(self.struct_dit, 'causal_temporal', False) else None

        # Forward through paired layers
        for layer_idx in range(min_layers):
            visual_block = self.video_dit.blocks[layer_idx]
            struct_block = self.struct_dit.blocks[layer_idx]

            # Bridge: struct <-> video bidirectional cross-attention
            if self.dual_tower_bridge.should_interact(layer_idx, 'a2v'):
                if self.use_gradient_checkpointing and self.training:
                    def _bridge_fn(li, vx, sx):
                        return self.dual_tower_bridge(
                            li, vx, sx,
                            x_freqs=visual_rope_cos_sin,
                            y_freqs=struct_rope_cos_sin,
                            condition_scale=1.0,
                            video_grid_size=grid_size,
                        )
                    if self.use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            visual_x, struct_x = torch.utils.checkpoint.checkpoint(
                                _bridge_fn, layer_idx, visual_x, struct_x,
                                use_reentrant=False,
                            )
                    else:
                        visual_x, struct_x = torch.utils.checkpoint.checkpoint(
                            _bridge_fn, layer_idx, visual_x, struct_x,
                            use_reentrant=False,
                        )
                else:
                    visual_x, struct_x = self.dual_tower_bridge(
                        layer_idx, visual_x, struct_x,
                        x_freqs=visual_rope_cos_sin,
                        y_freqs=struct_rope_cos_sin,
                        condition_scale=1.0,
                        video_grid_size=grid_size,
                    )

            # Visual DiT block
            if self.use_gradient_checkpointing and self.training:
                if self.use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        visual_x = torch.utils.checkpoint.checkpoint(
                            _make_custom_forward(visual_block),
                            visual_x, visual_context, visual_t_mod, visual_freqs, visual_causal_grid,
                            use_reentrant=False,
                        )
                else:
                    visual_x = torch.utils.checkpoint.checkpoint(
                        _make_custom_forward(visual_block),
                        visual_x, visual_context, visual_t_mod, visual_freqs, visual_causal_grid,
                        use_reentrant=False,
                    )
            else:
                visual_x = visual_block(visual_x, visual_context, visual_t_mod, visual_freqs, causal_grid_size=visual_causal_grid)

            # Struct DiT block
            if self.use_gradient_checkpointing and self.training:
                if self.use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        struct_x = torch.utils.checkpoint.checkpoint(
                            _make_custom_forward(struct_block),
                            struct_x, struct_context, struct_t_mod, struct_freqs, struct_causal_grid,
                            use_reentrant=False,
                        )
                else:
                    struct_x = torch.utils.checkpoint.checkpoint(
                        _make_custom_forward(struct_block),
                        struct_x, struct_context, struct_t_mod, struct_freqs, struct_causal_grid,
                        use_reentrant=False,
                    )
            else:
                struct_x = struct_block(struct_x, struct_context, struct_t_mod, struct_freqs, causal_grid_size=struct_causal_grid)

        # Process remaining video layers (if video has more layers than struct)
        for layer_idx in range(min_layers, len(self.video_dit.blocks)):
            visual_block = self.video_dit.blocks[layer_idx]
            if self.use_gradient_checkpointing and self.training:
                visual_x = torch.utils.checkpoint.checkpoint(
                    _make_custom_forward(visual_block),
                    visual_x, visual_context, visual_t_mod, visual_freqs, visual_causal_grid,
                    use_reentrant=False,
                )
            else:
                visual_x = visual_block(visual_x, visual_context, visual_t_mod, visual_freqs, causal_grid_size=visual_causal_grid)

        # Process remaining struct layers
        for layer_idx in range(min_layers, len(self.struct_dit.blocks)):
            struct_block = self.struct_dit.blocks[layer_idx]
            if self.use_gradient_checkpointing and self.training:
                struct_x = torch.utils.checkpoint.checkpoint(
                    _make_custom_forward(struct_block),
                    struct_x, struct_context, struct_t_mod, struct_freqs, struct_causal_grid,
                    use_reentrant=False,
                )
            else:
                struct_x = struct_block(struct_x, struct_context, struct_t_mod, struct_freqs, causal_grid_size=struct_causal_grid)

        # Gather from context parallel
        if sp_enabled:
            visual_x = _sp_all_gather_avg(visual_x, sp_group, visual_pad_len)
            struct_x = _sp_all_gather_avg(struct_x, sp_group, struct_pad_len)

        return visual_x, struct_x

    # ============================================================
    # Training Step
    # ============================================================

    def training_step(
        self,
        video_latents: torch.Tensor,       # [B, C=16, T, H', W'] pre-extracted
        struct_latents: torch.Tensor,       # [B, D=128, T] pre-extracted
        audio_features: torch.Tensor,       # [B, T_audio, 1024] HuBERT features (conditioning only)
        caption: List[str],
        first_frame: torch.Tensor,          # [B, C, H, W] first frame image
        flame_params: Optional[torch.Tensor] = None,  # [B, T, 159] FLAME parameters
        video_fps: float = 25.0,
        cp_mesh: Optional[DeviceMesh] = None,
    ) -> dict:
        """
        One training step with Diffusion Forcing.

        Args:
            video_latents: Pre-extracted video VAE latents [B, 16, T', H', W']
            struct_latents: Pre-extracted LivePortrait struct [B, 128, T]
            audio_features: HuBERT features [B, T_audio, 1024] (conditioning, not diffused)
            caption: Text descriptions
            first_frame: Reference image [B, C, H, W]
            flame_params: Optional FLAME parameters for alignment loss
            video_fps: Video frame rate

        Returns:
            dict with loss terms
        """
        B = video_latents.shape[0]
        device = video_latents.device
        dtype = torch.bfloat16

        video_latents = video_latents.to(dtype=dtype, device=device)
        struct_latents = struct_latents.to(dtype=dtype, device=device)

        T_video = video_latents.shape[2]  # Temporal frames in latent space
        T_struct = struct_latents.shape[2]  # Should match T_video

        # --------------------------------------------------
        # 1. Encode text
        # --------------------------------------------------
        with torch.no_grad():
            context = self._get_t5_prompt_embeds(caption, device=device)

        # --------------------------------------------------
        # 2. Build first frame condition (same as MOVA)
        # --------------------------------------------------
        with torch.no_grad():
            _, _, _, H_latent, W_latent = video_latents.shape

            # Mask: first frame = 1, rest = 0
            msk = torch.zeros(B, 4, T_video, H_latent, W_latent, device=device, dtype=dtype)
            msk[:, :, 0, :, :] = 1

            # Encode first frame through VAE
            first_frame = first_frame.to(dtype=dtype, device=device)
            C_img = first_frame.shape[1]
            H_img = H_latent * 8  # Approximate original spatial size
            W_img = W_latent * 8

            vae_input = torch.concat([
                first_frame.unsqueeze(2),
                torch.zeros(B, C_img, video_latents.shape[2] * 4 - 1, H_img, W_img,
                            device=device, dtype=dtype)
            ], dim=2)

            with torch.autocast("cuda", dtype=dtype):
                y = self.video_vae.encode(vae_input).latent_dist.mode()
                y = self.normalize_video_latents(y)

            # Concat mask + first_frame condition: [B, 20, T', H', W']
            y = torch.concat([msk, y], dim=1)

        # --------------------------------------------------
        # 3. Sample per-frame noise (Diffusion Forcing)
        # --------------------------------------------------
        sigma_v, sigma_s = self.scheduler.sample_dual_per_frame_sigma(
            batch_size=B,
            num_frames_video=T_video,
            num_frames_struct=T_struct,
            device=device,
        )

        # Add noise per frame
        video_noise = torch.randn_like(video_latents)
        struct_noise = torch.randn_like(struct_latents)

        noisy_video = self.scheduler.add_noise_per_frame(video_latents, video_noise, sigma_v)
        noisy_struct = self.scheduler.add_noise_per_frame(struct_latents, struct_noise, sigma_s)

        # --------------------------------------------------
        # 4. Compute timestep embeddings from per-frame sigma
        # --------------------------------------------------
        # For AdaLN, we need a single timestep per sample.
        # Use mean sigma across frames as the conditioning signal.
        # TODO: Implement per-frame AdaLN (DualAdaLNZero) for finer control
        video_timestep = self.scheduler.sigma_to_timestep(sigma_v.mean(dim=1))  # [B]
        struct_timestep = self.scheduler.sigma_to_timestep(sigma_s.mean(dim=1))  # [B]

        model_dtype = torch.bfloat16
        with torch.autocast("cuda", dtype=torch.float32):
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

        # --------------------------------------------------
        # 5. Prepare tokens
        # --------------------------------------------------
        # Text embeddings
        visual_context_emb = self.video_dit.text_embedding(context)
        struct_context_emb = self.struct_dit.text_embedding(context)

        # Video: concat with condition and patchify
        visual_x = noisy_video.to(model_dtype)
        if self.video_dit.require_vae_embedding:
            visual_x = torch.cat([visual_x, y.to(model_dtype)], dim=1)

        visual_x, (t, h, w) = self.video_dit.patchify(visual_x)
        grid_size = (t, h, w)

        # Video RoPE
        visual_freqs = tuple(freq.to(visual_x.device) for freq in self.video_dit.freqs)
        visual_freqs = torch.cat([
            visual_freqs[0][:t].view(t, 1, 1, -1).expand(t, h, w, -1),
            visual_freqs[1][:h].view(1, h, 1, -1).expand(t, h, w, -1),
            visual_freqs[2][:w].view(1, 1, w, -1).expand(t, h, w, -1)
        ], dim=-1).reshape(t * h * w, 1, -1).to(visual_x.device)

        # Struct: tokenize
        struct_x, (f,) = self.struct_dit.tokenize(noisy_struct.to(model_dtype))

        # Struct RoPE
        struct_freqs = torch.cat([
            self.struct_dit.freqs[0][:f].view(f, -1).expand(f, -1),
            self.struct_dit.freqs[1][:f].view(f, -1).expand(f, -1),
            self.struct_dit.freqs[2][:f].view(f, -1).expand(f, -1),
        ], dim=-1).reshape(f, 1, -1).to(struct_x.device)

        # --------------------------------------------------
        # 6. Forward through dual tower
        # --------------------------------------------------
        visual_x, struct_x = self.forward_dual_tower(
            visual_x=visual_x,
            struct_x=struct_x,
            visual_context=visual_context_emb,
            struct_context=struct_context_emb,
            visual_t_mod=visual_t_mod,
            struct_t_mod=struct_t_mod,
            visual_freqs=visual_freqs,
            struct_freqs=struct_freqs,
            grid_size=grid_size,
            video_fps=video_fps,
            cp_mesh=cp_mesh,
        )

        # --------------------------------------------------
        # 7. Output heads
        # --------------------------------------------------
        video_pred = self.video_dit.head(visual_x, visual_t)
        video_pred = self.video_dit.unpatchify(video_pred, grid_size)  # [B, 16, T', H', W']

        struct_pred = self.struct_dit.head(struct_x, struct_t)
        struct_pred = self.struct_dit.detokenize(struct_pred, (f,))  # [B, 128, T]

        # --------------------------------------------------
        # 8. Compute losses
        # --------------------------------------------------
        # Flow matching targets: v = noise - sample
        video_target = video_noise - video_latents
        struct_target = struct_noise - struct_latents

        # L_video: MSE loss on video velocity prediction
        video_loss = F.mse_loss(video_pred.to(video_target.dtype), video_target)

        # L_struct: MSE loss on struct velocity prediction
        struct_loss = F.mse_loss(struct_pred.to(struct_target.dtype), struct_target)

        # L_flame: FLAME alignment loss (optional)
        flame_loss = torch.tensor(0.0, device=device)
        if flame_params is not None and self.loss_config.flame_weight > 0:
            # TODO: Implement FLAME alignment once struct decoder is built
            pass

        # L_lip_sync: Contrastive lip-sync loss (optional)
        lip_sync_loss = torch.tensor(0.0, device=device)
        if audio_features is not None and self.loss_config.lip_sync_weight > 0:
            # TODO: Implement lip-sync contrastive loss
            pass

        # Total loss
        total_loss = (
            self.loss_config.video_weight * video_loss +
            self.loss_config.struct_weight * struct_loss +
            self.loss_config.flame_weight * flame_loss +
            self.loss_config.lip_sync_weight * lip_sync_loss
        )

        return {
            "loss": total_loss,
            "video_loss": video_loss,
            "struct_loss": struct_loss,
            "flame_loss": flame_loss,
            "lip_sync_loss": lip_sync_loss,
            "sigma_v_mean": sigma_v.mean().item(),
            "sigma_s_mean": sigma_s.mean().item(),
        }

    def get_trainable_parameters(self, train_modules: List[str] = None):
        """Get trainable parameters."""
        if train_modules is None:
            train_modules = ["video_dit", "struct_dit", "dual_tower_bridge"]

        params = []
        for name in train_modules:
            if hasattr(self, name):
                module = getattr(self, name)
                params.extend(module.parameters())
                print(f"[DualForce] {name}: {sum(p.numel() for p in module.parameters())/1e6:.2f}M params")

        total = sum(p.numel() for p in params)
        print(f"[DualForce] Total trainable: {total/1e6:.2f}M params")
        return params

    def freeze_for_training(self, train_modules: List[str] = None):
        """Freeze all modules except those in train_modules."""
        if train_modules is None:
            train_modules = ["video_dit", "struct_dit", "dual_tower_bridge"]

        # Freeze everything first
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze training modules
        for name in train_modules:
            if hasattr(self, name):
                module = getattr(self, name)
                for param in module.parameters():
                    param.requires_grad = True


# ============================================================
# Factory Function (registered for DIFFUSION_PIPELINES)
# ============================================================

from mova.registry import DIFFUSION_PIPELINES


@DIFFUSION_PIPELINES.register_module()
def DualForceTrain_from_pretrained(
    from_pretrained: Optional[str] = None,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    # Config dicts for building from scratch
    video_dit_config: Optional[Dict] = None,
    struct_dit_config: Optional[Dict] = None,
    bridge_config: Optional[Dict] = None,
    scheduler_config: Optional[Dict] = None,
    loss_config: Optional[Dict] = None,
    # Frozen model paths
    vae_path: Optional[str] = None,
    text_encoder_path: Optional[str] = None,
):
    """Build DualForceTrain pipeline.

    Two modes:
    1. from_pretrained is set: load a saved DualForce checkpoint
    2. from_pretrained is None: build from scratch using config dicts
       (requires vae_path and text_encoder_path for frozen components)
    """
    import os

    if from_pretrained is not None:
        # Load from saved checkpoint
        use_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"

        if use_fsdp or device == "cpu":
            model = DualForceTrain.from_pretrained(
                from_pretrained,
                torch_dtype=torch_dtype,
            )
        else:
            model = DualForceTrain.from_pretrained(
                from_pretrained,
                device_map=device,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
    else:
        # Build from scratch
        print("[DualForce] Building model from scratch...")

        # Build video DiT
        video_dit_cfg = video_dit_config or {}
        video_dit = WanModel(**video_dit_cfg).to(torch_dtype)
        print(f"[DualForce] Video DiT: {sum(p.numel() for p in video_dit.parameters())/1e6:.1f}M params")

        # Build struct DiT
        struct_dit_cfg = struct_dit_config or {}
        struct_dit = WanStructModel(**struct_dit_cfg).to(torch_dtype)
        print(f"[DualForce] Struct DiT: {sum(p.numel() for p in struct_dit.parameters())/1e6:.1f}M params")

        # Build bridge
        bridge_cfg = bridge_config or {}
        dual_tower_bridge = DualTowerConditionalBridge(**bridge_cfg).to(torch_dtype)
        print(f"[DualForce] Bridge: {sum(p.numel() for p in dual_tower_bridge.parameters())/1e6:.1f}M params")

        # Build scheduler
        scheduler_cfg = scheduler_config or {}
        scheduler = DiffusionForcingScheduler(**scheduler_cfg)

        # Load frozen VAE
        if vae_path:
            video_vae = AutoencoderKLWan.from_pretrained(
                vae_path, torch_dtype=torch_dtype
            )
        else:
            raise ValueError("vae_path is required when building from scratch")

        # Load frozen text encoder + tokenizer
        if text_encoder_path:
            text_encoder = UMT5EncoderModel.from_pretrained(
                text_encoder_path, torch_dtype=torch_dtype
            )
            tokenizer = T5TokenizerFast.from_pretrained(text_encoder_path)
        else:
            raise ValueError("text_encoder_path is required when building from scratch")

        model = DualForceTrain(
            video_vae=video_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            video_dit=video_dit,
            struct_dit=struct_dit,
            dual_tower_bridge=dual_tower_bridge,
            loss_config=loss_config,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    model.use_gradient_checkpointing = use_gradient_checkpointing
    model.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
    return model
