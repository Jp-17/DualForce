"""
DualForce 3D Structure DiT Model.

Forked from WanAudioModel (wan_audio_dit.py), adapted to process
LivePortrait motion features as a parallel modality tower.

Key differences from WanAudioModel:
- Input: LivePortrait struct latents (128-dim per frame) instead of DAC audio latents
- Tokenization: MLP projection instead of Conv1d patchification (no stride)
- RoPE: 1D temporal (same as audio), but at video FPS (not audio FPS)
- No audio VAE dependency
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

from .wan_video_dit import DiTBlock


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 16384, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    pos = torch.arange(end, dtype=torch.float64, device=freqs.device)
    freqs = torch.outer(pos, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0):
    """Precompute 1D RoPE frequencies, split into 3 chunks for compatibility with bridge."""
    f_freqs_cis = precompute_freqs_cis(dim, end, theta)
    return f_freqs_cis.chunk(3, dim=-1)


class Head(nn.Module):
    """Output head that projects hidden states back to struct latent space."""
    def __init__(self, dim: int, out_dim: int, eps: float):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class StructTokenProjector(nn.Module):
    """
    Project LivePortrait struct latents to DiT hidden dimension.

    Input: [B, T, D_struct] where D_struct=128 (LivePortrait motion features)
    Output: [B, T * N_tokens, dim] where N_tokens is number of tokens per frame

    Unlike audio DiT which uses Conv1d with stride for downsampling,
    struct tokens use MLP projection with optional token expansion.
    """
    def __init__(self, in_dim: int, dim: int, n_tokens_per_frame: int = 1):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.n_tokens_per_frame = n_tokens_per_frame

        if n_tokens_per_frame == 1:
            # Simple projection: 1 token per frame
            self.proj = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, dim),
                nn.GELU(approximate='tanh'),
                nn.Linear(dim, dim),
            )
        else:
            # Expand to multiple tokens per frame
            self.proj = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, dim * n_tokens_per_frame),
                nn.GELU(approximate='tanh'),
                nn.Linear(dim * n_tokens_per_frame, dim * n_tokens_per_frame),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D_struct] struct latents
        Returns:
            [B, T * N_tokens, dim] projected tokens
        """
        B, T, _ = x.shape
        x = self.proj(x)  # [B, T, dim * N_tokens]
        if self.n_tokens_per_frame > 1:
            x = x.view(B, T * self.n_tokens_per_frame, self.dim)
        return x


class WanStructModel(ModelMixin, ConfigMixin):
    """
    3D Structure DiT for DualForce.

    Processes LivePortrait motion features as a parallel modality tower.
    Architecture mirrors WanAudioModel but with MLP tokenization instead
    of Conv1d patchification.
    """
    _repeated_blocks = ("DiTBlock",)

    @register_to_config
    def __init__(
        self,
        dim: int = 1536,
        in_dim: int = 128,         # LivePortrait struct latent dim
        ffn_dim: int = 6144,
        out_dim: int = 128,        # Output same as input dim
        text_dim: int = 4096,
        freq_dim: int = 256,
        eps: float = 1e-6,
        num_heads: int = 12,
        num_layers: int = 20,
        n_tokens_per_frame: int = 1,  # Tokens per frame (1 = simple, 8 = richer)
        has_image_input: bool = False,
        has_image_pos_emb: bool = False,
        require_clip_embedding: bool = True,
        causal_temporal: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.n_tokens_per_frame = n_tokens_per_frame
        self.causal_temporal = causal_temporal

        # Struct token projector (replaces Conv1d patch_embedding)
        self.struct_projector = StructTokenProjector(in_dim, dim, n_tokens_per_frame)

        # Text embedding
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )

        # Timestep embedding (for per-frame noise level in Diffusion Forcing)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # DiT blocks (reuse from video DiT)
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])

        # Output head (no patch_size needed since we use MLP projection)
        self.head = Head(dim, out_dim * n_tokens_per_frame, eps)

        # 1D RoPE for temporal position encoding
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_1d(head_dim)

        # Optional CLIP image embedding
        if has_image_input:
            self.img_emb = nn.Sequential(
                nn.LayerNorm(1280),
                nn.Linear(1280, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )
        self.has_image_pos_emb = has_image_pos_emb

    def tokenize(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int]]:
        """
        Project struct latents to token sequence.

        Args:
            x: [B, T, D_struct] or [B, D_struct, T] struct latents
        Returns:
            tokens: [B, T * N_tokens, dim]
            grid_size: (T * N_tokens,) for compatibility with bridge
        """
        # Ensure format is [B, T, D_struct]
        if x.dim() == 3 and x.shape[1] == self.struct_projector.in_dim:
            x = x.transpose(1, 2)  # [B, D, T] -> [B, T, D]

        tokens = self.struct_projector(x)  # [B, T * N_tokens, dim]
        seq_len = tokens.shape[1]
        return tokens, (seq_len,)

    def detokenize(self, x: torch.Tensor, grid_size: Tuple[int]) -> torch.Tensor:
        """
        Convert tokens back to struct latent format.

        Args:
            x: [B, T * N_tokens, out_dim * N_tokens] from head
            grid_size: (T * N_tokens,)
        Returns:
            [B, out_dim, T] struct latents (channel-first for compatibility)
        """
        B = x.shape[0]
        seq_len = grid_size[0]

        if self.n_tokens_per_frame > 1:
            T = seq_len // self.n_tokens_per_frame
            # Reshape: [B, T*N, out_dim*N] -> [B, T, N, out_dim, N] -> average over token dim
            x = x.view(B, T, self.n_tokens_per_frame, -1)
            # Average pool over tokens per frame
            x = x.mean(dim=2)  # [B, T, out_dim]

        # Transpose to channel-first: [B, T, D] -> [B, D, T]
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        """
        Args:
            x: [B, D_struct, T] or [B, T, D_struct] struct latents
            timestep: [B] diffusion timestep
            context: [B, seq_len, text_dim] text embeddings
            clip_feature: Optional CLIP features for image conditioning
        """
        # Timestep embedding
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # Text embedding
        context = self.text_embedding(context)

        # Optional CLIP conditioning
        if self.has_image_input and clip_feature is not None:
            clip_emb = self.img_emb(clip_feature)
            context = torch.cat([clip_emb, context], dim=1)

        # Tokenize struct latents
        x, (f,) = self.tokenize(x)

        # Compute 1D RoPE frequencies
        freqs = torch.cat([
            self.freqs[0][:f].view(f, -1).expand(f, -1),
            self.freqs[1][:f].view(f, -1).expand(f, -1),
            self.freqs[2][:f].view(f, -1).expand(f, -1),
        ], dim=-1).reshape(f, 1, -1).to(x.device)

        # Forward through DiT blocks
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        # Output head
        x = self.head(x, t)

        # Detokenize back to struct latent format
        x = self.detokenize(x, (f,))

        return x
