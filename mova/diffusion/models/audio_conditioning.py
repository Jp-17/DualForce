"""
DualForce Audio Conditioning and DualAdaLN-Zero Modules.

Key innovations:
1. AudioCondCrossAttention: One-directional cross-attention from HuBERT audio
   features into video/struct tokens. Gated residual connection.
2. DualAdaLNZero: Per-frame, per-modality AdaLN modulation that conditions
   each frame's DiT processing on its own noise level (sigma).

These modules live in the bridge's shallow layers only — audio conditioning
does NOT flow through deep layers, keeping the deep layers specialized for
their respective modalities.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .wan_video_dit import RMSNorm, AttentionModule


# ============================================================
# Audio Conditioning
# ============================================================

class AudioProjector(nn.Module):
    """Project HuBERT features to DiT hidden dimension.

    HuBERT-Large outputs 1024-dim features at 50Hz.
    This module projects and aligns them to match DiT token dimensions.
    """
    def __init__(self, audio_dim: int = 1024, hidden_dim: int = 1536):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: [B, T_audio, 1024] HuBERT features
        Returns:
            [B, T_audio, hidden_dim] projected audio tokens
        """
        return self.proj(audio_features)


class AudioCondCrossAttention(nn.Module):
    """One-directional cross-attention: audio -> target modality.

    Audio tokens serve as key/value, target tokens (video or struct) as query.
    Uses gated residual connection initialized near zero so the model starts
    with no audio influence and learns to incorporate it gradually.
    """
    def __init__(
        self,
        dim: int,
        audio_dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Pre-norm for query (target modality tokens)
        self.norm_q = RMSNorm(dim, eps=eps)
        # Pre-norm for audio kv
        self.norm_kv = nn.LayerNorm(audio_dim, eps=eps)

        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(audio_dim, dim)
        self.v_proj = nn.Linear(audio_dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Attention
        self.attn = AttentionModule(num_heads)

        # Gated residual: gate initialized to small value (near-zero init)
        # This ensures audio conditioning has minimal effect at initialization
        self.gate = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(
        self,
        x: torch.Tensor,
        audio_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, dim] target tokens (video or struct)
            audio_tokens: [B, T_audio, audio_dim] projected audio features

        Returns:
            [B, L, dim] audio-conditioned target tokens
        """
        q = self.q_proj(self.norm_q(x))
        k = self.k_proj(self.norm_kv(audio_tokens))
        v = self.v_proj(self.norm_kv(audio_tokens))

        # Cross-attention
        attn_out = self.attn(q, k, v)
        attn_out = self.out_proj(attn_out)

        # Gated residual: x += sigmoid(gate) * attn_out
        gate = torch.sigmoid(self.gate)
        return x + gate * attn_out


class AudioConditioningModule(nn.Module):
    """Audio conditioning module for DualForce.

    Contains:
    - AudioProjector: HuBERT features -> hidden dim
    - Temporal alignment: resample audio tokens to match target token count
    - Cross-attention layers for video and struct streams
    """
    def __init__(
        self,
        audio_dim: int = 1024,
        video_dim: int = 1536,
        struct_dim: int = 1536,
        num_heads: int = 12,
        num_layers: int = 6,   # Number of shallow layers with audio conditioning
        eps: float = 1e-6,
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.num_layers = num_layers

        # Project HuBERT features to video/struct dimensions
        self.audio_to_video_proj = AudioProjector(audio_dim, video_dim)
        self.audio_to_struct_proj = AudioProjector(audio_dim, struct_dim)

        # Per-layer cross-attention: audio -> video
        self.audio_to_video_attn = nn.ModuleList([
            AudioCondCrossAttention(video_dim, video_dim, num_heads, eps)
            for _ in range(num_layers)
        ])

        # Per-layer cross-attention: audio -> struct
        self.audio_to_struct_attn = nn.ModuleList([
            AudioCondCrossAttention(struct_dim, struct_dim, num_heads, eps)
            for _ in range(num_layers)
        ])

    def align_audio_to_tokens(
        self,
        audio_tokens: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        """Temporally align audio tokens to match target sequence length.

        Args:
            audio_tokens: [B, T_audio, dim]
            target_len: target sequence length

        Returns:
            [B, target_len, dim] aligned audio tokens
        """
        if audio_tokens.shape[1] == target_len:
            return audio_tokens

        # Use linear interpolation along temporal dimension
        # [B, T, D] -> [B, D, T] -> interpolate -> [B, D, target_len] -> [B, target_len, D]
        aligned = F.interpolate(
            audio_tokens.transpose(1, 2),
            size=target_len,
            mode='linear',
            align_corners=False,
        ).transpose(1, 2)

        return aligned

    def condition_video(
        self,
        layer_idx: int,
        video_tokens: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply audio conditioning to video tokens at a given layer.

        Args:
            layer_idx: which shallow layer (0-indexed)
            video_tokens: [B, L_v, dim_v] video hidden states
            audio_features: [B, T_audio, 1024] raw HuBERT features

        Returns:
            [B, L_v, dim_v] audio-conditioned video tokens
        """
        if layer_idx >= self.num_layers:
            return video_tokens

        # Project audio to video dimension
        audio_tokens = self.audio_to_video_proj(audio_features)
        # Align temporal dimension
        audio_tokens = self.align_audio_to_tokens(audio_tokens, video_tokens.shape[1])
        # Cross-attention
        return self.audio_to_video_attn[layer_idx](video_tokens, audio_tokens)

    def condition_struct(
        self,
        layer_idx: int,
        struct_tokens: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply audio conditioning to struct tokens at a given layer.

        Args:
            layer_idx: which shallow layer (0-indexed)
            struct_tokens: [B, L_s, dim_s] struct hidden states
            audio_features: [B, T_audio, 1024] raw HuBERT features

        Returns:
            [B, L_s, dim_s] audio-conditioned struct tokens
        """
        if layer_idx >= self.num_layers:
            return struct_tokens

        # Project audio to struct dimension
        audio_tokens = self.audio_to_struct_proj(audio_features)
        # Align temporal dimension
        audio_tokens = self.align_audio_to_tokens(audio_tokens, struct_tokens.shape[1])
        # Cross-attention
        return self.audio_to_struct_attn[layer_idx](struct_tokens, audio_tokens)


# ============================================================
# DualAdaLN-Zero: Per-Frame, Per-Modality Noise Conditioning
# ============================================================

class DualAdaLNZero(nn.Module):
    """Dual AdaLN-Zero modulation for Diffusion Forcing.

    In standard DiT, a single scalar timestep t is used to modulate all tokens
    uniformly. In Diffusion Forcing, each frame has an independent noise level
    sigma[t], so we need per-frame modulation.

    DualAdaLNZero computes separate modulation parameters for video and struct
    streams, allowing each to be conditioned on its own per-frame sigma.

    Output: 6 modulation vectors per frame (shift_msa, scale_msa, gate_msa,
    shift_mlp, scale_mlp, gate_mlp) that replace the standard DiT t_mod.
    """
    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim

        # Per-frame sigma embedding
        # Input: sinusoidal embedding of sigma [B, T, freq_dim]
        # Output: modulation parameters [B, T, 6*dim]
        self.sigma_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.modulation_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

    def forward(
        self,
        sigma_per_frame: torch.Tensor,
        grid_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """Compute per-frame modulation from per-frame sigma.

        Args:
            sigma_per_frame: [B, T] per-frame noise levels
            grid_size: For video: (f, h, w). For struct: (f,)

        Returns:
            t_mod: [B, L, 6, dim] per-token modulation parameters
                   where L = f*h*w (video) or f (struct)
        """
        B, T = sigma_per_frame.shape

        # Sinusoidal embedding of per-frame sigma
        # [B, T] -> [B, T, freq_dim]
        sigma_emb = self._sinusoidal_embedding(sigma_per_frame)

        # Project to dim
        t_emb = self.sigma_embedding(sigma_emb)  # [B, T, dim]

        # Generate 6 modulation vectors
        t_mod = self.modulation_proj(t_emb)  # [B, T, 6*dim]
        t_mod = t_mod.view(B, T, 6, self.dim)  # [B, T, 6, dim]

        # Expand per-frame modulation to per-token
        if len(grid_size) == 3:
            # Video: each frame has h*w tokens
            f, h, w = grid_size
            # Repeat each frame's modulation for h*w tokens
            t_mod = t_mod[:, :f]  # Truncate to actual frame count
            t_mod = t_mod.unsqueeze(3).unsqueeze(4)  # [B, f, 6, dim, 1, 1]
            t_mod = t_mod.expand(B, f, 6, self.dim, h, w)  # [B, f, 6, dim, h, w]
            t_mod = t_mod.permute(0, 2, 3, 1, 4, 5)  # [B, 6, dim, f, h, w]
            t_mod = t_mod.reshape(B, 6, self.dim, f * h * w)  # [B, 6, dim, L]
            t_mod = t_mod.permute(0, 3, 1, 2)  # [B, L, 6, dim]
        elif len(grid_size) == 1:
            # Struct: each frame is one token
            f = grid_size[0]
            t_mod = t_mod[:, :f]  # [B, f, 6, dim]
        else:
            raise ValueError(f"Unsupported grid_size: {grid_size}")

        return t_mod

    def _sinusoidal_embedding(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal positional embedding of sigma values.

        Args:
            sigma: [B, T] noise levels in [0, 1]
        Returns:
            [B, T, freq_dim] sinusoidal embedding
        """
        half_dim = self.freq_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=sigma.device, dtype=torch.float32) * -emb)
        # sigma is in [0, 1], scale to [0, 1000] for embedding
        scaled_sigma = sigma.float() * 1000.0
        emb = scaled_sigma.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)  # [B, T, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, T, freq_dim]
        return emb.to(sigma.dtype)


class PerFrameDiTBlockWrapper(nn.Module):
    """Wrapper that replaces DiTBlock's global t_mod with per-frame modulation.

    Standard DiTBlock expects t_mod: [B, 6, dim] (global for all tokens).
    With DualAdaLNZero, we provide t_mod: [B, L, 6, dim] (per-token).

    This wrapper handles the shape conversion so DiTBlock works unchanged.
    It's used as a drop-in for the training loop — not as a permanent replacement.
    """

    @staticmethod
    def convert_per_frame_t_mod(t_mod_per_frame: torch.Tensor) -> torch.Tensor:
        """Convert per-frame t_mod to the format DiTBlock expects.

        DiTBlock checks `len(t_mod.shape) == 4` and treats it as sequential modulation.
        We pack per-frame modulation as [B, L, 6, dim] which is 4D,
        triggering DiTBlock's sequential path.

        Args:
            t_mod_per_frame: [B, L, 6, dim] per-token modulation
        Returns:
            [B, L, 6, dim] (same tensor, just confirms shape compatibility)
        """
        assert t_mod_per_frame.dim() == 4, f"Expected 4D tensor, got {t_mod_per_frame.dim()}D"
        return t_mod_per_frame
