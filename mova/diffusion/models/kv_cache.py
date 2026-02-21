"""
MultiModalKVCache for DualForce Autoregressive Inference.

Stores per-layer K/V tensors for both video and struct towers,
enabling efficient autoregressive generation without recomputing
attention over all previously generated frames.

Architecture:
- Each DiT layer has its own K/V cache for self-attention
- Video and struct towers have separate caches
- Supports sliding window (for memory-bounded inference)
- Thread-safe cache update for context parallel

Usage during inference:
    cache = MultiModalKVCache(num_layers=20, num_heads=12, head_dim=128)

    # First frame: full forward, populate cache
    out_1 = model(x_1, ..., kv_cache=cache, cache_mode='populate')

    # Subsequent frames: only compute new frame, read from cache
    out_2 = model(x_2, ..., kv_cache=cache, cache_mode='extend')
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field


@dataclass
class LayerKVCache:
    """KV cache for a single attention layer."""
    k: Optional[torch.Tensor] = None  # [B, S_cached, num_heads * head_dim]
    v: Optional[torch.Tensor] = None  # [B, S_cached, num_heads * head_dim]
    seq_len: int = 0  # Current cached sequence length

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V and return full K/V for attention computation.

        Args:
            new_k: [B, S_new, dim] new key states
            new_v: [B, S_new, dim] new value states

        Returns:
            full_k: [B, S_cached + S_new, dim]
            full_v: [B, S_cached + S_new, dim]
        """
        if self.k is None:
            self.k = new_k
            self.v = new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=1)
            self.v = torch.cat([self.v, new_v], dim=1)

        self.seq_len = self.k.shape[1]
        return self.k, self.v

    def get(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached K/V tensors."""
        return self.k, self.v

    def trim(self, max_len: int):
        """Trim cache to maximum length (sliding window)."""
        if self.k is not None and self.k.shape[1] > max_len:
            self.k = self.k[:, -max_len:]
            self.v = self.v[:, -max_len:]
            self.seq_len = self.k.shape[1]

    def clear(self):
        """Clear the cache."""
        self.k = None
        self.v = None
        self.seq_len = 0


class MultiModalKVCache:
    """
    Multi-modal KV cache for DualForce autoregressive inference.

    Maintains separate caches for video and struct towers.
    Each tower has per-layer caches for self-attention K/V states.
    """

    def __init__(
        self,
        num_video_layers: int = 20,
        num_struct_layers: int = 20,
        max_cache_len: Optional[int] = None,  # None = unlimited, else sliding window
    ):
        self.num_video_layers = num_video_layers
        self.num_struct_layers = num_struct_layers
        self.max_cache_len = max_cache_len

        # Per-layer caches
        self.video_caches: List[LayerKVCache] = [
            LayerKVCache() for _ in range(num_video_layers)
        ]
        self.struct_caches: List[LayerKVCache] = [
            LayerKVCache() for _ in range(num_struct_layers)
        ]

        # Track total generated frames
        self.num_generated_frames = 0

    def update_video(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update video tower cache at given layer and return full K/V."""
        full_k, full_v = self.video_caches[layer_idx].update(new_k, new_v)
        if self.max_cache_len is not None:
            self.video_caches[layer_idx].trim(self.max_cache_len)
        return full_k, full_v

    def update_struct(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update struct tower cache at given layer and return full K/V."""
        full_k, full_v = self.struct_caches[layer_idx].update(new_k, new_v)
        if self.max_cache_len is not None:
            self.struct_caches[layer_idx].trim(self.max_cache_len)
        return full_k, full_v

    def get_video(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached video K/V for a layer."""
        return self.video_caches[layer_idx].get()

    def get_struct(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached struct K/V for a layer."""
        return self.struct_caches[layer_idx].get()

    def increment_frame(self):
        """Mark that a new frame has been generated."""
        self.num_generated_frames += 1

    def clear(self):
        """Clear all caches."""
        for cache in self.video_caches:
            cache.clear()
        for cache in self.struct_caches:
            cache.clear()
        self.num_generated_frames = 0

    @property
    def video_seq_len(self) -> int:
        """Current total cached video sequence length."""
        if self.video_caches and self.video_caches[0].k is not None:
            return self.video_caches[0].seq_len
        return 0

    @property
    def struct_seq_len(self) -> int:
        """Current total cached struct sequence length."""
        if self.struct_caches and self.struct_caches[0].k is not None:
            return self.struct_caches[0].seq_len
        return 0

    def memory_usage_bytes(self) -> int:
        """Estimate total memory usage of the cache."""
        total = 0
        for cache in self.video_caches + self.struct_caches:
            if cache.k is not None:
                total += cache.k.nelement() * cache.k.element_size()
            if cache.v is not None:
                total += cache.v.nelement() * cache.v.element_size()
        return total

    def memory_usage_mb(self) -> float:
        """Estimate total memory usage in MB."""
        return self.memory_usage_bytes() / (1024 * 1024)


class CachedSelfAttention(nn.Module):
    """
    SelfAttention wrapper that supports KV-cache for autoregressive inference.

    During inference with cache:
    - Query is computed only for new tokens
    - Key/Value are computed for new tokens and concatenated with cached K/V
    - Attention is computed: Q_new attends to all K (cached + new)
    """

    def __init__(self, self_attn: nn.Module):
        """Wrap an existing SelfAttention module."""
        super().__init__()
        self.self_attn = self_attn

    @property
    def dim(self):
        return self.self_attn.dim

    @property
    def num_heads(self):
        return self.self_attn.num_heads

    @property
    def head_dim(self):
        return self.self_attn.head_dim

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        kv_cache: Optional[LayerKVCache] = None,
        cache_mode: str = 'none',  # 'none', 'populate', 'extend'
    ) -> torch.Tensor:
        """
        Forward with optional KV-cache.

        Args:
            x: [B, S, dim] input tokens
            freqs: RoPE frequencies
            kv_cache: Optional cache for this layer
            cache_mode:
                'none': Standard forward, no caching
                'populate': Compute full attention, store K/V in cache
                'extend': Only new tokens as Q, full K/V from cache + new
        """
        from .wan_video_dit import rope_apply_head_dim, flash_attention
        from torch.distributed.tensor import DTensor

        q = self.self_attn.norm_q(self.self_attn.q(x))
        k = self.self_attn.norm_k(self.self_attn.k(x))
        v = self.self_attn.v(x)

        if isinstance(freqs, DTensor):
            freqs = freqs.to_local()

        q = rope_apply_head_dim(q, freqs, self.head_dim)
        k = rope_apply_head_dim(k, freqs, self.head_dim)

        if cache_mode == 'none' or kv_cache is None:
            # Standard forward
            x = flash_attention(q, k, v, self.num_heads)
        elif cache_mode == 'populate':
            # Full attention, store K/V
            kv_cache.update(k, v)
            x = flash_attention(q, k, v, self.num_heads)
        elif cache_mode == 'extend':
            # Extend cache with new K/V, Q only for new tokens
            full_k, full_v = kv_cache.update(k, v)
            # Q: new tokens only, K/V: all cached + new
            # Need to build RoPE-adjusted K for cached tokens too
            # Since we already applied RoPE before caching, this works directly
            x = flash_attention(q, full_k, full_v, self.num_heads)

        return self.self_attn.o(x)
