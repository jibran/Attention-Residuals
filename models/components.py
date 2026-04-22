"""Shared neural-network building blocks.

Provides:

* :class:`RMSNorm`          — Root-mean-square layer normalisation.
* :class:`SwiGLU`           — SwiGLU feed-forward block (Llama-style MLP).
* :class:`RotaryEmbedding`  — Rotary position embeddings (RoPE).
* :class:`CausalSelfAttention` — Multi-head causal self-attention with RoPE.

All modules follow the Google docstring convention and are written to be
framework-agnostic beyond requiring PyTorch >= 2.0.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (no centring, no bias).

    Defined as:  ``x / rms(x) * weight``  where  ``rms(x) = sqrt(mean(x²) + ε)``.

    Preferred over LayerNorm in modern LLMs (Llama, Mistral, Kimi) because it
    omits the mean subtraction step — cheaper and equally effective in practice.

    Args:
        dim: Feature dimension to normalise over (last axis).
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the last dimension of ``x``.

        Args:
            x: Input tensor of arbitrary shape ``(..., dim)``.

        Returns:
            Normalised tensor with the same shape as ``x``.
        """
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block.

    Implements the gated variant of the Swish (SiLU) activation::

        SwiGLU(x) = down( silu(gate(x)) ⊙ up(x) )

    The hidden dimension is scaled by 8/3 (rounded) so that the total
    parameter count matches a conventional 4× FFN when both ``gate`` and ``up``
    projections are counted.

    Args:
        dim: Input and output feature dimension.
        hidden_dim: Inner dimension; defaults to ``int(dim * 8 / 3)``.
        dropout: Dropout probability applied after the gate.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the SwiGLU transformation.

        Args:
            x: Input tensor ``(B, T, dim)``.

        Returns:
            Output tensor ``(B, T, dim)``.
        """
        return self.down(self.drop(F.silu(self.gate(x))) * self.up(x))


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE).

    Applies a rotation in 2D sub-spaces of the head dimension to inject
    relative position information without any learned parameters.

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
    Position Embedding" (2021).

    Args:
        head_dim: Dimension of each attention head (must be even).
        max_seq_len: Maximum sequence length for pre-computation.
        base: Base for the geometric sequence of frequencies (default 10 000).
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 512,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE."
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)  # (T, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors.

        Args:
            q: Query tensor ``(B, n_heads, T, head_dim)``.
            k: Key tensor   ``(B, n_heads, T, head_dim)``.

        Returns:
            Tuple ``(q_rot, k_rot)`` with rotations applied; same shapes.
        """
        T = q.shape[2]
        if T > self.cos_cache.shape[0]:
            self._build_cache(T)
        cos = self.cos_cache[:T].unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
        sin = self.sin_cache[:T].unsqueeze(0).unsqueeze(0)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Causal Self-Attention
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE.

    Uses :func:`torch.nn.functional.scaled_dot_product_attention` (Flash
    Attention when available via PyTorch 2.x) with ``is_causal=True``.

    Args:
        dim: Model hidden dimension.
        heads: Number of attention heads.
        head_dim: Dimension per head.
        max_seq_len: Maximum sequence length (passed to RoPE).
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim

        self.qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.proj = nn.Linear(inner, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(head_dim, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute causal self-attention.

        Args:
            x: Input tensor ``(B, T, dim)``.

        Returns:
            Output tensor ``(B, T, dim)``.
        """
        B, T, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)  # 3 × (B,T,inner)
        q, k, v = (t.view(B, T, self.heads, self.head_dim).transpose(1, 2) for t in qkv)
        q, k = self.rope(q, k)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop.p if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(out)
