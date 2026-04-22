"""Shared neural-network building blocks.

Provides:

* :class:`RMSNorm`             — Root-mean-square layer normalisation.
* :class:`SwiGLU`              — SwiGLU feed-forward block (Llama-style MLP).
* :class:`RotaryEmbedding`     — Rotary position embeddings (RoPE).
* :class:`KVCache`             — Per-layer key/value cache for autoregressive inference.
* :class:`CausalSelfAttention` — Multi-head causal self-attention with RoPE and
                                  optional KV-cache support.

All modules follow the Google docstring convention and are written to be
framework-agnostic beyond requiring PyTorch >= 2.0.
"""

from __future__ import annotations

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
        hidden_dim: int | None = None,
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
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors.

        Args:
            q: Query tensor ``(B, n_heads, T_q, head_dim)``.
            k: Key tensor   ``(B, n_heads, T_k, head_dim)``.
            offset: Starting position index.  During cached inference this is
                the number of tokens already in the KV cache, so that new
                tokens receive correct absolute position encodings.

        Returns:
            Tuple ``(q_rot, k_rot)`` with rotations applied; same shapes.
        """
        T_q = q.shape[2]
        T_k = k.shape[2]
        max_pos = offset + max(T_q, T_k)
        if max_pos > self.cos_cache.shape[0]:
            self._build_cache(max_pos)

        # Positions for new keys/queries start at `offset`
        cos_q = self.cos_cache[offset : offset + T_q].unsqueeze(0).unsqueeze(0)
        sin_q = self.sin_cache[offset : offset + T_q].unsqueeze(0).unsqueeze(0)
        cos_k = self.cos_cache[offset : offset + T_k].unsqueeze(0).unsqueeze(0)
        sin_k = self.sin_cache[offset : offset + T_k].unsqueeze(0).unsqueeze(0)

        q_rot = q * cos_q + self._rotate_half(q) * sin_q
        k_rot = k * cos_k + self._rotate_half(k) * sin_k
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------


class KVCache:
    """Per-layer key/value cache for autoregressive inference.

    During autoregressive decoding, re-computing keys and values for all
    previously seen tokens at every new step is redundant.  :class:`KVCache`
    stores the accumulated ``(k, v)`` tensors for one attention layer and
    appends new entries at each decode step.

    A :class:`KVCache` instance is **not** an ``nn.Module``; it carries no
    learnable parameters and is never serialised in checkpoints.  It lives only
    for the duration of a single generation call and must be reset between
    independent sequences.

    Attributes:
        k_cache: Accumulated key tensor   ``(B, n_heads, T_cached, head_dim)``
            or ``None`` before the first update.
        v_cache: Accumulated value tensor ``(B, n_heads, T_cached, head_dim)``
            or ``None`` before the first update.

    Example::

        cache = KVCache()
        # --- decode step 1 ---
        k_full, v_full = cache.update(k_new, v_new)   # k_full == k_new (T_cached=0)
        # --- decode step 2 ---
        k_full, v_full = cache.update(k_new2, v_new2) # k_full has T=2
        cache.clear()   # reset between sequences
    """

    def __init__(self) -> None:
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def length(self) -> int:
        """Number of token positions currently stored in the cache.

        Returns:
            Cached sequence length (0 when cache is empty).
        """
        return 0 if self.k_cache is None else self.k_cache.shape[2]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values and return the full accumulated tensors.

        The new tensors are concatenated along the sequence dimension (dim=2)
        with any previously cached entries.

        Args:
            k_new: New key tensor   ``(B, n_heads, T_new, head_dim)``.
            v_new: New value tensor ``(B, n_heads, T_new, head_dim)``.

        Returns:
            Tuple ``(k_full, v_full)`` containing all cached tokens including
            the new ones, each of shape ``(B, n_heads, T_cached + T_new, head_dim)``.
        """
        if self.k_cache is None:
            self.k_cache = k_new
            self.v_cache = v_new
        else:
            self.k_cache = torch.cat([self.k_cache, k_new], dim=2)
            self.v_cache = torch.cat([self.v_cache, v_new], dim=2)
        return self.k_cache, self.v_cache

    def clear(self) -> None:
        """Reset the cache, discarding all stored keys and values.

        Call between independent generation sequences to prevent cross-
        contamination.
        """
        self.k_cache = None
        self.v_cache = None

    def reset(self) -> None:
        """Alias for :meth:`clear`. Discards all stored keys and values."""
        self.clear()


# ---------------------------------------------------------------------------
# Causal Self-Attention
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional KV-cache.

    Uses :func:`torch.nn.functional.scaled_dot_product_attention` (Flash
    Attention when available via PyTorch 2.x).

    KV-cache behaviour
    ------------------
    When a :class:`KVCache` is supplied to :meth:`forward`:

    * Only the **new** token(s) in ``x`` are projected to Q, K, V.
    * The new K and V are appended to the cache via :meth:`KVCache.update`.
    * Attention is computed with Q over the **full** accumulated K/V history.
    * RoPE is applied with the correct position offset so that new tokens
      receive absolute positions ``[cache.length, cache.length + T_new)``.
    * ``is_causal=False`` is passed to SDPA because the causal mask is already
      implicit: Q only attends to cached tokens from earlier steps plus itself,
      and ``scaled_dot_product_attention`` would otherwise mask out valid cached
      positions.

    The cache is **never** used during training (``model.training == True``).
    Pass ``kv_cache=None`` (the default) for all training and prefill steps.

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

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Compute causal self-attention, optionally with a KV cache.

        Args:
            x: Input tensor ``(B, T, dim)``.  During cached decoding ``T``
                is typically 1 (one new token per step).
            kv_cache: Optional :class:`KVCache` instance for this layer.
                When provided and ``not self.training``, new K/V are appended
                to the cache and attention spans the full history.  Ignored
                (treated as ``None``) during training.

        Returns:
            Output tensor ``(B, T, dim)``.
        """
        B, T, _ = x.shape

        # Project to Q, K, V for the *new* tokens only
        qkv = self.qkv(x).chunk(3, dim=-1)  # 3 × (B, T, inner)
        q, k, v = (t.view(B, T, self.heads, self.head_dim).transpose(1, 2) for t in qkv)

        use_cache = kv_cache is not None

        if use_cache:
            # Apply RoPE with the current cache offset so new tokens get the
            # correct absolute positions.
            offset = kv_cache.length
            q, k = self.rope(q, k, offset=offset)
            # Append to cache; k_full / v_full span [0, offset + T)
            k_full, v_full = kv_cache.update(k, v)
            # Q attends over the full history — causal mask is already implicit
            # (Q positions are all >= any cached K position), so is_causal=False.
            out = F.scaled_dot_product_attention(
                q,
                k_full,
                v_full,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            # Standard path: full-sequence causal attention (training + prefill)
            q, k = self.rope(q, k, offset=0)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.drop.p if self.training else 0.0,
                is_causal=True,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(out)
