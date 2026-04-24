"""Causal language model built on Attention Residuals layers.

Wraps :class:`~models.attn_res.AttnResTransformerLayer` into a standard
autoregressive decoder that predicts the next character given a context
window.  This is the model used for training on Tiny-Shakespeare.

Architecture::

    token ids  →  Embedding(vocab_size, dim)
               →  + LearnedPositionalEmbedding(seq_len, dim)
               →  [AttnResTransformerLayer × depth]
               →  RMSNorm
               →  Linear(dim, vocab_size)   ← logits over next token

Two public classes are provided:

* :class:`AttnResLM`    — Attention Residuals language model.
* :class:`BaselineLM`   — Identical architecture but with standard residuals.

Both expose a :meth:`generate` method for autoregressive text generation.
KV-cache support is enabled via ``model.use_kv_cache: true`` in the YAML
config and is automatically engaged during :meth:`generate` when set.

KV-cache behaviour
------------------
When ``cfg.use_kv_cache`` is ``True``, :meth:`generate` operates as follows:

1. **Prefill phase** — the full prompt is processed in one forward pass with
   ``kv_cache=None`` (standard causal attention).  This is identical to the
   non-cached path and populates the KV caches for all layers via
   :class:`~models.components.KVCache.update` during the prefill step.

   .. note::
       To populate the caches during prefill, the model runs the prompt
       through the cached path with T > 1.  Since :class:`~models.components.KVCache`
       is empty at the start, the first call is equivalent to the standard
       non-cached forward.

2. **Decode phase** — each new token is processed as a single-token input
   ``(B=1, T=1)``.  Only the new token's Q, K, V are projected.  K and V
   are appended to the per-layer cache.  Attention spans the full history
   without recomputing past keys and values.

3. **Reset** — caches are cleared between independent generation calls so
   that sequences do not cross-contaminate each other.

Training is always cache-free (``kv_cache=None`` is never passed during
the training loop).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attn_res import AttnResTransformerLayer
from models.components import RMSNorm, CausalSelfAttention, SwiGLU, KVCache
from utils.config import ModelConfig


# ---------------------------------------------------------------------------
# AttnRes Language Model
# ---------------------------------------------------------------------------


class AttnResLM(nn.Module):
    """Autoregressive character-level language model with Attention Residuals.

    Args:
        cfg: :class:`~utils.config.ModelConfig` containing architecture
            hyper-parameters.  ``cfg.use_kv_cache`` controls whether the
            :meth:`generate` method uses KV-caching for faster decoding.
        vocab_size: Number of tokens in the vocabulary (output logit dimension).
        seq_len: Maximum context length (used for positional embeddings).
    """

    def __init__(
        self,
        cfg: ModelConfig,
        vocab_size: int,
        seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.seq_len = seq_len
        self.use_block = cfg.use_block_attn_res
        self.use_kv_cache = cfg.use_kv_cache

        # Token + position embeddings
        self.tok_embed = nn.Embedding(vocab_size, cfg.dim)
        self.pos_embed = nn.Embedding(seq_len, cfg.dim)

        # Transformer layers — pass use_kv_cache so each layer knows whether
        # to expect a cache argument during inference.
        self.layers = nn.ModuleList(
            [
                AttnResTransformerLayer(
                    dim=cfg.dim,
                    heads=cfg.heads,
                    head_dim=cfg.head_dim,
                    mlp_multiplier=cfg.mlp_multiplier,
                    dropout=cfg.dropout,
                    max_seq_len=seq_len,
                    layer_number=i + 1,
                    block_size=cfg.block_size,
                    use_block_attn_res=cfg.use_block_attn_res,
                    norm_eps=cfg.norm_eps,
                    use_kv_cache=cfg.use_kv_cache,
                )
                for i in range(cfg.depth)
            ]
        )

        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, vocab_size, bias=False)

        # Weight tying: embedding and output projection share weights.
        self.head.weight = self.tok_embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise all linear and embedding layers."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_kv_caches(self) -> list[KVCache]:
        """Allocate one :class:`~models.components.KVCache` per layer.

        Returns:
            List of :class:`~models.components.KVCache` instances, one for
            each transformer layer, all initially empty.
        """
        return [KVCache() for _ in self.layers]

    def _clear_kv_caches(self, caches: list[KVCache]) -> None:
        """Clear all caches in-place.

        Args:
            caches: List returned by :meth:`_make_kv_caches`.
        """
        for c in caches:
            c.reset()

    def _run_layers(
        self,
        h: torch.Tensor,
        kv_caches: Optional[list[KVCache]] = None,
    ) -> torch.Tensor:
        """Run all transformer layers and aggregate their outputs.

        Args:
            h: Initial hidden state ``(B, T, d)`` (embedding + pos).
            kv_caches: Per-layer :class:`~models.components.KVCache` list, or
                ``None`` for the standard (no-cache) path.

        Returns:
            Aggregated output tensor ``(B, T, d)`` ready for the final norm.
        """
        if self.use_block:
            block_reps: list[torch.Tensor] = [h]
            partial = torch.zeros_like(h)
            for i, layer in enumerate(self.layers):
                cache = kv_caches[i] if kv_caches else None
                block_reps, partial = layer(block_reps, partial, kv_cache=cache)
            return sum(block_reps[1:], partial)
        else:
            layer_outputs: list[torch.Tensor] = [h]
            for i, layer in enumerate(self.layers):
                cache = kv_caches[i] if kv_caches else None
                layer_outputs = layer(layer_outputs, kv_cache=cache)
            return sum(layer_outputs[1:])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[KVCache]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the forward pass.

        During training call without ``kv_caches`` (the default).  During
        cached inference :meth:`generate` supplies the cache list automatically.

        Args:
            x: Input token ids of shape ``(B, T)`` where ``T ≤ seq_len``.
                During cached decode steps ``T`` is typically 1.
            targets: Optional target token ids ``(B, T)`` for loss computation.
                If provided, cross-entropy loss is computed and returned.
                Must be ``None`` when using KV caches (inference only).
            kv_caches: Optional list of :class:`~models.components.KVCache`,
                one per layer, for incremental decoding.  ``None`` during
                all training and full-sequence prefill steps.

        Returns:
            Tuple ``(logits, loss)`` where:

            * ``logits`` has shape ``(B, T, vocab_size)``.
            * ``loss`` is a scalar tensor if ``targets`` is given, else ``None``.
        """
        B, T = x.shape
        assert (
            T <= self.seq_len
        ), f"Sequence length {T} exceeds model max {self.seq_len}."

        # Position indices: when using a cache the new tokens start at the
        # current cache length (supplied via the first cache's .length).
        if kv_caches and kv_caches[0].length > 0:
            offset = kv_caches[0].length
        else:
            offset = 0

        positions = torch.arange(offset, offset + T, device=x.device)
        h = self.tok_embed(x) + self.pos_embed(positions)  # (B, T, d)

        out = self._run_layers(h, kv_caches=kv_caches)
        out = self.norm(out)  # (B, T, d)
        logits = self.head(out)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_kv_cache: Optional[bool] = None,
    ) -> torch.Tensor:
        """Autoregressively generate new tokens from a prompt.

        When ``cfg.use_kv_cache`` is ``True`` (or overridden here), the prompt
        is processed in a single prefill forward pass that populates all KV
        caches, then each new token is decoded with a single-token forward pass.
        When ``use_kv_cache`` is ``False``, the full growing context is
        re-processed at each step (the original behaviour).

        Args:
            prompt_ids: Seed token ids ``(1, T_prompt)`` (single sequence).
            max_new_tokens: Number of new tokens to sample.
            temperature: Softmax temperature; lower = more greedy.
            top_k: If set, restricts sampling to the top-k logits at each step.
            use_kv_cache: Override ``cfg.use_kv_cache`` for this call.
                ``None`` (default) defers to ``self.use_kv_cache``.

        Returns:
            Token id tensor ``(1, T_prompt + max_new_tokens)``.
        """
        self.eval()
        ctx = prompt_ids  # (1, T)
        cache_on = self.use_kv_cache if use_kv_cache is None else use_kv_cache

        if cache_on:
            # ── Cached generation ──────────────────────────────────────────
            caches = self._make_kv_caches()
            logits, _ = self(ctx[:, -self.seq_len :], kv_caches=caches)
            next_id = self._sample(logits[:, -1, :], temperature, top_k)
            ctx = torch.cat([ctx, next_id], dim=1)
            for _ in range(max_new_tokens - 1):
                logits, _ = self(next_id, kv_caches=caches)
                next_id = self._sample(logits[:, -1, :], temperature, top_k)
                ctx = torch.cat([ctx, next_id], dim=1)
            self._clear_kv_caches(caches)
        else:
            # ── Standard generation (no cache) ─────────────────────────────
            for _ in range(max_new_tokens):
                ctx_cond = ctx[:, -self.seq_len :]
                logits, _ = self(ctx_cond)
                next_id = self._sample(logits[:, -1, :], temperature, top_k)
                ctx = torch.cat([ctx, next_id], dim=1)

        return ctx

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
    ) -> torch.Tensor:
        """Sample the next token id from logits.

        Args:
            logits: Un-normalised logits ``(B, vocab_size)``.
            temperature: Softmax temperature.
            top_k: If provided, keep only the top-k logits before sampling.

        Returns:
            Sampled token id tensor ``(B, 1)``.
        """
        logits = logits / temperature
        if top_k is not None:
            topk_vals, _ = logits.topk(min(top_k, logits.size(-1)))
            logits[logits < topk_vals[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters.

        Returns:
            Parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Baseline Language Model (standard residuals)
# ---------------------------------------------------------------------------


class BaselineLM(nn.Module):
    """Autoregressive character-level language model with standard residuals.

    Identical architecture to :class:`AttnResLM` but replaces AttnRes
    operations with fixed unit residuals.  Used as the ablation baseline.
    Supports KV-caching via ``cfg.use_kv_cache``.

    Args:
        cfg: Model configuration.
        vocab_size: Vocabulary size.
        seq_len: Maximum context length.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        vocab_size: int,
        seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.seq_len = seq_len
        self.use_kv_cache = cfg.use_kv_cache

        self.tok_embed = nn.Embedding(vocab_size, cfg.dim)
        self.pos_embed = nn.Embedding(seq_len, cfg.dim)

        self.layers = nn.ModuleList(
            [
                _BaselineLMLayer(
                    dim=cfg.dim,
                    heads=cfg.heads,
                    head_dim=cfg.head_dim,
                    mlp_multiplier=cfg.mlp_multiplier,
                    dropout=cfg.dropout,
                    max_seq_len=seq_len,
                    norm_eps=cfg.norm_eps,
                )
                for _ in range(cfg.depth)
            ]
        )

        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_kv_caches(self) -> list[KVCache]:
        """Allocate one KVCache per layer.

        Returns:
            List of empty :class:`~models.components.KVCache` instances.
        """
        return [KVCache() for _ in self.layers]

    def _clear_kv_caches(self, caches: list[KVCache]) -> None:
        """Reset all caches in-place.

        Args:
            caches: Cache list from :meth:`_make_kv_caches`.
        """
        for c in caches:
            c.reset()

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[KVCache]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the baseline LM forward pass.

        Args:
            x: Input token ids ``(B, T)``.
            targets: Optional target ids ``(B, T)`` for loss computation.
            kv_caches: Per-layer KV caches for incremental decoding.

        Returns:
            Tuple ``(logits, loss)``; ``loss`` is ``None`` when no targets given.
        """
        B, T = x.shape
        offset = kv_caches[0].length if (kv_caches and kv_caches[0].length > 0) else 0
        positions = torch.arange(offset, offset + T, device=x.device)
        h = self.tok_embed(x) + self.pos_embed(positions)

        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            h = layer(h, kv_cache=cache)

        logits = self.head(self.norm(h))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_kv_cache: Optional[bool] = None,
    ) -> torch.Tensor:
        """Autoregressively generate new tokens.

        Args:
            prompt_ids: Seed token ids ``(1, T_prompt)``.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filter (optional).
            use_kv_cache: Override ``cfg.use_kv_cache`` for this call.

        Returns:
            Token id tensor ``(1, T_prompt + max_new_tokens)``.
        """
        self.eval()
        ctx = prompt_ids
        cache_on = self.use_kv_cache if use_kv_cache is None else use_kv_cache

        if cache_on:
            caches = self._make_kv_caches()
            logits, _ = self(ctx[:, -self.seq_len :], kv_caches=caches)
            next_id = AttnResLM._sample(logits[:, -1, :], temperature, top_k)
            ctx = torch.cat([ctx, next_id], dim=1)
            for _ in range(max_new_tokens - 1):
                logits, _ = self(next_id, kv_caches=caches)
                next_id = AttnResLM._sample(logits[:, -1, :], temperature, top_k)
                ctx = torch.cat([ctx, next_id], dim=1)
            self._clear_kv_caches(caches)
        else:
            for _ in range(max_new_tokens):
                ctx_cond = ctx[:, -self.seq_len :]
                logits, _ = self(ctx_cond)
                next_id = AttnResLM._sample(logits[:, -1, :], temperature, top_k)
                ctx = torch.cat([ctx, next_id], dim=1)

        return ctx

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters.

        Returns:
            Parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Baseline internal layer
# ---------------------------------------------------------------------------


class _BaselineLMLayer(nn.Module):
    """Standard pre-norm transformer layer (fixed unit residuals) with KV-cache.

    Args:
        dim: Hidden dimension.
        heads: Number of attention heads.
        head_dim: Dimension per head.
        mlp_multiplier: SwiGLU inner-dim multiplier.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE.
        norm_eps: RMSNorm epsilon.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 256,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.mlp_norm = RMSNorm(dim, eps=norm_eps)
        self.attn = CausalSelfAttention(dim, heads, head_dim, max_seq_len, dropout)
        self.mlp = SwiGLU(dim, hidden_dim=dim * mlp_multiplier, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """Apply attention and MLP with standard residuals.

        Args:
            x: Hidden state ``(B, T, d)``.
            kv_cache: Optional :class:`~models.components.KVCache` for
                incremental decoding.  ``None`` during training.

        Returns:
            Updated hidden state ``(B, T, d)``.
        """
        x = x + self.attn(self.attn_norm(x), kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x
