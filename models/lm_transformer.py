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
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attn_res import AttnResTransformerLayer
from models.components import RMSNorm, CausalSelfAttention, SwiGLU
from utils.config import ModelConfig


# ---------------------------------------------------------------------------
# AttnRes Language Model
# ---------------------------------------------------------------------------


class AttnResLM(nn.Module):
    """Autoregressive character-level language model with Attention Residuals.

    Args:
        cfg: :class:`~utils.config.ModelConfig` containing architecture
            hyper-parameters.
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

        # Token + position embeddings
        self.tok_embed = nn.Embedding(vocab_size, cfg.dim)
        self.pos_embed = nn.Embedding(seq_len, cfg.dim)

        # Transformer layers
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
                )
                for i in range(cfg.depth)
            ]
        )

        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, vocab_size, bias=False)

        # Weight tying: embedding and output projection share weights
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
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the forward pass.

        Args:
            x: Input token ids of shape ``(B, T)`` where ``T ≤ seq_len``.
            targets: Optional target token ids ``(B, T)`` for loss computation.
                If provided, cross-entropy loss is computed and returned.

        Returns:
            Tuple ``(logits, loss)`` where:

            * ``logits`` has shape ``(B, T, vocab_size)``.
            * ``loss`` is a scalar tensor if ``targets`` is given, else ``None``.
        """
        B, T = x.shape
        assert (
            T <= self.seq_len
        ), f"Sequence length {T} exceeds model max {self.seq_len}."

        positions = torch.arange(T, device=x.device)
        h = self.tok_embed(x) + self.pos_embed(positions)  # (B, T, d)

        if self.use_block:
            block_reps: list[torch.Tensor] = [h]
            partial = torch.zeros_like(h)
            for layer in self.layers:
                block_reps, partial = layer(block_reps, partial)
            # Final hidden state: sum all block summaries (skip embedding block 0) + partial
            out = sum(block_reps[1:], partial)
        else:
            layer_outputs: list[torch.Tensor] = [h]
            for layer in self.layers:
                layer_outputs = layer(layer_outputs)
            out = sum(layer_outputs[1:])

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
    ) -> torch.Tensor:
        """Autoregressively generate new tokens from a prompt.

        Args:
            prompt_ids: Seed token ids ``(1, T_prompt)`` (single sequence).
            max_new_tokens: Number of new tokens to sample.
            temperature: Softmax temperature; lower = more greedy.
            top_k: If set, restricts sampling to the top-k logits at each step.

        Returns:
            Token id tensor ``(1, T_prompt + max_new_tokens)``.
        """
        self.eval()
        ctx = prompt_ids  # (1, T)
        for _ in range(max_new_tokens):
            # Crop context to the model's maximum window
            ctx_cond = ctx[:, -self.seq_len :]
            logits, _ = self(ctx_cond)  # (1, T, vocab)
            logits = logits[:, -1, :] / temperature  # (1, vocab)

            if top_k is not None:
                topk_vals, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
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
# Baseline Language Model (standard residuals)
# ---------------------------------------------------------------------------


class BaselineLM(nn.Module):
    """Autoregressive character-level language model with standard residuals.

    Identical architecture to :class:`AttnResLM` but replaces AttnRes
    operations with fixed unit residuals.  Used as the ablation baseline.

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
        self.seq_len = seq_len

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

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the baseline LM forward pass.

        Args:
            x: Input token ids ``(B, T)``.
            targets: Optional target ids ``(B, T)`` for loss computation.

        Returns:
            Tuple ``(logits, loss)``; ``loss`` is ``None`` when no targets given.
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        h = self.tok_embed(x) + self.pos_embed(positions)
        for layer in self.layers:
            h = layer(h)
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
    ) -> torch.Tensor:
        """Autoregressively generate new tokens.

        Args:
            prompt_ids: Seed token ids ``(1, T_prompt)``.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filter (optional).

        Returns:
            Token id tensor ``(1, T_prompt + max_new_tokens)``.
        """
        self.eval()
        ctx = prompt_ids
        for _ in range(max_new_tokens):
            ctx_cond = ctx[:, -self.seq_len :]
            logits, _ = self(ctx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                topk_vals, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
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
    """Standard pre-norm transformer layer (fixed unit residuals).

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP with standard residuals.

        Args:
            x: Hidden state ``(B, T, d)``.

        Returns:
            Updated hidden state ``(B, T, d)``.
        """
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x
