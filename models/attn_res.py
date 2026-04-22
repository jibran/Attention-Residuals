"""Attention Residuals — core operations.

Implements the two variants from the paper:

**Full AttnRes** (:class:`FullAttnResOp`)
    Each layer attends over *all* previous layer outputs.
    Memory: ``O(L·d)``.  No overhead in vanilla training (activations are
    already retained for backprop).

**Block AttnRes** (:class:`BlockAttnResOp`)
    Layers are grouped into ``N`` blocks.  Within each block, standard
    residuals accumulate.  Between blocks, softmax attention is applied over
    the ``N`` block-level summary tensors.
    Memory: ``O(N·d)`` — reduces from linear-in-layers to linear-in-blocks.

Both ops are wired into :class:`AttnResTransformerLayer`, a drop-in
replacement for a standard pre-norm transformer layer.

Reference
---------
Chen et al., "Attention Residuals" (2026).
https://arxiv.org/abs/2603.15031
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.components import CausalSelfAttention, KVCache, RMSNorm, SwiGLU

# ---------------------------------------------------------------------------
# Full Attention Residuals operation
# ---------------------------------------------------------------------------


class FullAttnResOp(nn.Module):
    """Full Attention Residuals operation.

    Replaces the fixed residual ``h_l = h_{l-1} + f(h_{l-1})`` with::

        h_l = Σ_{i=0}^{l-1}  α_{i→l} · v_i

    where the attention weights are::

        α_{i→l} = softmax_i( w_l^T · RMSNorm(v_i) )

    ``w_l ∈ R^d`` is a single *learned* pseudo-query vector per layer.
    RMSNorm on the keys prevents large-magnitude layers from dominating.

    Args:
        dim: Hidden dimension ``d``.
        eps: Epsilon for the key :class:`~models.components.RMSNorm`.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.query_vec = nn.Parameter(torch.zeros(dim))  # w_l  (d,)
        self.key_norm = RMSNorm(dim, eps=eps)
        nn.init.normal_(self.query_vec, std=0.02)

    def forward(self, layer_outputs: list[torch.Tensor]) -> torch.Tensor:
        """Compute the attention-weighted sum over all previous layer outputs.

        Args:
            layer_outputs: List of tensors ``v_0, v_1, …, v_{l-1}``, each of
                shape ``(B, T, d)``.  The first element is the token embedding
                ``h_1``; subsequent elements are sublayer outputs ``f_i(h_i)``.

        Returns:
            Aggregated hidden state ``h_l`` of shape ``(B, T, d)``.
        """
        # Stack: (n_prev, B, T, d)
        V = torch.stack(layer_outputs, dim=0)

        # Keys: RMSNorm applied per-position across d
        K = self.key_norm(V)  # (n_prev, B, T, d)

        # Scalar attention logit per previous output
        # w: (d,)  ·  K: (n_prev, B, T, d)  →  (n_prev, B, T)
        logits = torch.einsum("d, n b t d -> n b t", self.query_vec, K)
        alpha = logits.softmax(dim=0)  # (n_prev, B, T)

        # Weighted sum
        h = torch.einsum("n b t, n b t d -> b t d", alpha, V)
        return h


# ---------------------------------------------------------------------------
# Block Attention Residuals operation
# ---------------------------------------------------------------------------


class BlockAttnResOp(nn.Module):
    """Block Attention Residuals operation.

    Attends over *block-level* summaries rather than all individual layer
    outputs, reducing memory from ``O(L·d)`` to ``O(N·d)``.

    Within each block, layer outputs accumulate via standard residuals into a
    *partial block sum*.  At each AttnRes application point the op receives:

    * ``block_reps`` — the ``N`` completed block summaries so far (the token
      embedding counts as block 0).
    * ``partial_block`` — the intra-block accumulation up to this sublayer.

    Args:
        dim: Hidden dimension ``d``.
        eps: Epsilon for the key :class:`~models.components.RMSNorm`.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.query_vec = nn.Parameter(torch.zeros(dim))
        self.key_norm = RMSNorm(dim, eps=eps)
        nn.init.normal_(self.query_vec, std=0.02)

    def forward(
        self,
        block_reps: list[torch.Tensor],
        partial_block: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the attention-weighted sum over block reps + partial sum.

        Args:
            block_reps: List of completed block summaries, each ``(B, T, d)``.
                Includes the token embedding as the first "block".
            partial_block: Intra-block partial sum so far ``(B, T, d)``.

        Returns:
            Aggregated hidden state ``h`` of shape ``(B, T, d)``.
        """
        # Concatenate completed blocks with the ongoing partial sum
        all_reps = block_reps + [partial_block]  # N+1 tensors
        V = torch.stack(all_reps, dim=0)  # (N+1, B, T, d)

        K = self.key_norm(V)
        logits = torch.einsum("d, n b t d -> n b t", self.query_vec, K)
        alpha = logits.softmax(dim=0)
        h = torch.einsum("n b t, n b t d -> b t d", alpha, V)
        return h


# ---------------------------------------------------------------------------
# AttnRes Transformer Layer
# ---------------------------------------------------------------------------


class AttnResTransformerLayer(nn.Module):
    """Single transformer layer with Attention Residuals.

    Implements the pre-norm transformer update with AttnRes replacing the
    standard residual accumulation.  Each layer contains:

    1. An :class:`~models.attn_res.FullAttnResOp` or
       :class:`~models.attn_res.BlockAttnResOp` applied *before* self-attention.
    2. Causal self-attention (:class:`~models.components.CausalSelfAttention`).
    3. Another AttnRes op applied *before* the MLP.
    4. SwiGLU MLP (:class:`~models.components.SwiGLU`).

    Args:
        dim: Hidden dimension.
        heads: Number of self-attention heads.
        head_dim: Dimension per attention head.
        mlp_multiplier: SwiGLU inner-dim multiplier.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE.
        layer_number: 1-based index within the transformer (used for block
            boundary detection in Block AttnRes).
        block_size: Sublayer count per block (Block AttnRes); each transformer
            layer contributes 2 sublayers (Attn + MLP).
        use_block_attn_res: If ``True``, use :class:`BlockAttnResOp`;
            otherwise use :class:`FullAttnResOp`.
        norm_eps: Epsilon for pre-norm :class:`~models.components.RMSNorm`.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        layer_number: int = 1,
        block_size: int = 4,
        use_block_attn_res: bool = True,
        norm_eps: float = 1e-6,
        use_kv_cache: bool = False,
    ) -> None:
        super().__init__()
        self.use_kv_cache = use_kv_cache
        self.layer_number = layer_number
        self.block_size = block_size  # sublayers per block
        self.use_block_attn_res = use_block_attn_res

        # Pre-norm for attn and MLP inputs
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.mlp_norm = RMSNorm(dim, eps=norm_eps)

        # Sub-layers
        self.attn = CausalSelfAttention(dim, heads, head_dim, max_seq_len, dropout)
        self.mlp = SwiGLU(dim, hidden_dim=dim * mlp_multiplier, dropout=dropout)

        # AttnRes ops (one before attn, one before MLP)
        if use_block_attn_res:
            self.attn_res_attn = BlockAttnResOp(dim, eps=norm_eps)
            self.attn_res_mlp = BlockAttnResOp(dim, eps=norm_eps)
        else:
            self.attn_res_attn = FullAttnResOp(dim, eps=norm_eps)  # type: ignore[assignment]
            self.attn_res_mlp = FullAttnResOp(dim, eps=norm_eps)  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Forward — Full AttnRes
    # ------------------------------------------------------------------

    def forward_full(
        self,
        layer_outputs: list[torch.Tensor],
        kv_cache: KVCache | None = None,
    ) -> list[torch.Tensor]:
        """Full AttnRes forward pass.

        Appends the outputs of the Attn sublayer and the MLP sublayer to
        ``layer_outputs`` and returns the extended list.

        Args:
            layer_outputs: Accumulated list of all previous sublayer outputs
                ``[embedding, f_1(h_1), f_2(h_2), …]``.
            kv_cache: Optional :class:`~models.components.KVCache` for this
                layer.  Passed directly to :meth:`~models.components.CausalSelfAttention.forward`.
                ``None`` (default) uses full-context attention.

        Returns:
            Updated ``layer_outputs`` with two new tensors appended (attn out,
            MLP out).  The *last* tensor in the list is the new hidden state.
        """
        # --- Self-attention sublayer ---
        h_attn = self.attn_res_attn(layer_outputs)
        attn_out = self.attn(self.attn_norm(h_attn), kv_cache=kv_cache)
        layer_outputs.append(attn_out)

        # --- MLP sublayer ---
        h_mlp = self.attn_res_mlp(layer_outputs)
        mlp_out = self.mlp(self.mlp_norm(h_mlp))
        layer_outputs.append(mlp_out)

        return layer_outputs

    # ------------------------------------------------------------------
    # Forward — Block AttnRes
    # ------------------------------------------------------------------

    def forward_block(
        self,
        block_reps: list[torch.Tensor],
        partial_block: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Block AttnRes forward pass.

        Each transformer layer contains two sublayers (Attn + MLP), each
        counting as one sublayer toward the ``block_size`` boundary.

        Args:
            block_reps: List of completed block summaries (shape ``(B,T,d)``
                each).  The token embedding is block 0.
            partial_block: Running intra-block residual sum ``(B, T, d)``.
            kv_cache: Optional :class:`~models.components.KVCache` for this
                layer.  ``None`` (default) uses full-context attention.

        Returns:
            Tuple ``(block_reps, partial_block)`` after processing both
            sublayers.  If this layer reaches a block boundary, the old
            ``partial_block`` is appended to ``block_reps`` and a new one
            is started.
        """
        # --- Self-attention sublayer ---
        h_attn = self.attn_res_attn(block_reps, partial_block)
        attn_out = self.attn(self.attn_norm(h_attn), kv_cache=kv_cache)
        partial_block = partial_block + attn_out

        # Block boundary check: each transformer layer = 2 sublayers
        # Boundary triggers when (layer_number * 2) % block_size == 0
        sublayer_idx = self.layer_number * 2  # after attn
        if sublayer_idx % self.block_size == 0:
            block_reps = block_reps + [partial_block]
            partial_block = torch.zeros_like(partial_block)

        # --- MLP sublayer ---
        h_mlp = self.attn_res_mlp(block_reps, partial_block)
        mlp_out = self.mlp(self.mlp_norm(h_mlp))
        partial_block = partial_block + mlp_out

        sublayer_idx = self.layer_number * 2 + 1  # after MLP
        if sublayer_idx % self.block_size == 0:
            block_reps = block_reps + [partial_block]
            partial_block = torch.zeros_like(partial_block)

        return block_reps, partial_block

    # ------------------------------------------------------------------
    # Unified forward
    # ------------------------------------------------------------------

    def forward(self, *args, kv_cache: KVCache | None = None):
        """Route to full or block forward depending on construction.

        For Full AttnRes pass a single list ``layer_outputs``.
        For Block AttnRes pass ``(block_reps, partial_block)``.

        Args:
            *args: Positional state arguments as above.
            kv_cache: Optional :class:`~models.components.KVCache` for this
                layer's attention sublayer.  ``None`` during training and
                non-cached inference.

        Returns:
            Updated state as returned by :meth:`forward_full` or
            :meth:`forward_block`.
        """
        if self.use_block_attn_res:
            assert len(args) == 2, "Block AttnRes needs (block_reps, partial_block)."
            return self.forward_block(*args, kv_cache=kv_cache)
        else:
            assert len(args) == 1, "Full AttnRes needs (layer_outputs,)."
            return self.forward_full(*args, kv_cache=kv_cache)
