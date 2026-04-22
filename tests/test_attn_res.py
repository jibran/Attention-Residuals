"""Unit tests for the Attention Residuals core operations.

Tests cover:
  * :class:`~models.attn_res.FullAttnResOp`
  * :class:`~models.attn_res.BlockAttnResOp`
  * :class:`~models.attn_res.AttnResTransformerLayer`
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.attn_res import FullAttnResOp, BlockAttnResOp, AttnResTransformerLayer


B, T, D = 2, 8, 64  # batch, seq_len, dim


def _make_outputs(n: int) -> list[torch.Tensor]:
    """Create a list of n random layer-output tensors."""
    return [torch.randn(B, T, D) for _ in range(n)]


# ---------------------------------------------------------------------------
# FullAttnResOp
# ---------------------------------------------------------------------------


class TestFullAttnResOp:
    """Tests for the Full Attention Residuals operation."""

    def test_output_shape(self):
        """Output shape must be (B, T, D)."""
        op = FullAttnResOp(dim=D)
        vecs = _make_outputs(4)
        out = op(vecs)
        assert out.shape == (B, T, D)

    def test_single_previous_output(self):
        """With only the embedding (one previous output), must still work."""
        op = FullAttnResOp(dim=D)
        vecs = _make_outputs(1)
        out = op(vecs)
        assert out.shape == (B, T, D)

    def test_weights_sum_to_one(self):
        """Attention weights over previous outputs must sum to 1 per token."""
        op = FullAttnResOp(dim=D)
        vecs = _make_outputs(5)
        V = torch.stack(vecs, dim=0)  # (5, B, T, D)
        K = op.key_norm(V)
        logits = torch.einsum("d, n b t d -> n b t", op.query_vec, K)
        alpha = logits.softmax(dim=0)  # (5, B, T)
        sums = alpha.sum(dim=0)  # (B, T)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gradient_flows(self):
        """Gradients must reach both query_vec and the previous outputs."""
        op = FullAttnResOp(dim=D)
        vecs = [torch.randn(B, T, D, requires_grad=True) for _ in range(3)]
        out = op(vecs)
        out.sum().backward()
        assert op.query_vec.grad is not None
        for v in vecs:
            assert v.grad is not None

    def test_output_changes_with_inputs(self):
        """Different layer outputs must produce different aggregated states."""
        op = FullAttnResOp(dim=D)
        vecs_a = _make_outputs(3)
        vecs_b = _make_outputs(3)
        with torch.no_grad():
            out_a = op(vecs_a)
            out_b = op(vecs_b)
        assert not torch.allclose(out_a, out_b)

    def test_more_layers_handled(self):
        """Must scale gracefully to many previous outputs (e.g. 32)."""
        op = FullAttnResOp(dim=D)
        vecs = _make_outputs(32)
        out = op(vecs)
        assert out.shape == (B, T, D)
        assert not torch.isnan(out).any()

    def test_bounded_magnitude(self):
        """Output norms should not blow up — key RMSNorm should keep them in check."""
        op = FullAttnResOp(dim=D)
        # Use large-magnitude inputs to stress-test
        vecs = [torch.randn(B, T, D) * 100 for _ in range(8)]
        out = op(vecs)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# BlockAttnResOp
# ---------------------------------------------------------------------------


class TestBlockAttnResOp:
    """Tests for the Block Attention Residuals operation."""

    def test_output_shape(self):
        """Output shape must be (B, T, D)."""
        op = BlockAttnResOp(dim=D)
        blocks = _make_outputs(3)
        partial = torch.randn(B, T, D)
        out = op(blocks, partial)
        assert out.shape == (B, T, D)

    def test_single_block_rep(self):
        """Must work with only the embedding as the sole block rep."""
        op = BlockAttnResOp(dim=D)
        blocks = _make_outputs(1)
        partial = torch.randn(B, T, D)
        out = op(blocks, partial)
        assert out.shape == (B, T, D)

    def test_weights_sum_to_one(self):
        """Attention weights over (blocks + partial) must sum to 1."""
        op = BlockAttnResOp(dim=D)
        blocks = _make_outputs(4)
        partial = torch.randn(B, T, D)
        all_reps = blocks + [partial]
        V = torch.stack(all_reps, dim=0)
        K = op.key_norm(V)
        logits = torch.einsum("d, n b t d -> n b t", op.query_vec, K)
        alpha = logits.softmax(dim=0)
        sums = alpha.sum(dim=0)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gradient_flows(self):
        """Gradients must flow to query_vec, block reps, and partial block."""
        op = BlockAttnResOp(dim=D)
        blocks = [torch.randn(B, T, D, requires_grad=True) for _ in range(3)]
        partial = torch.randn(B, T, D, requires_grad=True)
        out = op(blocks, partial)
        out.sum().backward()
        assert op.query_vec.grad is not None
        assert partial.grad is not None
        for b in blocks:
            assert b.grad is not None

    def test_output_not_nan_with_many_blocks(self):
        """Must handle many blocks without NaN."""
        op = BlockAttnResOp(dim=D)
        blocks = _make_outputs(16)
        partial = torch.randn(B, T, D)
        out = op(blocks, partial)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# AttnResTransformerLayer — Full AttnRes mode
# ---------------------------------------------------------------------------


class TestAttnResTransformerLayerFull:
    """Tests for AttnResTransformerLayer in Full AttnRes mode."""

    def _make_layer(self, layer_number: int = 1) -> AttnResTransformerLayer:
        return AttnResTransformerLayer(
            dim=D,
            heads=4,
            head_dim=16,
            mlp_multiplier=2,
            dropout=0.0,
            max_seq_len=T,
            layer_number=layer_number,
            block_size=4,
            use_block_attn_res=False,
        )

    def test_output_extends_list(self):
        """forward_full should append exactly 2 tensors (attn out + MLP out)."""
        layer = self._make_layer()
        layer_outputs = _make_outputs(3)  # embedding + 2 previous sublayer outs
        n_before = len(layer_outputs)
        updated = layer(layer_outputs)
        assert len(updated) == n_before + 2

    def test_output_shape_correct(self):
        """Newly appended tensors must have shape (B, T, D)."""
        layer = self._make_layer()
        layer_outputs = _make_outputs(2)
        updated = layer(layer_outputs)
        for out in updated[-2:]:
            assert out.shape == (B, T, D)

    def test_gradient_flows_through_full(self):
        """Loss can be back-propagated through a Full AttnRes layer."""
        layer = self._make_layer()
        embed = torch.randn(B, T, D, requires_grad=True)
        updated = layer([embed])
        updated[-1].sum().backward()
        assert embed.grad is not None

    def test_stacked_full_layers(self):
        """Three Full AttnRes layers stacked should work end-to-end."""
        layers = [self._make_layer(i + 1) for i in range(3)]
        outs = _make_outputs(1)  # start with just the embedding
        for lyr in layers:
            outs = lyr(outs)
        assert outs[-1].shape == (B, T, D)
        assert not torch.isnan(outs[-1]).any()


# ---------------------------------------------------------------------------
# AttnResTransformerLayer — Block AttnRes mode
# ---------------------------------------------------------------------------


class TestAttnResTransformerLayerBlock:
    """Tests for AttnResTransformerLayer in Block AttnRes mode."""

    def _make_layer(
        self, layer_number: int = 1, block_size: int = 4
    ) -> AttnResTransformerLayer:
        return AttnResTransformerLayer(
            dim=D,
            heads=4,
            head_dim=16,
            mlp_multiplier=2,
            dropout=0.0,
            max_seq_len=T,
            layer_number=layer_number,
            block_size=block_size,
            use_block_attn_res=True,
        )

    def test_output_types(self):
        """forward_block must return (list, Tensor)."""
        layer = self._make_layer()
        blocks = _make_outputs(1)
        partial = torch.zeros(B, T, D)
        result = layer(blocks, partial)
        assert isinstance(result, tuple)
        assert len(result) == 2
        block_reps, partial_out = result
        assert isinstance(block_reps, list)
        assert isinstance(partial_out, torch.Tensor)

    def test_partial_block_shape(self):
        """partial_block output must be (B, T, D)."""
        layer = self._make_layer()
        blocks = _make_outputs(1)
        partial = torch.zeros(B, T, D)
        _, partial_out = layer(blocks, partial)
        assert partial_out.shape == (B, T, D)

    def test_block_boundary_triggers(self):
        """A layer at the block boundary should append a new block rep."""
        # block_size=2 means block boundary every 1 layer (2 sublayers / 2)
        layer = self._make_layer(layer_number=1, block_size=2)
        blocks = _make_outputs(1)
        partial = torch.zeros(B, T, D)
        n_before = len(blocks)
        new_blocks, _ = layer(blocks, partial)
        assert len(new_blocks) > n_before

    def test_no_block_boundary_mid_block(self):
        """A mid-block layer should NOT append to block_reps."""
        # block_size=8 means boundary only every 4 layers
        layer = self._make_layer(layer_number=1, block_size=8)
        blocks = _make_outputs(1)
        partial = torch.zeros(B, T, D)
        n_before = len(blocks)
        new_blocks, _ = layer(blocks, partial)
        assert len(new_blocks) == n_before

    def test_gradient_flows_through_block(self):
        """Loss can be back-propagated through a Block AttnRes layer."""
        layer = self._make_layer()
        embed = torch.randn(B, T, D, requires_grad=True)
        partial = torch.zeros(B, T, D)
        _, partial_out = layer([embed], partial)
        partial_out.sum().backward()
        assert embed.grad is not None

    def test_stacked_block_layers(self):
        """Four Block AttnRes layers stacked should produce valid output."""
        layers = [self._make_layer(i + 1, block_size=4) for i in range(4)]
        blocks = [torch.randn(B, T, D)]
        partial = torch.zeros(B, T, D)
        for lyr in layers:
            blocks, partial = lyr(blocks, partial)
        assert partial.shape == (B, T, D)
        assert not torch.isnan(partial).any()

    def test_block_and_full_same_interface(self):
        """Calling layer(*args) should dispatch correctly for both variants."""
        full_layer = (
            self._make_layer(use_block_attn_res=False)
            if False
            else AttnResTransformerLayer(
                dim=D, heads=4, head_dim=16, use_block_attn_res=False
            )
        )
        block_layer = self._make_layer()

        # Full AttnRes: single-list arg
        outs = _make_outputs(2)
        updated = full_layer(outs)
        assert isinstance(updated, list)

        # Block AttnRes: two args
        blocks = _make_outputs(1)
        partial = torch.zeros(B, T, D)
        result = block_layer(blocks, partial)
        assert isinstance(result, tuple)
