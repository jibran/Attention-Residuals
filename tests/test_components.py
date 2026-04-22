"""Unit tests for shared neural-network building blocks.

Tests cover:
  * :class:`~models.components.RMSNorm`
  * :class:`~models.components.SwiGLU`
  * :class:`~models.components.RotaryEmbedding`
  * :class:`~models.components.CausalSelfAttention`
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.components import CausalSelfAttention, RMSNorm, RotaryEmbedding, SwiGLU

# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_output_shape(self):
        """Output shape must match input shape."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64)
        assert norm(x).shape == x.shape

    def test_rms_close_to_one(self):
        """After normalisation, RMS of each vector should be ≈ 1 (when weight=1)."""
        norm = RMSNorm(dim=128)
        with torch.no_grad():
            norm.weight.fill_(1.0)
        x = torch.randn(4, 8, 128) * 5.0
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_learnable_weight(self):
        """Weight parameter should be learnable and affect output."""
        norm = RMSNorm(dim=32)
        x = torch.randn(2, 5, 32)
        out1 = norm(x)
        with torch.no_grad():
            norm.weight.mul_(2.0)
        out2 = norm(x)
        assert not torch.allclose(out1, out2)

    def test_gradient_flows(self):
        """Gradients must flow through RMSNorm."""
        norm = RMSNorm(dim=16)
        x = torch.randn(2, 4, 16, requires_grad=True)
        norm(x).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------


class TestSwiGLU:
    """Tests for SwiGLU feed-forward block."""

    def test_output_shape(self):
        """SwiGLU must preserve the last dimension."""
        mlp = SwiGLU(dim=64)
        x = torch.randn(3, 7, 64)
        assert mlp(x).shape == x.shape

    def test_custom_hidden_dim(self):
        """Custom hidden_dim should be respected."""
        mlp = SwiGLU(dim=32, hidden_dim=128)
        assert mlp.gate.out_features == 128
        assert mlp.down.in_features == 128

    def test_default_hidden_dim(self):
        """Default hidden dim should be int(dim * 8/3)."""
        mlp = SwiGLU(dim=48)
        expected = int(48 * 8 / 3)
        assert mlp.gate.out_features == expected

    def test_gradient_flows(self):
        """Gradients must back-propagate through SwiGLU."""
        mlp = SwiGLU(dim=32)
        x = torch.randn(2, 5, 32, requires_grad=True)
        mlp(x).sum().backward()
        assert x.grad is not None

    def test_no_bias(self):
        """All linear layers in SwiGLU should be bias-free."""
        mlp = SwiGLU(dim=64)
        for name, mod in mlp.named_modules():
            if isinstance(mod, torch.nn.Linear):
                assert mod.bias is None, f"{name} has a bias"


# ---------------------------------------------------------------------------
# RotaryEmbedding
# ---------------------------------------------------------------------------


class TestRotaryEmbedding:
    """Tests for Rotary Position Embeddings."""

    def test_output_shapes(self):
        """q_rot and k_rot must have the same shapes as q and k."""
        rope = RotaryEmbedding(head_dim=32, max_seq_len=64)
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 4, 16, 32)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_preserves_norm(self):
        """Rotation is norm-preserving — vector magnitudes should be unchanged."""
        rope = RotaryEmbedding(head_dim=64, max_seq_len=32)
        q = torch.randn(1, 1, 10, 64)
        k = torch.randn(1, 1, 10, 64)
        q_rot, k_rot = rope(q, k)
        assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
        assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)

    def test_different_positions_differ(self):
        """Two tokens at different positions should get different rotations."""
        rope = RotaryEmbedding(head_dim=32, max_seq_len=16)
        q = torch.ones(1, 1, 4, 32)
        q_rot, _ = rope(q, q)
        # Position 0 and position 1 should differ
        assert not torch.allclose(q_rot[:, :, 0], q_rot[:, :, 1])

    def test_cache_extends_for_longer_seq(self):
        """RoPE should rebuild its cache when sequence exceeds max_seq_len."""
        rope = RotaryEmbedding(head_dim=16, max_seq_len=8)
        q = torch.randn(1, 1, 20, 16)  # longer than max_seq_len=8
        k = torch.randn(1, 1, 20, 16)
        q_rot, k_rot = rope(q, k)  # should not raise
        assert q_rot.shape == q.shape


# ---------------------------------------------------------------------------
# CausalSelfAttention
# ---------------------------------------------------------------------------


class TestCausalSelfAttention:
    """Tests for CausalSelfAttention."""

    def test_output_shape(self):
        """Output shape must match input shape."""
        attn = CausalSelfAttention(dim=64, heads=4, head_dim=16, max_seq_len=32)
        x = torch.randn(2, 10, 64)
        assert attn(x).shape == x.shape

    def test_gradient_flows(self):
        """Gradients must flow back to the input."""
        attn = CausalSelfAttention(dim=32, heads=2, head_dim=16, max_seq_len=16)
        x = torch.randn(2, 8, 32, requires_grad=True)
        attn(x).sum().backward()
        assert x.grad is not None

    def test_single_token(self):
        """Attention should work with a sequence of length 1."""
        attn = CausalSelfAttention(dim=64, heads=4, head_dim=16)
        x = torch.randn(1, 1, 64)
        out = attn(x)
        assert out.shape == (1, 1, 64)

    def test_causal_mask_effect(self):
        """Future tokens must not influence past tokens.

        We zero out the last token and verify the first token's output is
        unchanged — if the mask were missing, it would change.
        """
        torch.manual_seed(0)
        attn = CausalSelfAttention(dim=32, heads=2, head_dim=16, max_seq_len=16)
        attn.eval()

        x = torch.randn(1, 4, 32)
        x_modified = x.clone()
        x_modified[:, -1, :] = 0.0  # zero the last (future) token

        with torch.no_grad():
            out_orig = attn(x)
            out_mod = attn(x_modified)

        # First token output must not change when future token is zeroed
        assert torch.allclose(out_orig[:, 0, :], out_mod[:, 0, :], atol=1e-5)

    def test_no_bias(self):
        """QKV and output projections must be bias-free."""
        attn = CausalSelfAttention(dim=64, heads=4, head_dim=16)
        assert attn.qkv.bias is None
        assert attn.proj.bias is None
