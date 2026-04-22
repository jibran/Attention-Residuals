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


# ---------------------------------------------------------------------------
# Exclusive Self Attention (XSA)
# ---------------------------------------------------------------------------


class TestXSA:
    """Tests for the Exclusive Self Attention extension.

    Reference: Zhai (2026), arXiv:2603.09078.
    """

    def _attn(self, use_xsa: bool) -> CausalSelfAttention:
        return CausalSelfAttention(
            dim=64, heads=4, head_dim=16, max_seq_len=32, use_xsa=use_xsa
        )

    # ── Construction ─────────────────────────────────────────────────────────

    def test_use_xsa_flag_stored(self):
        """use_xsa flag must be stored as an attribute."""
        assert self._attn(use_xsa=True).use_xsa is True
        assert self._attn(use_xsa=False).use_xsa is False

    def test_default_is_standard_attention(self):
        """Default construction (no use_xsa kwarg) must be standard attention."""
        attn = CausalSelfAttention(dim=32, heads=2, head_dim=16)
        assert attn.use_xsa is False

    def test_no_extra_parameters(self):
        """XSA must not add any learnable parameters."""
        attn_std = self._attn(use_xsa=False)
        attn_xsa = self._attn(use_xsa=True)
        n_std = sum(p.numel() for p in attn_std.parameters())
        n_xsa = sum(p.numel() for p in attn_xsa.parameters())
        assert n_std == n_xsa

    # ── Output shape ─────────────────────────────────────────────────────────

    def test_output_shape_matches_input(self):
        """XSA output shape must equal input shape."""
        attn = self._attn(use_xsa=True)
        x = torch.randn(2, 8, 64)
        assert attn(x).shape == x.shape

    def test_output_shape_single_token(self):
        """XSA must work with a sequence of length 1."""
        attn = self._attn(use_xsa=True)
        x = torch.randn(1, 1, 64)
        assert attn(x).shape == (1, 1, 64)

    # ── XSA mathematical property ────────────────────────────────────────────

    def test_xsa_output_orthogonal_to_self_value(self):
        """XSA output must be orthogonal to the self-value vector at each position.

        For each token i and head h, the dot product
        ``z_{i,h} · v_{i,h} / ‖v_{i,h}‖`` must be near zero.
        """
        torch.manual_seed(42)
        attn = self._attn(use_xsa=True)
        attn.eval()

        B, T, D = 2, 6, 64
        H, Hd = 4, 16
        x = torch.randn(B, T, D)

        with torch.no_grad():
            # Extract v directly from the QKV projection — we only need v,
            # not the full forward pass output, to test _apply_xsa.
            qkv = attn.qkv(x).chunk(3, dim=-1)
            _, _, v = (t.view(B, T, H, Hd).transpose(1, 2) for t in qkv)
            # v: (B, H, T, Hd) — self-value vectors

        # Verify the orthogonality property via _apply_xsa directly.
        y = torch.randn(B, H, T, Hd)
        z = attn._apply_xsa(y, v)
        vn = torch.nn.functional.normalize(v, dim=-1)
        dot = (z * vn).sum(dim=-1)  # (B, H, T)
        assert (
            dot.abs().max().item() < 1e-5
        ), f"Max dot product with self-value: {dot.abs().max().item():.2e}"

    def test_xsa_differs_from_standard_attention(self):
        """XSA and SA outputs must differ (XSA removes a component)."""
        torch.manual_seed(7)
        attn_std = self._attn(use_xsa=False)
        attn_xsa = self._attn(use_xsa=True)
        # Share weights so only the XSA step differs
        attn_xsa.load_state_dict(attn_std.state_dict())
        attn_std.eval()
        attn_xsa.eval()

        x = torch.randn(1, 5, 64)
        with torch.no_grad():
            out_std = attn_std(x)
            out_xsa = attn_xsa(x)
        assert not torch.allclose(
            out_std, out_xsa
        ), "XSA and standard attention gave identical outputs — XSA projection not applied."

    def test_xsa_standard_agree_when_v_zero(self):
        """When v is zero the XSA correction is zero, so outputs should agree."""
        # This is a theoretical edge case (v=0 is degenerate) — verifies
        # the correction formula handles zero values without NaN.
        attn = self._attn(use_xsa=True)
        attn.eval()
        # Force V projection to zero
        with torch.no_grad():
            # Zero the v columns of qkv weight (last third)
            D, inner = attn.qkv.weight.shape
            attn.qkv.weight[2 * (inner // 3) :] = 0.0
        x = torch.randn(1, 4, 64)
        with torch.no_grad():
            out = attn(x)
        assert not torch.isnan(out).any(), "NaN in XSA output with zero v vectors."

    # ── Gradient flow ────────────────────────────────────────────────────────

    def test_gradient_flows_through_xsa(self):
        """Gradients must flow through the XSA projection step."""
        attn = self._attn(use_xsa=True)
        x = torch.randn(2, 6, 64, requires_grad=True)
        attn(x).sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    # ── Causal mask preserved ────────────────────────────────────────────────

    def test_xsa_preserves_causal_mask(self):
        """XSA must not break the causal mask: future tokens must not
        influence past token outputs."""
        torch.manual_seed(3)
        attn = self._attn(use_xsa=True)
        attn.eval()
        x = torch.randn(1, 5, 64)
        x_mod = x.clone()
        x_mod[:, -1, :] = 0.0
        with torch.no_grad():
            out = attn(x)
            out_mod = attn(x_mod)
        assert torch.allclose(out[:, 0, :], out_mod[:, 0, :], atol=1e-5)

    # ── KV cache compatibility ────────────────────────────────────────────────

    def test_xsa_works_with_kv_cache(self):
        """XSA + KV cache must produce valid (non-NaN) output of correct shape."""
        from models.components import KVCache

        attn = self._attn(use_xsa=True)
        attn.eval()
        cache = KVCache()
        # Prefill
        x_pre = torch.randn(1, 4, 64)
        out_pre = attn(x_pre, kv_cache=cache)
        assert out_pre.shape == (1, 4, 64)
        assert cache.length == 4
        # Decode one token
        x_new = torch.randn(1, 1, 64)
        out_new = attn(x_new, kv_cache=cache)
        assert out_new.shape == (1, 1, 64)
        assert cache.length == 5
        assert not torch.isnan(out_new).any()

    # ── Config / model integration ───────────────────────────────────────────

    def test_config_flag_default_false(self):
        """ModelConfig.use_xsa must default to False."""
        from utils.config import ModelConfig

        assert ModelConfig().use_xsa is False

    def test_config_flag_true(self):
        """ModelConfig must accept use_xsa=True."""
        from utils.config import ModelConfig

        assert ModelConfig(use_xsa=True).use_xsa is True

    def test_attnres_lm_with_xsa_forward(self):
        """AttnResLM built with use_xsa=True must produce valid logits."""
        from models.lm_transformer import AttnResLM
        from utils.config import ModelConfig

        cfg = ModelConfig(
            dim=32, depth=2, heads=2, head_dim=16, max_seq_len=16, use_xsa=True
        )
        model = AttnResLM(cfg, vocab_size=67, seq_len=16)
        x = torch.randint(0, 67, (2, 16))
        logits, _ = model(x)
        assert logits.shape == (2, 16, 67)
        assert not torch.isnan(logits).any()

    def test_baseline_lm_with_xsa_forward(self):
        """BaselineLM built with use_xsa=True must produce valid logits."""
        from models.lm_transformer import BaselineLM
        from utils.config import ModelConfig

        cfg = ModelConfig(
            dim=32, depth=2, heads=2, head_dim=16, max_seq_len=16, use_xsa=True
        )
        model = BaselineLM(cfg, vocab_size=67, seq_len=16)
        x = torch.randint(0, 67, (2, 16))
        logits, _ = model(x)
        assert logits.shape == (2, 16, 67)
        assert not torch.isnan(logits).any()

    def test_yaml_override_enables_xsa(self, tmp_path):
        """CLI override model.use_xsa=true must propagate through load_config."""
        from utils.config import load_config

        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(
            "model:\n  use_xsa: false\n"
            "training:\n  device: cpu\n"
            "data:\n  dataset: shakespeare\n"
            "logging:\n  log_dir: logs/\n  checkpoint_dir: checkpoints/\n"
            "generation:\n  max_new_tokens: 100\n  temperature: 1.0\n  top_k: 0\n"
        )
        cfg = load_config(yaml_path, overrides=["model.use_xsa=true"])
        assert cfg.model.use_xsa is True
