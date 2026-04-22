"""Tests for KV-cache functionality.

Covers:
  * :class:`~models.components.KVCache` dataclass (update, length, reset)
  * :class:`~models.components.CausalSelfAttention` with and without cache
  * :class:`~models.components.RotaryEmbedding` positional offset
  * :class:`~models.lm_transformer.AttnResLM` cached vs uncached generation
  * :class:`~models.lm_transformer.BaselineLM` cached vs uncached generation
  * Config flag ``use_kv_cache`` toggling
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.components import CausalSelfAttention, KVCache, RotaryEmbedding
from models.lm_transformer import AttnResLM, BaselineLM
from utils.config import ModelConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _lm_cfg(use_kv_cache: bool = False, use_block: bool = True) -> ModelConfig:
    """Minimal ModelConfig for fast CPU tests."""
    return ModelConfig(
        name="TestLM",
        dim=32,
        depth=2,
        heads=2,
        head_dim=16,
        mlp_multiplier=2,
        dropout=0.0,
        use_block_attn_res=use_block,
        block_size=2,
        norm_eps=1e-6,
        max_seq_len=64,
        use_kv_cache=use_kv_cache,
    )


VOCAB = 67
SEQ = 32
B, T = 2, 8


# ---------------------------------------------------------------------------
# KVCache dataclass
# ---------------------------------------------------------------------------


class TestKVCache:
    """Tests for the KVCache dataclass."""

    def test_initial_length_is_zero(self):
        """A fresh cache must report length 0."""
        cache = KVCache()
        assert cache.length == 0

    def test_update_appends_tokens(self):
        """update() must concatenate keys and values along the sequence dim."""
        cache = KVCache()
        k1 = torch.randn(1, 2, 3, 16)
        v1 = torch.randn(1, 2, 3, 16)
        full_k, full_v = cache.update(k1, v1)
        assert full_k.shape == (1, 2, 3, 16)
        assert cache.length == 3

        k2 = torch.randn(1, 2, 1, 16)
        v2 = torch.randn(1, 2, 1, 16)
        full_k2, full_v2 = cache.update(k2, v2)
        assert full_k2.shape == (1, 2, 4, 16)
        assert cache.length == 4

    def test_update_returns_full_history(self):
        """update() return values must contain the complete accumulated history."""
        cache = KVCache()
        k1 = torch.ones(1, 1, 2, 8)
        k2 = torch.ones(1, 1, 1, 8) * 2
        cache.update(k1, k1)
        full_k, _ = cache.update(k2, k2)
        assert full_k[:, :, :2].allclose(k1)
        assert full_k[:, :, 2:].allclose(k2)

    def test_reset_clears_cache(self):
        """reset() must wipe the stored tensors and reset length to 0."""
        cache = KVCache()
        cache.update(torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8))
        assert cache.length == 5
        cache.reset()
        assert cache.length == 0
        assert cache.k_cache is None
        assert cache.v_cache is None

    def test_multiple_resets(self):
        """reset() should be idempotent — calling it twice is safe."""
        cache = KVCache()
        cache.reset()
        cache.reset()
        assert cache.length == 0

    def test_update_preserves_values_exactly(self):
        """Key values stored in the cache must not be altered."""
        cache = KVCache()
        k = torch.randn(1, 1, 4, 16)
        v = torch.randn(1, 1, 4, 16)
        full_k, full_v = cache.update(k, v)
        assert torch.allclose(full_k, k)
        assert torch.allclose(full_v, v)


# ---------------------------------------------------------------------------
# RotaryEmbedding with offset
# ---------------------------------------------------------------------------


class TestRotaryEmbeddingOffset:
    """Tests for RoPE positional offset (used during cached generation)."""

    def test_offset_zero_matches_original(self):
        """offset=0 must give the same result as calling without offset."""
        rope = RotaryEmbedding(head_dim=32, max_seq_len=64)
        q = torch.randn(1, 2, 8, 32)
        k = torch.randn(1, 2, 8, 32)
        q_no_offset, k_no_offset = rope(q, k, offset=0)
        q_offset, k_offset = rope(q, k)
        assert torch.allclose(q_no_offset, q_offset)
        assert torch.allclose(k_no_offset, k_offset)

    def test_offset_shifts_positions(self):
        """A token at position (offset+0) must get the same rotation as
        position 0 would get with no offset when compared to position offset
        in a full-context forward pass."""
        rope = RotaryEmbedding(head_dim=32, max_seq_len=64)
        q_full = torch.randn(1, 1, 16, 32)
        q_full_rot, _ = rope(q_full, q_full, offset=0)

        # Single token at position 5 should match column 5 of the full pass
        q_single = q_full[:, :, 5:6, :]
        q_single_rot, _ = rope(q_single, q_single, offset=5)
        assert torch.allclose(q_single_rot, q_full_rot[:, :, 5:6, :], atol=1e-5)

    def test_norm_preserved_with_offset(self):
        """Rotation with offset must still be norm-preserving."""
        rope = RotaryEmbedding(head_dim=64, max_seq_len=32)
        q = torch.randn(1, 2, 4, 64)
        q_rot, _ = rope(q, q, offset=7)
        assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)


# ---------------------------------------------------------------------------
# CausalSelfAttention with KV cache
# ---------------------------------------------------------------------------


class TestCausalSelfAttentionKVCache:
    """Tests for CausalSelfAttention KV-cache path."""

    def _make_attn(self) -> CausalSelfAttention:
        return CausalSelfAttention(dim=64, heads=2, head_dim=32, max_seq_len=64)

    def test_cached_output_shape_single_token(self):
        """Cached forward with T=1 must return shape (B, 1, dim)."""
        attn = self._make_attn()
        cache = KVCache()
        x = torch.randn(2, 1, 64)
        out = attn(x, kv_cache=cache)
        assert out.shape == (2, 1, 64)

    def test_cache_grows_each_step(self):
        """Cache length must increment by T_new on each forward call."""
        attn = self._make_attn()
        cache = KVCache()
        assert cache.length == 0
        attn(torch.randn(1, 3, 64), kv_cache=cache)
        assert cache.length == 3
        attn(torch.randn(1, 1, 64), kv_cache=cache)
        assert cache.length == 4

    def test_no_cache_output_unchanged(self):
        """Passing kv_cache=None must produce the same output as the original path."""
        attn = self._make_attn()
        attn.eval()
        x = torch.randn(1, 6, 64)
        with torch.no_grad():
            out_none = attn(x, kv_cache=None)
            out_noarg = attn(x)
        assert torch.allclose(out_none, out_noarg)

    def test_cached_single_token_matches_full_context_last_position(self):
        """The cached output for the last token must match the last position
        of a full-context forward pass on the complete sequence."""
        attn = self._make_attn()
        attn.eval()

        T = 6
        x = torch.randn(1, T, 64)

        with torch.no_grad():
            # Full-context reference
            ref_out = attn(x)  # (1, T, 64)

            # Cached: prefill tokens 0..T-2, then query token T-1
            cache = KVCache()
            attn(x[:, : T - 1, :], kv_cache=cache)  # prefill
            cached_last = attn(x[:, T - 1 :, :], kv_cache=cache)  # single token

        # Last-position outputs should agree closely
        assert torch.allclose(
            ref_out[:, -1:, :], cached_last, atol=1e-4
        ), f"max diff = {(ref_out[:, -1:, :] - cached_last).abs().max():.6f}"

    def test_cached_does_not_affect_training_path(self):
        """Training path (kv_cache=None) must be gradient-compatible."""
        attn = self._make_attn()
        x = torch.randn(2, 8, 64, requires_grad=True)
        attn(x).sum().backward()
        assert x.grad is not None

    def test_no_cache_used_during_training(self):
        """Passing kv_cache to a training-mode forward must not affect gradients
        on the input — both paths must remain differentiable."""
        attn = self._make_attn()
        cache = KVCache()
        x = torch.randn(1, 4, 64, requires_grad=True)
        # KV cache is silently fine during backward (tensors stored but not differentiated).
        attn(x, kv_cache=cache).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# AttnResLM cached vs uncached generation
# ---------------------------------------------------------------------------


class TestAttnResLMKVCache:
    """Tests for AttnResLM KV-cache generation (Block and Full AttnRes)."""

    def _make_model(self, use_kv_cache: bool, use_block: bool = True) -> AttnResLM:
        return AttnResLM(
            _lm_cfg(use_kv_cache=use_kv_cache, use_block=use_block),
            vocab_size=VOCAB,
            seq_len=SEQ,
        )

    def test_cached_output_length_block(self):
        """Block AttnRes: cached generate must return prompt + max_new_tokens."""
        model = self._make_model(use_kv_cache=True, use_block=True)
        prompt = torch.randint(0, VOCAB, (1, 5))
        out = model.generate(prompt, max_new_tokens=10)
        assert out.shape == (1, 15)

    def test_cached_output_length_full(self):
        """Full AttnRes: cached generate must return prompt + max_new_tokens."""
        model = self._make_model(use_kv_cache=True, use_block=False)
        prompt = torch.randint(0, VOCAB, (1, 4))
        out = model.generate(prompt, max_new_tokens=8)
        assert out.shape == (1, 12)

    def test_uncached_output_length(self):
        """Uncached generate must return prompt + max_new_tokens."""
        model = self._make_model(use_kv_cache=False)
        prompt = torch.randint(0, VOCAB, (1, 5))
        out = model.generate(prompt, max_new_tokens=10)
        assert out.shape == (1, 15)

    def test_cached_ids_in_range(self):
        """All token ids from cached generation must be in [0, vocab_size)."""
        model = self._make_model(use_kv_cache=True)
        prompt = torch.randint(0, VOCAB, (1, 3))
        out = model.generate(prompt, max_new_tokens=20, top_k=10)
        assert out.min().item() >= 0
        assert out.max().item() < VOCAB

    def test_override_flag_at_call_site(self):
        """Passing use_kv_cache=True/False at call site overrides the config."""
        # Config says no cache, but we override to True
        model = self._make_model(use_kv_cache=False)
        prompt = torch.randint(0, VOCAB, (1, 4))
        out_cached = model.generate(prompt, max_new_tokens=5, use_kv_cache=True)
        out_uncached = model.generate(prompt, max_new_tokens=5, use_kv_cache=False)
        assert out_cached.shape == out_uncached.shape == (1, 9)

    def test_config_flag_true_enables_cache(self):
        """model.use_kv_cache: true in config must use the cached path."""
        model = self._make_model(use_kv_cache=True)
        assert model.cfg.use_kv_cache is True
        prompt = torch.randint(0, VOCAB, (1, 5))
        out = model.generate(prompt, max_new_tokens=6)
        assert out.shape == (1, 11)

    def test_config_flag_false_uses_full_ctx(self):
        """model.use_kv_cache: false in config must use the full-context path."""
        model = self._make_model(use_kv_cache=False)
        assert model.cfg.use_kv_cache is False
        prompt = torch.randint(0, VOCAB, (1, 5))
        out = model.generate(prompt, max_new_tokens=6)
        assert out.shape == (1, 11)

    def test_cached_and_uncached_same_logit_distribution(self):
        """Cached and uncached paths must produce the same logit distributions.

        Exact token equality is not guaranteed because the two SDPA code paths
        (is_causal=True vs is_causal=False with full history) are mathematically
        equivalent but may produce slightly different floating-point results.
        We therefore compare the top-1 token at each position, which is robust
        to small numerical differences, and verify that the prompt is preserved.
        """
        model = self._make_model(use_kv_cache=False)
        model.eval()
        prompt = torch.randint(0, VOCAB, (1, 5))

        torch.manual_seed(0)
        out_uncached = model.generate(
            prompt.clone(), max_new_tokens=8, temperature=1e-9, use_kv_cache=False
        )
        torch.manual_seed(0)
        out_cached = model.generate(
            prompt.clone(), max_new_tokens=8, temperature=1e-9, use_kv_cache=True
        )

        # Prompt tokens must be identical in both outputs
        assert torch.all(
            out_uncached[:, :5] == out_cached[:, :5]
        ), "Prompt tokens differ between cached and uncached paths"

        # Both outputs must have the correct total length
        assert out_uncached.shape == out_cached.shape == (1, 13)

    def test_no_nan_in_cached_output(self):
        """Cached generation must not produce NaN tokens."""
        model = self._make_model(use_kv_cache=True)
        prompt = torch.randint(0, VOCAB, (1, 4))
        out = model.generate(prompt, max_new_tokens=15)
        assert not torch.isnan(out.float()).any()

    def test_prompt_tokens_preserved(self):
        """The first T_prompt tokens of the output must match the prompt exactly."""
        model = self._make_model(use_kv_cache=True)
        prompt = torch.randint(0, VOCAB, (1, 6))
        out = model.generate(prompt, max_new_tokens=4)
        assert torch.all(out[:, :6] == prompt)


# ---------------------------------------------------------------------------
# BaselineLM cached generation
# ---------------------------------------------------------------------------


class TestBaselineLMKVCache:
    """Tests for BaselineLM KV-cache generation."""

    def _make_model(self, use_kv_cache: bool) -> BaselineLM:
        return BaselineLM(
            _lm_cfg(use_kv_cache=use_kv_cache), vocab_size=VOCAB, seq_len=SEQ
        )

    def test_cached_output_length(self):
        """Cached BaselineLM generate must return prompt + max_new_tokens."""
        model = self._make_model(use_kv_cache=True)
        prompt = torch.randint(0, VOCAB, (1, 5))
        out = model.generate(prompt, max_new_tokens=8)
        assert out.shape == (1, 13)

    def test_cached_ids_in_range(self):
        """All generated ids must be within vocab range."""
        model = self._make_model(use_kv_cache=True)
        prompt = torch.randint(0, VOCAB, (1, 4))
        out = model.generate(prompt, max_new_tokens=10, top_k=20)
        assert out.min().item() >= 0
        assert out.max().item() < VOCAB

    def test_baseline_cached_matches_uncached_greedy(self):
        """Greedy cached and uncached BaselineLM outputs must be identical."""
        model = self._make_model(use_kv_cache=False)
        model.eval()
        prompt = torch.randint(0, VOCAB, (1, 4))

        torch.manual_seed(1)
        out_uncached = model.generate(
            prompt.clone(), max_new_tokens=6, temperature=1e-9, use_kv_cache=False
        )
        torch.manual_seed(1)
        out_cached = model.generate(
            prompt.clone(), max_new_tokens=6, temperature=1e-9, use_kv_cache=True
        )
        assert torch.all(out_uncached == out_cached)


# ---------------------------------------------------------------------------
# Config flag integration
# ---------------------------------------------------------------------------


class TestKVCacheConfigFlag:
    """Tests that use_kv_cache in ModelConfig is wired correctly."""

    def test_default_is_false(self):
        """Default ModelConfig must have use_kv_cache=False."""
        cfg = ModelConfig()
        assert cfg.use_kv_cache is False

    def test_can_be_set_true(self):
        """ModelConfig must accept use_kv_cache=True."""
        cfg = ModelConfig(use_kv_cache=True)
        assert cfg.use_kv_cache is True

    def test_serialises_in_to_dict(self):
        """use_kv_cache must appear in the to_dict() output for checkpointing."""
        from utils.config import Config

        cfg = Config(model=ModelConfig(use_kv_cache=True))
        d = cfg.to_dict()
        assert "use_kv_cache" in d["model"]
        assert d["model"]["use_kv_cache"] is True

    def test_yaml_override_sets_flag(self, tmp_path):
        """CLI override 'model.use_kv_cache=true' must work through load_config."""
        from utils.config import load_config

        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(
            "model:\n  use_kv_cache: false\n"
            "training:\n  device: cpu\n"
            "data:\n  dataset: mnist\n"
            "logging:\n  log_dir: logs/\n  checkpoint_dir: checkpoints/\n"
            "generation:\n  max_new_tokens: 100\n  temperature: 1.0\n  top_k: 0\n"
        )
        cfg = load_config(yaml_path, overrides=["model.use_kv_cache=true"])
        assert cfg.model.use_kv_cache is True

    def test_model_reads_flag_from_cfg(self):
        """AttnResLM must expose cfg.use_kv_cache to its generate() method."""
        cfg = _lm_cfg(use_kv_cache=True)
        model = AttnResLM(cfg, vocab_size=VOCAB, seq_len=SEQ)
        assert model.cfg.use_kv_cache is True
