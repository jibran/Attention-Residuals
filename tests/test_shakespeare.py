"""Tests for the Shakespeare text pipeline and language-model variants.

Covers:
  * :class:`~dataset.tokenizer.CharTokenizer`
  * :class:`~dataset.shakespeare_dataset._WindowDataset`
  * :class:`~models.lm_transformer.AttnResLM`
  * :class:`~models.lm_transformer.BaselineLM`
  * :func:`~models.build_lm` factory
  * Config :class:`~utils.config.GenerationConfig` round-trip
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.shakespeare_dataset import _WindowDataset
from dataset.tokenizer import CharTokenizer
from models import build_lm
from models.lm_transformer import AttnResLM, BaselineLM
from utils.config import Config, GenerationConfig, ModelConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "ROMEO: But soft, what light through yonder window breaks?\n"
    "JULIET: O Romeo, Romeo! Wherefore art thou Romeo?\n"
    "HAMLET: To be, or not to be, that is the question.\n"
)


def _small_cfg(**overrides) -> ModelConfig:
    """Return a tiny ModelConfig suitable for fast CPU tests."""
    defaults = dict(
        name="TestLM",
        dim=32,
        depth=2,
        heads=2,
        head_dim=16,
        mlp_multiplier=2,
        dropout=0.0,
        use_block_attn_res=True,
        block_size=2,
        norm_eps=1e-6,
        max_seq_len=64,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


# ---------------------------------------------------------------------------
# CharTokenizer
# ---------------------------------------------------------------------------


class TestCharTokenizer:
    """Tests for the character-level tokeniser."""

    def test_vocab_contains_all_chars(self):
        """Every character in the source text must appear in the vocabulary."""
        tok = CharTokenizer.from_text(SAMPLE_TEXT)
        for ch in set(SAMPLE_TEXT):
            assert ch in tok._stoi, f"Missing char: {repr(ch)}"

    def test_specials_always_present(self):
        """PAD and UNK tokens must always be in the vocabulary."""
        tok = CharTokenizer.from_text("abc")
        assert tok.pad_id == 0
        assert tok.unk_id == 1

    def test_encode_decode_roundtrip(self):
        """encode then decode must recover the original string exactly."""
        tok = CharTokenizer.from_text(SAMPLE_TEXT)
        ids = tok.encode(SAMPLE_TEXT)
        assert tok.decode(ids) == SAMPLE_TEXT

    def test_encode_returns_list_of_ints(self):
        """encode must return a list of integers."""
        tok = CharTokenizer.from_text("hello")
        ids = tok.encode("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_unknown_char_maps_to_unk(self):
        """Characters not in the vocabulary must map to unk_id."""
        tok = CharTokenizer.from_text("abc")
        ids = tok.encode("xyz")
        assert all(i == tok.unk_id for i in ids)

    def test_decode_skips_pad(self):
        """Padding tokens must not appear in the decoded string."""
        tok = CharTokenizer.from_text("abc")
        ids = [tok.pad_id, tok._stoi["a"], tok.pad_id, tok._stoi["b"]]
        assert tok.decode(ids) == "ab"

    def test_vocab_size(self):
        """vocab_size must equal number of unique chars + 2 specials."""
        text = "abcde"
        tok = CharTokenizer.from_text(text)
        assert tok.vocab_size == len(set(text)) + 2  # +PAD +UNK

    def test_len_equals_vocab_size(self):
        """__len__ must return the same value as vocab_size."""
        tok = CharTokenizer.from_text(SAMPLE_TEXT)
        assert len(tok) == tok.vocab_size

    def test_save_and_load(self, tmp_path):
        """save/load round-trip must preserve vocabulary perfectly."""
        tok = CharTokenizer.from_text(SAMPLE_TEXT)
        path = tmp_path / "vocab.json"
        tok.save(path)

        tok2 = CharTokenizer.load(path)
        assert tok.vocab_size == tok2.vocab_size
        assert tok._stoi == tok2._stoi
        # Encode → decode must still work after reload
        ids = tok2.encode(SAMPLE_TEXT)
        assert tok2.decode(ids) == SAMPLE_TEXT

    def test_load_missing_file_raises(self, tmp_path):
        """Loading a non-existent file must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CharTokenizer.load(tmp_path / "missing.json")

    def test_save_creates_valid_json(self, tmp_path):
        """The saved file must be valid JSON with 'stoi' and 'itos' keys."""
        tok = CharTokenizer.from_text("hello")
        path = tmp_path / "vocab.json"
        tok.save(path)
        data = json.loads(path.read_text())
        assert "stoi" in data
        assert "itos" in data


# ---------------------------------------------------------------------------
# _WindowDataset
# ---------------------------------------------------------------------------


class TestWindowDataset:
    """Tests for the sliding-window token dataset."""

    def _make_tokens(self, n: int = 1000) -> torch.Tensor:
        return torch.arange(n, dtype=torch.long)

    def test_len_non_overlapping(self):
        """Non-overlapping windows: length must be (N - seq_len - 1) // seq_len + 1."""
        tokens = self._make_tokens(1000)
        ds = _WindowDataset(tokens, seq_len=100, stride=100)
        expected = (1000 - 100 - 1) // 100 + 1
        assert len(ds) == expected

    def test_len_overlapping(self):
        """Overlapping windows must produce more samples than non-overlapping."""
        tokens = self._make_tokens(1000)
        ds_full = _WindowDataset(tokens, seq_len=100, stride=100)
        ds_half = _WindowDataset(tokens, seq_len=100, stride=50)
        assert len(ds_half) > len(ds_full)

    def test_item_shapes(self):
        """Each item must return tensors of shape (seq_len,)."""
        tokens = self._make_tokens(500)
        ds = _WindowDataset(tokens, seq_len=64)
        x, y = ds[0]
        assert x.shape == (64,)
        assert y.shape == (64,)

    def test_target_is_shifted_input(self):
        """y must equal x shifted left by one position."""
        tokens = torch.arange(200, dtype=torch.long)
        ds = _WindowDataset(tokens, seq_len=50)
        x, y = ds[0]
        assert torch.all(y == x + 1)

    def test_consecutive_windows_offset(self):
        """Second window must start stride positions after the first."""
        tokens = torch.arange(500, dtype=torch.long)
        stride = 32
        ds = _WindowDataset(tokens, seq_len=64, stride=stride)
        x0, _ = ds[0]
        x1, _ = ds[1]
        assert x1[0].item() == x0[0].item() + stride

    def test_empty_for_short_sequence(self):
        """A token sequence shorter than seq_len + 1 must yield 0 windows."""
        tokens = torch.arange(10, dtype=torch.long)
        ds = _WindowDataset(tokens, seq_len=20)
        assert len(ds) == 0

    def test_dtype_long(self):
        """Items must have dtype torch.long."""
        ds = _WindowDataset(torch.arange(200, dtype=torch.long), seq_len=64)
        x, y = ds[0]
        assert x.dtype == torch.long
        assert y.dtype == torch.long


# ---------------------------------------------------------------------------
# AttnResLM
# ---------------------------------------------------------------------------

VOCAB = 67  # typical Shakespeare char vocab size
SEQ = 32  # short window for tests
B = 4


class TestAttnResLM:
    """Integration tests for the AttnRes language model."""

    def _make_model(self, **overrides) -> AttnResLM:
        return AttnResLM(_small_cfg(**overrides), vocab_size=VOCAB, seq_len=SEQ)

    def test_logits_shape(self):
        """Forward pass must return logits of shape (B, T, vocab_size)."""
        model = self._make_model()
        x = torch.randint(0, VOCAB, (B, SEQ))
        logits, loss = model(x)
        assert logits.shape == (B, SEQ, VOCAB)
        assert loss is None

    def test_loss_computed_with_targets(self):
        """Supplying targets must return a scalar loss."""
        model = self._make_model()
        x = torch.randint(0, VOCAB, (B, SEQ))
        y = torch.randint(0, VOCAB, (B, SEQ))
        _, loss = model(x, y)
        assert loss is not None
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_gradient_flows(self):
        """Back-propagating the loss must produce gradients on all parameters."""
        model = self._make_model()
        x = torch.randint(0, VOCAB, (B, SEQ))
        y = torch.randint(0, VOCAB, (B, SEQ))
        _, loss = model(x, y)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_weight_tying(self):
        """Embedding weight and output projection weight must be the same object."""
        model = self._make_model()
        assert model.tok_embed.weight is model.head.weight

    def test_generate_shape(self):
        """generate must return a tensor of shape (1, prompt_len + max_new_tokens)."""
        model = self._make_model()
        prompt = torch.randint(0, VOCAB, (1, 5))
        out = model.generate(prompt, max_new_tokens=10)
        assert out.shape == (1, 15)

    def test_generate_ids_in_range(self):
        """All generated token ids must be within [0, vocab_size)."""
        model = self._make_model()
        prompt = torch.randint(0, VOCAB, (1, 4))
        out = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=10)
        assert out.min().item() >= 0
        assert out.max().item() < VOCAB

    def test_no_nan_output(self):
        """Logits must be finite for random inputs."""
        model = self._make_model()
        x = torch.randint(0, VOCAB, (B, SEQ))
        logits, _ = model(x)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_full_attnres_variant(self):
        """Full AttnRes variant must also produce correct output shape."""
        model = self._make_model(use_block_attn_res=False)
        x = torch.randint(0, VOCAB, (B, SEQ))
        logits, _ = model(x)
        assert logits.shape == (B, SEQ, VOCAB)

    def test_param_count_positive(self):
        """num_parameters must be a positive integer."""
        model = self._make_model()
        assert model.num_parameters > 0

    def test_seq_length_assertion(self):
        """Passing a sequence longer than seq_len must raise AssertionError."""
        model = self._make_model()
        x = torch.randint(0, VOCAB, (1, SEQ + 10))
        with pytest.raises(AssertionError):
            model(x)


# ---------------------------------------------------------------------------
# BaselineLM
# ---------------------------------------------------------------------------


class TestBaselineLM:
    """Tests for the standard-residual baseline language model."""

    def test_logits_shape(self):
        """Baseline must return (B, T, vocab_size) logits."""
        model = BaselineLM(_small_cfg(), vocab_size=VOCAB, seq_len=SEQ)
        x = torch.randint(0, VOCAB, (B, SEQ))
        logits, _ = model(x)
        assert logits.shape == (B, SEQ, VOCAB)

    def test_loss_computed(self):
        """Supplying targets must return a scalar loss for BaselineLM."""
        model = BaselineLM(_small_cfg(), vocab_size=VOCAB, seq_len=SEQ)
        x = torch.randint(0, VOCAB, (B, SEQ))
        y = torch.randint(0, VOCAB, (B, SEQ))
        _, loss = model(x, y)
        assert loss is not None and loss.shape == ()

    def test_generate(self):
        """BaselineLM.generate must return the correct length tensor."""
        model = BaselineLM(_small_cfg(), vocab_size=VOCAB, seq_len=SEQ)
        prompt = torch.randint(0, VOCAB, (1, 5))
        out = model.generate(prompt, max_new_tokens=8)
        assert out.shape == (1, 13)


# ---------------------------------------------------------------------------
# build_lm factory
# ---------------------------------------------------------------------------


class TestBuildLM:
    """Tests for the build_lm model factory."""

    def test_returns_attnres_lm(self):
        """build_lm without baseline=True must return AttnResLM."""
        model = build_lm(_small_cfg(), vocab_size=VOCAB, seq_len=SEQ, baseline=False)
        assert isinstance(model, AttnResLM)

    def test_returns_baseline_lm(self):
        """build_lm with baseline=True must return BaselineLM."""
        model = build_lm(_small_cfg(), vocab_size=VOCAB, seq_len=SEQ, baseline=True)
        assert isinstance(model, BaselineLM)

    def test_factory_output_shape(self):
        """Factory-built model must produce correct logit shape."""
        model = build_lm(_small_cfg(), vocab_size=VOCAB, seq_len=SEQ)
        x = torch.randint(0, VOCAB, (2, SEQ))
        logits, _ = model(x)
        assert logits.shape == (2, SEQ, VOCAB)


# ---------------------------------------------------------------------------
# GenerationConfig
# ---------------------------------------------------------------------------


class TestGenerationConfig:
    """Tests for GenerationConfig defaults and integration with Config."""

    def test_defaults(self):
        """GenerationConfig must have sensible defaults."""
        gcfg = GenerationConfig()
        assert gcfg.max_new_tokens > 0
        assert 0.0 < gcfg.temperature <= 2.0
        assert gcfg.top_k >= 0

    def test_included_in_config(self):
        """Config.generation must be a GenerationConfig instance."""
        cfg = Config()
        assert isinstance(cfg.generation, GenerationConfig)

    def test_to_dict_includes_generation(self):
        """Config.to_dict() must include a 'generation' key."""
        cfg = Config()
        d = cfg.to_dict()
        assert "generation" in d
        assert "max_new_tokens" in d["generation"]
        assert "temperature" in d["generation"]
        assert "top_k" in d["generation"]
