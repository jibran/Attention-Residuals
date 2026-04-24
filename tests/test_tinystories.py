"""Unit tests for the TinyStories dataset pipeline.

All tests run fully offline — no HuggingFace download is required.  We mock
the HF ``load_dataset`` call and build a tiny synthetic corpus so the tests
complete in milliseconds.

Covers:
  * :class:`~dataset.tinystories_dataset._WindowDataset` (reused from Shakespeare)
  * :class:`~dataset.tinystories_dataset.TinyStoriesDataset` tokenisation,
    caching, and DataLoader construction
  * ``train_lm.py`` dataset dispatcher for ``"tinystories"``
  * Config flag ``data.dataset = "tinystories"``
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.tinystories_dataset import TinyStoriesDataset, _WindowDataset

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

STORIES = [
    "Once upon a time there was a little cat.",
    "The dog ran fast across the green field.",
    "A small bird flew over the tall tree.",
    "There was a happy child who loved to play.",
    "One day the sun shone bright in the sky.",
]


class _FakeTokenizer:
    """Minimal tokeniser that maps chars to ids (stand-in for GPT-Neo)."""

    vocab_size: int = 128
    eos_token_id: int = 0
    pad_token: str = "<|endoftext|>"

    def encode(self, text, **kwargs):
        """Return a list of int IDs (one per character)."""
        if isinstance(text, list):
            return {"input_ids": [[ord(c) % 127 + 1 for c in t] for t in text]}
        return [ord(c) % 127 + 1 for c in text]

    def __call__(self, texts, **kwargs):
        if isinstance(texts, list):
            return {"input_ids": [[ord(c) % 127 + 1 for c in t] for t in texts]}
        return {"input_ids": [ord(c) % 127 + 1 for c in texts]}

    def decode(self, ids):
        return "".join(chr((i % 127) + 1) if i > 0 else "" for i in ids)


def _make_hf_dataset(stories=None):
    """Return a minimal HF-style dataset object."""
    stories = stories or STORIES
    ds = MagicMock()
    ds.__len__ = MagicMock(return_value=len(stories))
    ds.__getitem__ = MagicMock(
        side_effect=lambda s: {
            "text": (
                stories[s]
                if isinstance(s, int)
                else [stories[i] for i in range(*s.indices(len(stories)))]
            )
        }
    )
    ds.select = MagicMock(return_value=ds)

    # Support slicing ds[start:end] to return {"text": [...]}
    class _Indexable:
        def __init__(self, data):
            self._data = data

        def __len__(self):
            return len(self._data)

        def __getitem__(self, key):
            if isinstance(key, slice):
                return {"text": self._data[key]}
            return {"text": self._data[key]}

        def select(self, indices):
            return _Indexable([self._data[i] for i in indices])

    return _Indexable(stories)


# ---------------------------------------------------------------------------
# _WindowDataset (also tested in test_shakespeare; verify it works for BPE)
# ---------------------------------------------------------------------------


class TestWindowDatasetBPE:
    """Verify _WindowDataset with BPE-style (longer) token sequences."""

    def _tokens(self, n=2000):
        return torch.arange(n, dtype=torch.long)

    def test_length_non_overlapping(self):
        """Non-overlapping stride gives floor(N-seq-1 / seq) + 1 windows."""
        ds = _WindowDataset(self._tokens(2000), seq_len=128, stride=128)
        expected = (2000 - 128 - 1) // 128 + 1
        assert len(ds) == expected

    def test_x_y_shapes(self):
        """Each item must return tensors of shape (seq_len,)."""
        ds = _WindowDataset(self._tokens(1000), seq_len=64)
        x, y = ds[0]
        assert x.shape == (64,)
        assert y.shape == (64,)

    def test_y_is_x_shifted(self):
        """y == x shifted left by 1 for BPE token ids."""
        tokens = torch.arange(500, dtype=torch.long)
        ds = _WindowDataset(tokens, seq_len=50)
        x, y = ds[0]
        assert torch.all(y == x + 1)

    def test_empty_for_short_corpus(self):
        """A corpus shorter than seq_len + 1 must give 0 windows."""
        ds = _WindowDataset(torch.arange(10, dtype=torch.long), seq_len=50)
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# TinyStoriesDataset tokenisation + caching
# ---------------------------------------------------------------------------


class TestTinyStoriesDatasetTokenisation:
    """Test the tokenisation and disk-caching logic."""

    def _inst(self, tmp_path) -> TinyStoriesDataset:
        return TinyStoriesDataset(data_dir=str(tmp_path))

    @patch("dataset.tinystories_dataset.TinyStoriesDataset._load_tokenizer")
    @patch("dataset.tinystories_dataset.load_dataset")
    def test_tokenise_split_creates_cache(self, mock_load_ds, mock_tok, tmp_path):
        """First call must create a .pt cache file."""
        mock_load_ds.return_value = _make_hf_dataset()
        mock_tok.return_value = _FakeTokenizer()

        inst = self._inst(tmp_path)
        tok = _FakeTokenizer()
        cache = inst.processed_dir / "test_cache.pt"
        tokens = inst._tokenise_split("train", tok, cache)

        assert cache.exists()
        assert isinstance(tokens, torch.Tensor)
        assert tokens.dtype == torch.long
        assert tokens.numel() > 0

    @patch("dataset.tinystories_dataset.TinyStoriesDataset._load_tokenizer")
    @patch("dataset.tinystories_dataset.load_dataset")
    def test_cache_loaded_on_second_call(self, mock_load_ds, mock_tok, tmp_path):
        """Second call must load from cache without calling load_dataset again."""
        mock_load_ds.return_value = _make_hf_dataset()
        mock_tok.return_value = _FakeTokenizer()

        inst = self._inst(tmp_path)
        tok = _FakeTokenizer()
        cache = inst.processed_dir / "test_cache.pt"

        inst._tokenise_split("train", tok, cache)
        first_call_count = mock_load_ds.call_count

        inst._tokenise_split("train", tok, cache)  # should use cache
        assert mock_load_ds.call_count == first_call_count  # no extra call

    @patch("dataset.tinystories_dataset.TinyStoriesDataset._load_tokenizer")
    @patch("dataset.tinystories_dataset.load_dataset")
    def test_eos_separates_stories(self, mock_load_ds, mock_tok, tmp_path):
        """EOS tokens (id=0) must appear between stories."""
        mock_load_ds.return_value = _make_hf_dataset()
        tok = _FakeTokenizer()
        mock_tok.return_value = tok

        inst = self._inst(tmp_path)
        cache = inst.processed_dir / "eos_test.pt"
        tokens = inst._tokenise_split("train", tok, cache)

        # EOS (id=0 for fake tokeniser) should appear at least len(STORIES)-1 times
        eos_count = (tokens == tok.eos_token_id).sum().item()
        assert eos_count >= len(STORIES) - 1

    @patch("dataset.tinystories_dataset.TinyStoriesDataset._load_tokenizer")
    @patch("dataset.tinystories_dataset.load_dataset")
    def test_max_stories_truncates(self, mock_load_ds, mock_tok, tmp_path):
        """max_stories parameter must cap the number of stories processed."""
        full_ds = _make_hf_dataset(STORIES)
        trimmed_ds = _make_hf_dataset(STORIES[:2])
        mock_load_ds.return_value = full_ds
        tok = _FakeTokenizer()
        mock_tok.return_value = tok

        inst = self._inst(tmp_path)

        # Full
        full_cache = inst.processed_dir / "full.pt"
        full_tokens = inst._tokenise_split("train", tok, full_cache)

        # Truncated — use a different cache so it re-tokenises
        inst2 = TinyStoriesDataset(data_dir=str(tmp_path / "sub"))
        inst2.processed_dir.mkdir(parents=True, exist_ok=True)
        mock_load_ds.return_value = trimmed_ds
        trunc_cache = inst2.processed_dir / "trunc.pt"
        trunc_tokens = inst2._tokenise_split("train", tok, trunc_cache, max_stories=2)

        assert trunc_tokens.numel() < full_tokens.numel()


# ---------------------------------------------------------------------------
# TinyStoriesDataset.get_loaders
# ---------------------------------------------------------------------------


class TestTinyStoriesGetLoaders:
    """Integration tests for the full get_loaders factory."""

    @patch("dataset.tinystories_dataset.TinyStoriesDataset._load_tokenizer")
    @patch("dataset.tinystories_dataset.load_dataset")
    def test_returns_loaders_and_tokenizer(self, mock_load_ds, mock_tok, tmp_path):
        """get_loaders must return (dict, tokenizer)."""
        mock_load_ds.return_value = _make_hf_dataset()
        fake_tok = _FakeTokenizer()
        mock_tok.return_value = fake_tok

        loaders, tok = TinyStoriesDataset.get_loaders(
            data_dir=str(tmp_path),
            seq_len=32,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        assert "train" in loaders
        assert "val" in loaders
        assert tok is fake_tok

    @patch("dataset.tinystories_dataset.TinyStoriesDataset._load_tokenizer")
    @patch("dataset.tinystories_dataset.load_dataset")
    def test_batch_shapes(self, mock_load_ds, mock_tok, tmp_path):
        """Batches from the train loader must have shape (B, seq_len)."""
        mock_load_ds.return_value = _make_hf_dataset(STORIES * 20)
        mock_tok.return_value = _FakeTokenizer()

        loaders, _ = TinyStoriesDataset.get_loaders(
            data_dir=str(tmp_path),
            seq_len=16,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
        )
        x, y = next(iter(loaders["train"]))
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)
        assert x.dtype == torch.long

    @patch("dataset.tinystories_dataset.TinyStoriesDataset._load_tokenizer")
    @patch("dataset.tinystories_dataset.load_dataset")
    def test_vocab_size_exposed(self, mock_load_ds, mock_tok, tmp_path):
        """The returned tokeniser must expose vocab_size."""
        mock_load_ds.return_value = _make_hf_dataset()
        mock_tok.return_value = _FakeTokenizer()

        _, tok = TinyStoriesDataset.get_loaders(
            data_dir=str(tmp_path),
            seq_len=16,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        assert tok.vocab_size == 128


# ---------------------------------------------------------------------------
# train_lm dispatcher
# ---------------------------------------------------------------------------


class TestTrainLMDispatcher:
    """Tests for the dataset dispatcher in train_lm.py."""

    def test_tinystories_key_recognised(self, tmp_path):
        """_load_dataset must not raise for dataset='tinystories'."""
        import torch

        from train.train_lm import _load_dataset
        from utils.config import Config, DataConfig

        fake_tok = _FakeTokenizer()
        fake_tokens = torch.arange(1000, dtype=torch.long)
        from torch.utils.data import DataLoader

        from dataset.tinystories_dataset import _WindowDataset

        fake_ds = _WindowDataset(fake_tokens, seq_len=16)
        fake_loaders = {
            "train": DataLoader(fake_ds, batch_size=2),
            "val": DataLoader(fake_ds, batch_size=2),
        }

        with patch(
            "train.train_lm.TinyStoriesDataset.get_loaders",
            return_value=(fake_loaders, fake_tok),
        ):
            cfg = Config(data=DataConfig(dataset="tinystories", seq_len=16))
            loaders, tok = _load_dataset(cfg, torch.device("cpu"))
            assert "train" in loaders
            assert tok is fake_tok

    def test_shakespeare_key_recognised(self, tmp_path):
        """_load_dataset must not raise for dataset='shakespeare'."""
        import torch

        from train.train_lm import _load_dataset
        from utils.config import Config, DataConfig

        fake_tok = MagicMock()
        fake_tok.vocab_size = 67
        fake_tokens = torch.arange(500, dtype=torch.long)
        from torch.utils.data import DataLoader

        from dataset.tinystories_dataset import _WindowDataset

        fake_ds = _WindowDataset(fake_tokens, seq_len=16)
        fake_loaders = {
            "train": DataLoader(fake_ds, batch_size=2),
            "val": DataLoader(fake_ds, batch_size=2),
        }

        with patch(
            "train.train_lm.ShakespeareDataset.get_loaders",
            return_value=(fake_loaders, fake_tok),
        ):
            cfg = Config(data=DataConfig(dataset="shakespeare", seq_len=16))
            loaders, tok = _load_dataset(cfg, torch.device("cpu"))
            assert "train" in loaders

    def test_unknown_dataset_raises(self):
        """_load_dataset must raise ValueError for an unknown dataset name."""
        import torch

        from train.train_lm import _load_dataset
        from utils.config import Config, DataConfig

        cfg = Config(data=DataConfig(dataset="imagenet"))
        with pytest.raises(ValueError, match="Unknown dataset"):
            _load_dataset(cfg, torch.device("cpu"))


# ---------------------------------------------------------------------------
# Config flag
# ---------------------------------------------------------------------------


class TestTinyStoriesConfig:
    """Tests for TinyStories-related config fields."""

    def test_tinystories_yaml_loads(self, tmp_path):
        """config/tinystories.yaml must load without errors."""
        from utils.config import load_config

        yaml_path = Path(__file__).resolve().parents[1] / "config" / "tinystories.yaml"
        if not yaml_path.exists():
            pytest.skip("tinystories.yaml not found")
        cfg = load_config(yaml_path)
        assert cfg.data.dataset == "tinystories"
        assert cfg.data.seq_len == 512
        assert cfg.model.dim == 512

    def test_tinystories_vocab_size_in_model(self):
        """AttnResLM must accept vocab_size=50257 without error."""
        from models import build_lm
        from utils.config import ModelConfig

        cfg = ModelConfig(dim=32, depth=2, heads=2, head_dim=16, max_seq_len=16)
        model = build_lm(cfg, vocab_size=50257, seq_len=16)
        x = torch.randint(0, 50257, (1, 16))
        logits, _ = model(x)
        assert logits.shape == (1, 16, 50257)
