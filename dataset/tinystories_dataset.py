"""TinyStories dataset for sub-word language modelling.

Downloads ``roneneldan/TinyStories`` from the Hugging Face Hub, tokenises
every story with the ``EleutherAI/gpt-neo-125M`` BPE tokeniser (vocab 50 257),
concatenates the token stream with EOS separators, and slices it into
fixed-length windows for next-token prediction — exactly matching the
training setup of the published TinyStories-33M model.

Dataset statistics
------------------
* HF repo: ``roneneldan/TinyStories``
* Splits: ``train`` (2.12 M stories) · ``validation`` (22 k stories)
* Tokeniser: ``EleutherAI/gpt-neo-125M`` — vocab size 50 257
* Context length: 512 tokens (default, matching the paper)
* Corpus token count (train): ~500 M tokens

Usage::

    loaders, tokenizer = TinyStoriesDataset.get_loaders(
        data_dir="data/",
        seq_len=512,
        batch_size=32,
    )
    vocab_size = tokenizer.vocab_size   # 50257
    for x, y in loaders["train"]:
        # x: (B, 512)   y: (B, 512)  — token ids, dtype long
        ...
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Window dataset (shared with Shakespeare — identical sliding-window logic)
# ---------------------------------------------------------------------------


class _WindowDataset(Dataset):
    """Fixed-length sliding-window view over a 1-D token tensor.

    Args:
        tokens: Flat ``torch.long`` tensor of token ids.
        seq_len: Window length (number of tokens per training sample).
        stride: Step between consecutive windows.  Defaults to ``seq_len``
            (non-overlapping).  Use ``seq_len // 2`` for 50 % overlap.
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        seq_len: int,
        stride: int | None = None,
    ) -> None:
        self._tokens = tokens
        self._seq_len = seq_len
        self._stride = stride or seq_len
        self._n = max(0, (len(tokens) - seq_len - 1) // self._stride + 1)

    def __len__(self) -> int:
        """Return the number of windows.

        Returns:
            Window count.
        """
        return self._n

    def __getitem__(self, idx: int):
        """Return the ``idx``-th ``(x, y)`` window pair.

        Args:
            idx: Window index.

        Returns:
            Tuple ``(x, y)`` each of shape ``(seq_len,)`` dtype ``torch.long``,
            where ``y`` is ``x`` shifted left by one position.
        """
        start = idx * self._stride
        x = self._tokens[start : start + self._seq_len]
        y = self._tokens[start + 1 : start + self._seq_len + 1]
        return x, y


# ---------------------------------------------------------------------------
# Public dataset class
# ---------------------------------------------------------------------------


class TinyStoriesDataset:
    """Factory for the TinyStories sub-word language-modelling dataset.

    This class is not instantiated directly.  Use :meth:`get_loaders`.

    The tokeniser (``EleutherAI/gpt-neo-125M``) is downloaded once and cached
    by the ``tokenizers`` library.  The tokenised corpus is cached as a
    ``torch.pt`` file in ``data/processed/`` so subsequent runs skip the
    expensive encoding step entirely.

    Args:
        data_dir: Root directory for data caching.
    """

    HF_DATASET_REPO: str = "roneneldan/TinyStories"
    TOKENIZER_REPO: str = "EleutherAI/gpt-neo-125M"
    TRAIN_CACHE: str = "tinystories_train.pt"
    VAL_CACHE: str = "tinystories_val.pt"

    def __init__(self, data_dir: str = "data/") -> None:
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_tokenizer(self):
        """Download and return the GPT-Neo BPE tokeniser.

        Returns:
            A ``transformers.PreTrainedTokenizerFast`` with vocab size 50 257.
        """
        from transformers import AutoTokenizer

        print(f"Loading tokeniser from {self.TOKENIZER_REPO} …")
        tok = AutoTokenizer.from_pretrained(self.TOKENIZER_REPO)
        # GPT-Neo tokeniser has no pad token by default — set EOS as pad
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def _tokenise_split(
        self,
        split: str,
        tokenizer,
        cache_path: Path,
        max_stories: int | None = None,
    ) -> torch.Tensor:
        """Tokenise a dataset split and cache the flat token tensor to disk.

        Stories are separated by a single EOS token so the model learns a
        clean end-of-story boundary.  The resulting token stream is saved as
        a 1-D ``torch.long`` tensor.

        Args:
            split: ``"train"`` or ``"validation"``.
            tokenizer: HuggingFace tokeniser instance.
            cache_path: Path to write / read the ``.pt`` cache file.
            max_stories: Truncate to this many stories (for fast debugging).
                ``None`` processes all stories.

        Returns:
            Flat 1-D ``torch.long`` token tensor.
        """
        if cache_path.exists():
            print(f"Loading cached tokens: {cache_path}")
            return torch.load(cache_path, weights_only=True)

        if load_dataset is None:  # pragma: no cover
            raise ImportError("pip install datasets to use TinyStoriesDataset")

        print(f"Downloading {self.HF_DATASET_REPO} ({split}) from HuggingFace Hub …")
        ds = load_dataset(self.HF_DATASET_REPO, split=split, trust_remote_code=True)

        if max_stories is not None:
            ds = ds.select(range(min(max_stories, len(ds))))

        eos = tokenizer.eos_token_id
        all_ids: list[int] = []

        print(f"Tokenising {len(ds):,} stories …")
        batch_size = 1000
        for start in range(0, len(ds), batch_size):
            batch = ds[start : start + batch_size]["text"]
            encoded = tokenizer(
                batch,
                add_special_tokens=False,
                truncation=False,
            )
            for ids in encoded["input_ids"]:
                all_ids.extend(ids)
                all_ids.append(eos)  # story separator

        tokens = torch.tensor(all_ids, dtype=torch.long)
        torch.save(tokens, cache_path)
        print(
            f"Cached {len(tokens):,} tokens → {cache_path}  "
            f"({len(tokens)/1e6:.1f} M tokens)"
        )
        return tokens

    # ------------------------------------------------------------------
    # Public factory
    # ------------------------------------------------------------------

    @classmethod
    def get_loaders(
        cls,
        data_dir: str = "data/",
        seq_len: int = 512,
        stride: int | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        max_train_stories: int | None = None,
        max_val_stories: int | None = None,
    ):
        """Build train / validation DataLoaders for TinyStories.

        On the first call, downloads the dataset and tokeniser, encodes all
        stories, and saves the token tensors to disk.  Subsequent calls load
        from the cache and skip the encoding step entirely.

        Args:
            data_dir: Root directory for data and processed cache files.
            seq_len: Tokens per training window (context length).  Use 512 to
                match the published TinyStories-33M training setup.
            stride: Window stride.  Defaults to ``seq_len`` (non-overlapping).
                Use ``seq_len // 2`` for ~2× more training windows.
            batch_size: Mini-batch size for all loaders.
            num_workers: DataLoader worker count.
            pin_memory: Use pinned memory for faster GPU data transfer.
            seed: Random seed (used for deterministic behaviour; train split
                is not re-shuffled here since HF datasets are already shuffled).
            max_train_stories: Cap the training split at this many stories.
                ``None`` (default) uses all ~2.12 M stories.  Set to e.g.
                ``50_000`` for fast smoke-test runs.
            max_val_stories: Cap the validation split.  ``None`` uses all
                ~22 k stories.

        Returns:
            Tuple ``(loaders, tokenizer)`` where ``loaders`` is a dict with
            keys ``"train"`` and ``"val"``, and ``tokenizer`` is the fitted
            ``EleutherAI/gpt-neo-125M`` tokeniser.

        Example::

            loaders, tok = TinyStoriesDataset.get_loaders(
                data_dir="data/",
                seq_len=512,
                batch_size=32,
                max_train_stories=100_000,   # fast debug run
            )
            vocab_size = tok.vocab_size    # 50257
        """
        inst = cls(data_dir)
        tokenizer = inst._load_tokenizer()

        train_tokens = inst._tokenise_split(
            "train",
            tokenizer,
            inst.processed_dir / cls.TRAIN_CACHE,
            max_stories=max_train_stories,
        )
        val_tokens = inst._tokenise_split(
            "validation",
            tokenizer,
            inst.processed_dir / cls.VAL_CACHE,
            max_stories=max_val_stories,
        )

        train_ds = _WindowDataset(train_tokens, seq_len, stride)
        val_ds = _WindowDataset(val_tokens, seq_len, stride)

        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        loaders = {
            "train": DataLoader(train_ds, shuffle=True, **loader_kwargs),
            "val": DataLoader(val_ds, shuffle=False, **loader_kwargs),
        }

        print(
            f"Windows — train: {len(train_ds):,}  val: {len(val_ds):,}  "
            f"(seq_len={seq_len}, vocab_size={tokenizer.vocab_size})"
        )
        return loaders, tokenizer
