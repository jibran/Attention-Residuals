"""Tiny-Shakespeare dataset for character-level language modelling.

Downloads ``Trelis/tiny-shakespeare`` from the Hugging Face Hub, concatenates
all passages into a single corpus string, builds a :class:`~dataset.tokenizer.CharTokenizer`,
and slices the token sequence into overlapping fixed-length windows that serve
as (input, target) pairs for next-character prediction.

Dataset statistics
------------------
* HF splits: ``train`` (472 rows) and ``test`` (49 rows)
* Typical corpus length after concatenation: ~1 M characters
* Default ``seq_len`` = 256 characters per training window

Usage::

    loaders = ShakespeareDataset.get_loaders(
        data_dir="data/",
        seq_len=256,
        batch_size=64,
        val_split=0.1,
    )
    for x, y in loaders["train"]:
        # x: (B, seq_len)  — input token ids
        # y: (B, seq_len)  — target token ids (x shifted right by 1)
        ...
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from dataset.tokenizer import CharTokenizer

# ---------------------------------------------------------------------------
# Window dataset
# ---------------------------------------------------------------------------


class _WindowDataset(Dataset):
    """Fixed-length sliding-window view over a 1-D token tensor.

    Each ``__getitem__`` returns ``(x, y)`` where ``y = x`` shifted left by
    one position — the standard next-token prediction setup.

    Args:
        tokens: 1-D :class:`torch.Tensor` of token ids (dtype ``torch.long``).
        seq_len: Window length (number of tokens per sample).
        stride: Step size between consecutive windows.  Defaults to
            ``seq_len`` (non-overlapping).  Use a smaller value for more
            training examples at the cost of data redundancy.
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
        # Number of complete windows that fit in the token stream
        self._n = max(0, (len(tokens) - seq_len - 1) // self._stride + 1)

    def __len__(self) -> int:
        """Return the number of windows.

        Returns:
            Window count.
        """
        return self._n

    def __getitem__(self, idx: int):
        """Return the ``idx``-th (input, target) window pair.

        Args:
            idx: Window index.

        Returns:
            Tuple ``(x, y)`` where both tensors have shape ``(seq_len,)`` and
            dtype ``torch.long``.
        """
        start = idx * self._stride
        x = self._tokens[start : start + self._seq_len]
        y = self._tokens[start + 1 : start + self._seq_len + 1]
        return x, y


# ---------------------------------------------------------------------------
# Public dataset class
# ---------------------------------------------------------------------------


class ShakespeareDataset:
    """Factory and namespace for Tiny-Shakespeare data loading.

    This class is not instantiated directly.  Use :meth:`get_loaders` to
    obtain ready-to-use :class:`~torch.utils.data.DataLoader` objects and
    a fitted :class:`~dataset.tokenizer.CharTokenizer`.

    Args:
        data_dir: Root directory for caching downloaded data and the
            processed vocabulary file.
    """

    HF_REPO: str = "Trelis/tiny-shakespeare"
    VOCAB_FILE: str = "vocab.json"
    CORPUS_FILE: str = "shakespeare_corpus.txt"

    def __init__(self, data_dir: str = "data/") -> None:
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_and_build_corpus(self) -> str:
        """Download the HF dataset and concatenate all text passages.

        Caches the concatenated corpus as a plain text file in
        ``data/processed/`` so subsequent calls are instant.

        Returns:
            The full corpus as a single string.
        """
        processed_dir = self.data_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = processed_dir / self.CORPUS_FILE

        if corpus_path.exists():
            return corpus_path.read_text(encoding="utf-8")

        from datasets import load_dataset  # lazy import

        print(f"Downloading {self.HF_REPO} from Hugging Face Hub …")
        ds = load_dataset(self.HF_REPO, trust_remote_code=True)

        # Collect all passages from all available splits
        passages: list[str] = []
        for split_name in ds:
            for row in ds[split_name]:
                text = row.get("Text") or row.get("text") or ""
                if text.strip():
                    passages.append(text.strip())

        corpus = "\n\n".join(passages)
        corpus_path.write_text(corpus, encoding="utf-8")
        print(f"Corpus saved → {corpus_path}  ({len(corpus):,} characters)")
        return corpus

    def _build_or_load_tokenizer(self, corpus: str) -> CharTokenizer:
        """Build a :class:`~dataset.tokenizer.CharTokenizer` from the corpus.

        Saves the vocabulary to ``data/processed/vocab.json`` on first build;
        loads from disk on subsequent runs.

        Args:
            corpus: Full corpus string used to derive the vocabulary.

        Returns:
            A fitted :class:`~dataset.tokenizer.CharTokenizer`.
        """
        vocab_path = self.data_dir / "processed" / self.VOCAB_FILE
        if vocab_path.exists():
            return CharTokenizer.load(vocab_path)
        tok = CharTokenizer.from_text(corpus)
        tok.save(vocab_path)
        print(f"Vocabulary saved → {vocab_path}  (vocab_size={tok.vocab_size})")
        return tok

    # ------------------------------------------------------------------
    # Public factory
    # ------------------------------------------------------------------

    @classmethod
    def get_loaders(
        cls,
        data_dir: str = "data/",
        seq_len: int = 256,
        stride: int | None = None,
        val_split: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
    ) -> tuple[dict[str, DataLoader], CharTokenizer]:
        """Download, tokenise, and split the Shakespeare corpus into DataLoaders.

        Args:
            data_dir: Root directory for raw/processed data and vocabulary cache.
            seq_len: Number of tokens per training window.
            stride: Step between consecutive windows (defaults to ``seq_len``
                for non-overlapping windows; use ``seq_len // 2`` for 50%
                overlap and roughly 2× more samples).
            val_split: Fraction of the training corpus windows used for
                validation.
            batch_size: Mini-batch size for all loaders.
            num_workers: DataLoader worker processes.
            pin_memory: Use pinned memory for faster GPU data transfer.
            seed: Random seed for the train/val split.

        Returns:
            Tuple ``(loaders, tokenizer)`` where ``loaders`` is a dict with
            keys ``"train"``, ``"val"``, and ``"test"``, and ``tokenizer`` is
            the fitted :class:`~dataset.tokenizer.CharTokenizer`.

        Example::

            loaders, tok = ShakespeareDataset.get_loaders(
                data_dir="data/", seq_len=256, batch_size=64
            )
            vocab_size = tok.vocab_size   # ~67
        """
        inst = cls(data_dir)
        corpus = inst._download_and_build_corpus()
        tok = inst._build_or_load_tokenizer(corpus)

        # Tokenise the entire corpus once
        all_ids = torch.tensor(tok.encode(corpus), dtype=torch.long)

        # Split the raw token stream: first 90% train+val, last 10% test
        n_test = max(seq_len + 1, int(len(all_ids) * 0.10))
        train_ids = all_ids[: len(all_ids) - n_test]
        test_ids = all_ids[len(all_ids) - n_test :]

        # Build window datasets
        train_full = _WindowDataset(train_ids, seq_len, stride)
        test_ds = _WindowDataset(test_ids, seq_len, stride)

        # Split train_full into train and val
        n_val = max(1, int(len(train_full) * val_split))
        n_train = len(train_full) - n_val
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(
            train_full, [n_train, n_val], generator=generator
        )

        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        loaders: dict[str, DataLoader] = {
            "train": DataLoader(train_ds, shuffle=True, **loader_kwargs),
            "val": DataLoader(val_ds, shuffle=False, **loader_kwargs),
            "test": DataLoader(test_ds, shuffle=False, **loader_kwargs),
        }

        print(
            f"Windows — train: {len(train_ds):,}  "
            f"val: {len(val_ds):,}  "
            f"test: {len(test_ds):,}  "
            f"(seq_len={seq_len}, vocab_size={tok.vocab_size})"
        )
        return loaders, tok
