"""Character-level tokeniser for the Tiny-Shakespeare dataset.

Builds a vocabulary from the raw text corpus and maps every character to a
unique integer id.  The tokeniser can be saved to / loaded from a JSON file
so that inference always uses the same mapping as training.

Design rationale
----------------
The paper trains on character-level sequences (no sub-word BPE) to keep the
vocabulary small and the task tractable on a single GPU.  A vocab of ~65
unique ASCII characters is typical for the Shakespeare corpus.

Usage::

    tok = CharTokenizer.from_text("Hello, World!")
    ids = tok.encode("Hello")          # [18, 29, 38, 38, 41]
    text = tok.decode(ids)             # "Hello"
    tok.save("data/processed/vocab.json")

    tok2 = CharTokenizer.load("data/processed/vocab.json")
"""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """Character-level tokeniser.

    Attributes:
        vocab_size: Number of unique characters in the vocabulary.
        pad_id: Integer id of the padding token ``"<pad>"``.
        unk_id: Integer id of the unknown token ``"<unk>"``.

    Args:
        chars: Sorted list of unique characters to include in the vocabulary.
            Two special tokens — ``"<pad>"`` (id 0) and ``"<unk>"`` (id 1) —
            are prepended automatically.
    """

    PAD_TOKEN: str = "<pad>"
    UNK_TOKEN: str = "<unk>"

    def __init__(self, chars: list[str]) -> None:
        special = [self.PAD_TOKEN, self.UNK_TOKEN]
        vocab = special + chars
        self._stoi: dict[str, int] = {ch: i for i, ch in enumerate(vocab)}
        self._itos: dict[int, str] = {i: ch for i, ch in enumerate(vocab)}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary (including specials).

        Returns:
            Vocabulary size.
        """
        return len(self._stoi)

    @property
    def pad_id(self) -> int:
        """Integer id of the padding token.

        Returns:
            Padding token id (always 0).
        """
        return self._stoi[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        """Integer id of the unknown token.

        Returns:
            Unknown token id (always 1).
        """
        return self._stoi[self.UNK_TOKEN]

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of integer token ids.

        Unknown characters are mapped to :attr:`unk_id`.

        Args:
            text: Input string.

        Returns:
            List of integer ids, one per character.
        """
        return [self._stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Convert a list of integer token ids back to a string.

        Padding tokens are omitted.  Unknown ids are replaced with ``"?"``.

        Args:
            ids: List of integer token ids.

        Returns:
            Decoded string.
        """
        return "".join(self._itos.get(i, "?") for i in ids if i != self.pad_id)

    def __len__(self) -> int:
        """Return the vocabulary size.

        Returns:
            Number of tokens in the vocabulary.
        """
        return self.vocab_size

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the tokeniser vocabulary to a JSON file.

        Args:
            path: Destination file path (e.g. ``"data/processed/vocab.json"``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stoi": self._stoi,
            "itos": {str(k): v for k, v in self._itos.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Deserialise a tokeniser from a JSON file created by :meth:`save`.

        Args:
            path: Path to the JSON vocabulary file.

        Returns:
            Reconstructed :class:`CharTokenizer` instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
        payload = json.loads(path.read_text())
        obj = cls.__new__(cls)
        obj._stoi = payload["stoi"]
        obj._itos = {int(k): v for k, v in payload["itos"].items()}
        return obj

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a vocabulary from a raw text string.

        Args:
            text: Full corpus text; every unique character becomes a token.

        Returns:
            A :class:`CharTokenizer` with a vocabulary derived from ``text``.
        """
        chars = sorted(set(text))
        # Remove the special-token strings from chars if accidentally present
        chars = [c for c in chars if c not in (cls.PAD_TOKEN, cls.UNK_TOKEN)]
        return cls(chars)
