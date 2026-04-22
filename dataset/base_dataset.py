"""Abstract base class for AttnRes datasets.

All dataset implementations should inherit from :class:`BaseDataset` and
implement :meth:`__len__` and :meth:`__getitem__`.

Provides a standard :meth:`split` helper that returns train / validation
:class:`torch.utils.data.DataLoader` objects from a single dataset instance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class BaseDataset(Dataset, ABC):
    """Abstract base for image classification datasets.

    Subclasses must implement :meth:`__len__` and :meth:`__getitem__`.

    Attributes:
        root: Root directory for raw data storage.
        train: Whether this is the training split.
        transform: Optional torchvision transform pipeline.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Any] = None,
    ) -> None:
        self.root = root
        self.train = train
        self.transform = transform

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples.

        Returns:
            Dataset size.
        """

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Return a single (image, label) pair.

        Args:
            index: Sample index.

        Returns:
            Tuple of ``(image_tensor, class_index)``.
        """

    # ------------------------------------------------------------------
    # Factory helper
    # ------------------------------------------------------------------

    @staticmethod
    def make_loaders(
        train_dataset: Dataset,
        val_split: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        test_dataset: Optional[Dataset] = None,
    ) -> dict[str, DataLoader]:
        """Split ``train_dataset`` into train/val and wrap in DataLoaders.

        Args:
            train_dataset: Full training dataset before the val split.
            val_split: Fraction of ``train_dataset`` to use as validation.
            batch_size: Mini-batch size for all loaders.
            num_workers: DataLoader worker count.
            pin_memory: Whether to use pinned memory.
            seed: Random seed for the split.
            test_dataset: Optional separate test dataset.

        Returns:
            Dict with keys ``"train"``, ``"val"``, and (if provided) ``"test"``,
            each mapping to a :class:`~torch.utils.data.DataLoader`.
        """
        n_val = int(len(train_dataset) * val_split)
        n_train = len(train_dataset) - n_val
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(
            train_dataset, [n_train, n_val], generator=generator
        )

        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        loaders: dict[str, DataLoader] = {
            "train": DataLoader(train_ds, shuffle=True, **loader_kwargs),
            "val": DataLoader(val_ds, shuffle=False, **loader_kwargs),
        }
        if test_dataset is not None:
            loaders["test"] = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        return loaders
