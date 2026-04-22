"""Concrete dataset wrappers for MNIST and CIFAR-10.

Both classes wrap :mod:`torchvision.datasets` and expose a class-level
:meth:`get_loaders` factory that returns ready-to-use DataLoader dicts.

Usage::

    loaders = MNISTDataset.get_loaders(
        data_dir="data/",
        val_split=0.1,
        batch_size=64,
    )
    for images, labels in loaders["train"]:
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset.base_dataset import BaseDataset


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------


class MNISTDataset(BaseDataset):
    """MNIST handwritten-digit dataset (28×28 greyscale, 10 classes).

    Downloads automatically to ``root/MNIST/`` on first use.

    Args:
        root: Directory to download data into.
        train: If ``True`` load the training split (60 000 samples); otherwise
            load the test split (10 000 samples).
        transform: Optional transform pipeline; defaults to normalised tensor.
    """

    #: Image spatial size (height == width).
    IMG_SIZE: int = 28
    #: Number of input channels.
    IN_CHANNELS: int = 1
    #: Number of output classes.
    NUM_CLASSES: int = 10

    _DEFAULT_TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    _AUGMENT_TRANSFORM = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Any] = None,
        augment: bool = False,
    ) -> None:
        if transform is None:
            transform = (
                self._AUGMENT_TRANSFORM
                if (train and augment)
                else self._DEFAULT_TRANSFORM
            )
        super().__init__(root, train, transform)
        Path(root).mkdir(parents=True, exist_ok=True)
        self._ds = datasets.MNIST(
            root=root, train=train, download=True, transform=transform
        )

    def __len__(self) -> int:
        """Return dataset length.

        Returns:
            Number of samples.
        """
        return len(self._ds)

    def __getitem__(self, index: int):
        """Return a single (image, label) pair.

        Args:
            index: Sample index.

        Returns:
            Tuple ``(image_tensor (1, 28, 28), label)``.
        """
        return self._ds[index]

    @classmethod
    def get_loaders(
        cls,
        data_dir: str = "data/",
        val_split: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        augment: bool = True,
    ) -> dict[str, DataLoader]:
        """Build train / val / test DataLoaders for MNIST.

        Args:
            data_dir: Root data directory.
            val_split: Fraction of training data for validation.
            batch_size: Mini-batch size.
            num_workers: DataLoader workers.
            pin_memory: Pin memory for faster GPU transfer.
            seed: Random seed for the train/val split.
            augment: Apply random rotation augmentation on training split.

        Returns:
            Dict with keys ``"train"``, ``"val"``, ``"test"``.
        """
        train_ds = cls(data_dir, train=True, augment=augment)
        test_ds = cls(data_dir, train=False, augment=False)
        return BaseDataset.make_loaders(
            train_dataset=train_ds,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
            test_dataset=test_ds,
        )


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------


class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 image dataset (32×32 RGB, 10 classes).

    Downloads automatically to ``root/cifar-10-batches-py/`` on first use.

    Args:
        root: Directory to download data into.
        train: If ``True`` load the training split (50 000 samples); otherwise
            load the test split (10 000 samples).
        transform: Optional transform pipeline; defaults to normalised tensor.
        augment: Apply random-crop + horizontal-flip augmentation when training.
    """

    IMG_SIZE: int = 32
    IN_CHANNELS: int = 3
    NUM_CLASSES: int = 10

    _MEAN = (0.4914, 0.4822, 0.4465)
    _STD = (0.2470, 0.2435, 0.2616)

    _DEFAULT_TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )

    _AUGMENT_TRANSFORM = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Any] = None,
        augment: bool = False,
    ) -> None:
        if transform is None:
            transform = (
                self._AUGMENT_TRANSFORM
                if (train and augment)
                else self._DEFAULT_TRANSFORM
            )
        super().__init__(root, train, transform)
        Path(root).mkdir(parents=True, exist_ok=True)
        self._ds = datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )

    def __len__(self) -> int:
        """Return dataset length.

        Returns:
            Number of samples.
        """
        return len(self._ds)

    def __getitem__(self, index: int):
        """Return a single (image, label) pair.

        Args:
            index: Sample index.

        Returns:
            Tuple ``(image_tensor (3, 32, 32), label)``.
        """
        return self._ds[index]

    @classmethod
    def get_loaders(
        cls,
        data_dir: str = "data/",
        val_split: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        augment: bool = True,
    ) -> dict[str, DataLoader]:
        """Build train / val / test DataLoaders for CIFAR-10.

        Args:
            data_dir: Root data directory.
            val_split: Fraction of training data for validation.
            batch_size: Mini-batch size.
            num_workers: DataLoader workers.
            pin_memory: Pin memory for faster GPU transfer.
            seed: Random seed for the train/val split.
            augment: Apply crop + flip augmentation on training split.

        Returns:
            Dict with keys ``"train"``, ``"val"``, ``"test"``.
        """
        train_ds = cls(data_dir, train=True, augment=augment)
        test_ds = cls(data_dir, train=False, augment=False)
        return BaseDataset.make_loaders(
            train_dataset=train_ds,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
            test_dataset=test_ds,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, type] = {
    "mnist": MNISTDataset,
    "cifar10": CIFAR10Dataset,
}


def get_dataset_class(name: str) -> type:
    """Look up a dataset class by name.

    Args:
        name: Dataset identifier — ``"mnist"`` or ``"cifar10"``.

    Returns:
        The corresponding dataset class.

    Raises:
        ValueError: If ``name`` is not in the registry.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. " f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name]
