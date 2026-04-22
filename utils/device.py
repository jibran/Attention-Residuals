"""Device resolution and seeding utilities.

Usage::

    device = resolve_device("auto")   # picks cuda > mps > cpu
    seed_everything(42)
"""

from __future__ import annotations

import random

import numpy as np
import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Return the best available :class:`torch.device`.

    Args:
        preference: ``"auto"`` picks CUDA > MPS > CPU automatically.
            Pass ``"cpu"``, ``"cuda"``, or ``"mps"`` to force a specific backend.

    Returns:
        A :class:`torch.device` instance.

    Raises:
        ValueError: If an explicit device is requested but unavailable.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    dev = torch.device(preference)
    if preference == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available.")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but not available.")
    return dev


def seed_everything(seed: int) -> None:
    """Fix all random seeds for reproducibility.

    Sets seeds for :mod:`random`, :mod:`numpy`, and :mod:`torch` (CPU + all
    CUDA devices). Also enables deterministic CUDA algorithms.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
