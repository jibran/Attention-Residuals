"""Checkpoint save and restore utilities.

Saves two checkpoints per training run:

* ``checkpoints/latest/latest.pt`` — overwritten every ``save_every`` epochs;
  used to resume interrupted training.
* ``checkpoints/best/best.pt`` — overwritten only when ``val_loss`` improves;
  used for final evaluation and inference.

Each ``.pt`` file is a plain ``torch.save`` dict with keys:

    ``model_state``, ``optimizer_state``, ``scheduler_state``,
    ``epoch``, ``step``, ``val_loss``, ``val_acc``, ``config``

Usage::

    manager = CheckpointManager(checkpoint_dir="checkpoints/", config=cfg)
    manager.save(model, optimizer, scheduler,
                 epoch=3, step=1200, val_loss=0.25, val_acc=0.93)
    start_epoch, step = manager.load(model, optimizer, scheduler,
                                     path="checkpoints/latest/latest.pt")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from utils.config import Config


class CheckpointManager:
    """Manages *best* and *latest* checkpoints for a training run.

    Args:
        checkpoint_dir: Root directory; ``best/`` and ``latest/``
            subdirectories are created automatically.
        config: Experiment config — serialised into every checkpoint for
            full reproducibility.
    """

    def __init__(self, checkpoint_dir: str | Path, config: Config) -> None:
        self._root = Path(checkpoint_dir)
        self._best_dir = self._root / "best"
        self._latest_dir = self._root / "latest"
        self._best_dir.mkdir(parents=True, exist_ok=True)
        self._latest_dir.mkdir(parents=True, exist_ok=True)
        self._config = config
        self._best_val_loss: float = float("inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        epoch: int,
        step: int,
        val_loss: float,
        val_acc: float,
    ) -> dict[str, str]:
        """Save *latest* checkpoint; save *best* checkpoint if val_loss improved.

        Args:
            model: The model whose ``state_dict`` is saved.
            optimizer: Optimiser whose ``state_dict`` is saved.
            scheduler: LR scheduler whose ``state_dict`` is saved (may be ``None``).
            epoch: Completed epoch number (1-based).
            step: Global optimiser step count.
            val_loss: Validation loss for this epoch.
            val_acc: Validation accuracy for this epoch (0–1).

        Returns:
            Dict with keys ``"latest"`` (path always written) and optionally
            ``"best"`` (path written only when val_loss improved).
        """
        payload: dict[str, Any] = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": self._config.to_dict(),
        }

        latest_path = self._latest_dir / "latest.pt"
        torch.save(payload, latest_path)
        saved = {"latest": str(latest_path)}

        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            best_path = self._best_dir / "best.pt"
            torch.save(payload, best_path)
            saved["best"] = str(best_path)

        return saved

    def load(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        path: str | Path | None = None,
        device: torch.device | None = None,
    ) -> tuple[int, int]:
        """Load a checkpoint into *model* (and optionally optimiser/scheduler).

        If ``path`` is ``None``, the *latest* checkpoint is used automatically.

        Args:
            model: Model to load weights into.
            optimizer: If provided, restores optimiser state.
            scheduler: If provided, restores scheduler state.
            path: Explicit ``.pt`` path; defaults to ``latest/latest.pt``.
            device: Device to map tensors onto; defaults to CPU.

        Returns:
            Tuple ``(epoch, step)`` indicating where training left off.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if path is None:
            path = self._latest_dir / "latest.pt"
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        map_loc = device or torch.device("cpu")
        ckpt = torch.load(path, map_location=map_loc, weights_only=False)

        model.load_state_dict(ckpt["model_state"])
        if optimizer and ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])

        # Restore best-loss tracker so saves continue correctly
        self._best_val_loss = ckpt.get("val_loss", float("inf"))

        return ckpt.get("epoch", 0), ckpt.get("step", 0)

    @property
    def best_val_loss(self) -> float:
        """Best validation loss seen so far in this run."""
        return self._best_val_loss
