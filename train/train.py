"""Main training script for AttnRes experiments.

Trains either :class:`~models.AttnResTransformer` or
:class:`~models.BaselineTransformer` on MNIST or CIFAR-10, logging metrics
to a timestamped CSV and saving checkpoints to ``checkpoints/best/`` and
``checkpoints/latest/``.

Usage::

    # From the project root:
    python train/train.py --config config/base.yaml

    # Override individual hyperparameters:
    python train/train.py --config config/base.yaml \\
        --override training.lr=1e-3 model.depth=12

    # Resume from the latest checkpoint:
    python train/train.py --config config/base.yaml --resume

    # Train the baseline (standard residuals) for comparison:
    python train/train.py --config config/base.yaml --baseline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.image_datasets import get_dataset_class
from models import build_model
from utils import (
    CheckpointManager,
    TrainingLogger,
    load_config,
    resolve_device,
    seed_everything,
)
from utils.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_dataset_meta(dataset_name: str) -> dict:
    """Return dataset-specific constants (img_size, channels, classes).

    Args:
        dataset_name: ``"mnist"`` or ``"cifar10"``.

    Returns:
        Dict with keys ``img_size``, ``in_channels``, ``num_classes``.
    """
    meta = {
        "mnist": dict(img_size=28, in_channels=1, num_classes=10),
        "cifar10": dict(img_size=32, in_channels=3, num_classes=10),
    }
    if dataset_name not in meta:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")
    return meta[dataset_name]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run inference over a DataLoader and return loss and accuracy.

    Args:
        model: Model in eval mode.
        loader: DataLoader to iterate over.
        criterion: Loss function (e.g. CrossEntropyLoss).
        device: Device to run on.

    Returns:
        Tuple ``(mean_loss, accuracy)`` where accuracy is in ``[0, 1]``.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += images.size(0)
    model.train()
    return total_loss / total, correct / total


def build_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> SequentialLR:
    """Build a linear warm-up → cosine-decay LR schedule.

    Args:
        optimizer: Optimiser whose LR is scheduled.
        warmup_steps: Number of steps for the linear warm-up phase.
        total_steps: Total training steps (warm-up + cosine decay).

    Returns:
        A :class:`~torch.optim.lr_scheduler.SequentialLR` instance.
    """
    warmup = LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: Config, baseline: bool = False, resume: bool = False) -> None:
    """Run the full training loop.

    Args:
        cfg: Experiment configuration.
        baseline: If ``True`` train the standard-residual baseline model.
        resume: If ``True`` resume from ``checkpoints/latest/latest.pt``.
    """
    seed_everything(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    print(f"Device: {device}")

    # ------------------------------------------------------------------ Data
    ds_meta = _get_dataset_meta(cfg.data.dataset)
    ds_cls = get_dataset_class(cfg.data.dataset)
    loaders = ds_cls.get_loaders(
        data_dir=cfg.data.data_dir,
        val_split=cfg.data.val_split,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type != "cpu",
        seed=cfg.training.seed,
        augment=True,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # ----------------------------------------------------------------- Model
    model = build_model(
        cfg=cfg.model,
        num_classes=ds_meta["num_classes"],
        img_size=ds_meta["img_size"],
        patch_size=4,
        in_channels=ds_meta["in_channels"],
        baseline=baseline,
    ).to(device)

    model_name = "Baseline" if baseline else cfg.model.name
    print(f"Model: {model_name}  |  params: {model.num_parameters:,}")

    # ------------------------------------------------- Optimiser + scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    total_steps = len(train_loader) * cfg.training.epochs
    scheduler = build_scheduler(optimizer, cfg.training.warmup_steps, total_steps)

    # ----------------------------------------------- Logger + checkpoint mgr
    logger = TrainingLogger(log_dir=cfg.logging.log_dir, model_name=model_name)
    ckpt_mgr = CheckpointManager(checkpoint_dir=cfg.logging.checkpoint_dir, config=cfg)

    criterion = nn.CrossEntropyLoss()
    start_epoch, global_step = 0, 0

    if resume:
        try:
            start_epoch, global_step = ckpt_mgr.load(
                model, optimizer, scheduler, device=device
            )
            print(f"Resumed from epoch {start_epoch}, step {global_step}.")
        except FileNotFoundError:
            print("No checkpoint found — starting from scratch.")

    # ----------------------------------------------------------------- Train
    model.train()
    for epoch in range(start_epoch + 1, cfg.training.epochs + 1):
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}", leave=False
        )
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            if cfg.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)

            optimizer.step()
            scheduler.step()
            global_step += 1

            batch_acc = (logits.argmax(-1) == labels).float().mean().item()
            epoch_loss += loss.item() * images.size(0)
            epoch_correct += (logits.argmax(-1) == labels).sum().item()
            epoch_total += images.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc*100:.1f}%")

            if global_step % cfg.training.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                logger.log_step(epoch, global_step, loss.item(), batch_acc, lr)

        # --------------------------------------------------------- Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        lr_now = scheduler.get_last_lr()[0]

        logger.log_epoch(
            epoch, global_step, train_loss, train_acc, val_loss, val_acc, lr_now
        )

        # ------------------------------------------------------ Checkpointing
        if epoch % cfg.training.save_every == 0:
            saved = ckpt_mgr.save(
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                step=global_step,
                val_loss=val_loss,
                val_acc=val_acc,
            )
            tags = []
            if "best" in saved:
                tags.append("★ new best")
            if "latest" in saved:
                tags.append("saved latest")
            if tags:
                print(f"  Checkpoint: {', '.join(tags)}")

    logger.close()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and launch training."""
    parser = argparse.ArgumentParser(description="Train an AttnRes model.")
    parser.add_argument(
        "--config", default="config/base.yaml", help="Path to YAML config."
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Config overrides, e.g. training.lr=1e-3",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Train the standard-residual baseline instead.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints/latest/latest.pt.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    train(cfg, baseline=args.baseline, resume=args.resume)


if __name__ == "__main__":
    main()
