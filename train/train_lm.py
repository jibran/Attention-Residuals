"""Language-model training script for Tiny-Shakespeare.

Trains :class:`~models.AttnResLM` (or :class:`~models.BaselineLM`) on the
``Trelis/tiny-shakespeare`` dataset using character-level next-token
prediction.  Logs perplexity + loss to a timestamped CSV and saves
checkpoints to ``checkpoints/best/`` and ``checkpoints/latest/``.

After each epoch a short sample is generated from a fixed seed prompt so
you can qualitatively track how the model learns to write Shakespeare.

Usage::

    # Block AttnRes (default):
    python train/train_lm.py --config config/shakespeare.yaml

    # Full AttnRes:
    python train/train_lm.py --config config/shakespeare.yaml \\
        --override model.use_block_attn_res=false

    # Standard-residual baseline:
    python train/train_lm.py --config config/shakespeare.yaml --baseline

    # Resume from latest checkpoint:
    python train/train_lm.py --config config/shakespeare.yaml --resume

    # Custom prompt for generation preview:
    python train/train_lm.py --config config/shakespeare.yaml \\
        --prompt "ROMEO: " --max_new_tokens 300
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.shakespeare_dataset import ShakespeareDataset
from dataset.tokenizer import CharTokenizer
from models import build_lm
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


def _build_scheduler(
    optimizer: AdamW, warmup_steps: int, total_steps: int
) -> SequentialLR:
    """Build a linear warm-up → cosine-decay learning rate schedule.

    Args:
        optimizer: Optimiser to schedule.
        warmup_steps: Linear warm-up duration in steps.
        total_steps: Total training steps.

    Returns:
        :class:`~torch.optim.lr_scheduler.SequentialLR` instance.
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


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Compute mean cross-entropy loss and perplexity over a DataLoader.

    Args:
        model: Model in eval mode.
        loader: DataLoader yielding ``(x, y)`` token-id pairs.
        device: Inference device.

    Returns:
        Tuple ``(mean_loss, perplexity)``.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
    model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    return mean_loss, math.exp(mean_loss)


def _generate_sample(
    model: nn.Module,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    """Generate a text sample from a prompt string.

    Args:
        model: Trained language model.
        tokenizer: Fitted :class:`~dataset.tokenizer.CharTokenizer`.
        prompt: Seed text string.
        max_new_tokens: Number of new characters to generate.
        temperature: Sampling temperature.
        top_k: Top-k logit filter (0 = disabled).
        device: Inference device.

    Returns:
        Generated text string (prompt + continuation).
    """
    model.eval()
    ids = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device=device
    ).unsqueeze(0)
    out_ids = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k if top_k > 0 else None,
    )
    model.train()
    return tokenizer.decode(out_ids[0].tolist())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_lm(
    cfg: Config,
    baseline: bool = False,
    resume: bool = False,
    prompt: str = "ROMEO: ",
    max_new_tokens: int | None = None,
) -> None:
    """Run the full language-model training loop on Tiny-Shakespeare.

    Args:
        cfg: Experiment configuration.
        baseline: If ``True``, train the standard-residual :class:`~models.BaselineLM`.
        resume: If ``True``, resume from ``checkpoints/latest/latest.pt``.
        prompt: Seed string for per-epoch generation previews.
        max_new_tokens: Override generation length (uses config value if ``None``).
    """
    seed_everything(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    print(f"Device: {device}")

    # ------------------------------------------------------------------ Data
    loaders, tokenizer = ShakespeareDataset.get_loaders(
        data_dir=cfg.data.data_dir,
        seq_len=cfg.data.seq_len,
        stride=cfg.data.stride,
        val_split=cfg.data.val_split,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type != "cpu",
        seed=cfg.training.seed,
    )

    # ----------------------------------------------------------------- Model
    model = build_lm(
        cfg=cfg.model,
        vocab_size=tokenizer.vocab_size,
        seq_len=cfg.data.seq_len,
        baseline=baseline,
    ).to(device)

    model_name = "BaselineLM" if baseline else cfg.model.name
    print(
        f"Model: {model_name}  |  params: {model.num_parameters:,}  |  vocab: {tokenizer.vocab_size}"
    )

    # ------------------------------------------------- Optimiser + scheduler
    optimizer = AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    total_steps = len(loaders["train"]) * cfg.training.epochs
    scheduler = _build_scheduler(optimizer, cfg.training.warmup_steps, total_steps)

    # ------------------------------------------ Logger + checkpoint manager
    logger = TrainingLogger(log_dir=cfg.logging.log_dir, model_name=model_name)
    ckpt_mgr = CheckpointManager(checkpoint_dir=cfg.logging.checkpoint_dir, config=cfg)

    start_epoch, global_step = 0, 0
    gen_tokens = max_new_tokens or cfg.generation.max_new_tokens

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
        epoch_loss, epoch_tokens = 0.0, 0

        pbar = tqdm(
            loaders["train"], desc=f"Epoch {epoch}/{cfg.training.epochs}", leave=False
        )
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()

            if cfg.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)

            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item() * x.numel()
            epoch_tokens += x.numel()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ppl=f"{math.exp(min(loss.item(), 20)):.1f}",
            )

            if global_step % cfg.training.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                # Log step-level metrics; reuse train_acc field for perplexity (inverted)
                logger.log_step(
                    epoch=epoch,
                    step=global_step,
                    train_loss=loss.item(),
                    train_acc=1.0 / max(math.exp(min(loss.item(), 20)), 1),
                    lr=lr,
                )

        # -------------------------------------------------------------- Val
        val_loss, val_ppl = _evaluate(model, loaders["val"], device)
        train_loss = epoch_loss / max(epoch_tokens, 1)
        train_ppl = math.exp(min(train_loss, 20))
        lr_now = scheduler.get_last_lr()[0]

        print(
            f"\nEpoch {epoch} | "
            f"train loss {train_loss:.4f}  ppl {train_ppl:.1f} | "
            f"val loss {val_loss:.4f}  ppl {val_ppl:.1f} | "
            f"lr {lr_now:.2e}"
        )

        logger.log_epoch(
            epoch=epoch,
            step=global_step,
            train_loss=train_loss,
            train_acc=1.0 / max(train_ppl, 1),
            val_loss=val_loss,
            val_acc=1.0 / max(val_ppl, 1),
            lr=lr_now,
        )

        # -------------------------------------------- Checkpoint
        if epoch % cfg.training.save_every == 0:
            saved = ckpt_mgr.save(
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                step=global_step,
                val_loss=val_loss,
                val_acc=1.0 / max(val_ppl, 1),
            )
            if "best" in saved:
                print("  ★ New best checkpoint saved.")

        # -------------------------------------- Generation preview
        sample = _generate_sample(
            model,
            tokenizer,
            prompt,
            max_new_tokens=gen_tokens,
            temperature=cfg.generation.temperature,
            top_k=cfg.generation.top_k,
            device=device,
        )
        print(f"\n{'─'*60}")
        print(sample[:600])
        print(f"{'─'*60}\n")

    logger.close()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and launch LM training."""
    parser = argparse.ArgumentParser(description="Train AttnResLM on Tiny-Shakespeare.")
    parser.add_argument(
        "--config", default="config/shakespeare.yaml", help="YAML config path."
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
        help="Train the standard-residual BaselineLM.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints/latest/latest.pt.",
    )
    parser.add_argument(
        "--prompt",
        default="ROMEO: ",
        help="Seed text for per-epoch generation preview.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Override number of generated tokens per preview.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    train_lm(
        cfg,
        baseline=args.baseline,
        resume=args.resume,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
