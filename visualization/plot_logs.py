"""Plot training curves from a single CSV log file.

Generates two subplots:
  * Loss  — training loss per step + per-epoch val loss.
  * Accuracy — per-epoch train and val accuracy.

Usage::

    python visualization/plot_logs.py --log logs/AttnResTransformer-202406011200.csv
    python visualization/plot_logs.py --log logs/run.csv --out reports/curves.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_log(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a training CSV and split into step-level and epoch-level rows.

    Args:
        path: Path to the CSV log file.

    Returns:
        Tuple ``(steps_df, epochs_df)`` where *steps_df* contains rows with
        ``phase == "train"`` and *epochs_df* contains rows with
        ``phase == "epoch"``.
    """
    df = pd.read_csv(path)
    steps = df[df["phase"] == "train"].copy()
    epochs = df[df["phase"] == "epoch"].copy()
    return steps, epochs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_run(log_path: Path, out_path: Path | None = None) -> None:
    """Plot loss and accuracy curves for one training run.

    Args:
        log_path: Path to the CSV log produced by :class:`~utils.logger.TrainingLogger`.
        out_path: If provided, save the figure here (PNG/PDF); otherwise display
            interactively.
    """
    steps, epochs = load_log(log_path)
    model_name = log_path.stem

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Training curves — {model_name}", fontsize=13, fontweight="bold")

    # ------------------------------------------------------------------ Loss
    ax = axes[0]
    if not steps.empty:
        ax.plot(
            steps["step"],
            steps["train_loss"],
            alpha=0.4,
            color="#6366f1",
            linewidth=0.8,
            label="train (step)",
        )
    if not epochs.empty:
        ax.plot(
            epochs["step"],
            epochs["train_loss"].astype(float),
            color="#6366f1",
            linewidth=2,
            marker="o",
            markersize=4,
            label="train (epoch)",
        )
        ax.plot(
            epochs["step"],
            epochs["val_loss"].astype(float),
            color="#f59e0b",
            linewidth=2,
            marker="s",
            markersize=4,
            label="val (epoch)",
        )
    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    # ------------------------------------------------------------ Accuracy
    ax = axes[1]
    if not epochs.empty:
        train_acc = epochs["train_acc"].astype(float) * 100
        val_acc = epochs["val_acc"].astype(float) * 100
        ax.plot(
            epochs["epoch"].astype(int),
            train_acc,
            color="#6366f1",
            linewidth=2,
            marker="o",
            markersize=4,
            label="train",
        )
        ax.plot(
            epochs["epoch"].astype(int),
            val_acc,
            color="#10b981",
            linewidth=2,
            marker="s",
            markersize=4,
            label="val",
        )
        ax.set_ylim(0, 100)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI args and produce the plot."""
    parser = argparse.ArgumentParser(description="Plot AttnRes training curves.")
    parser.add_argument("--log", required=True, help="Path to CSV log file.")
    parser.add_argument(
        "--out", default=None, help="Output path (PNG/PDF). Shows GUI if omitted."
    )
    args = parser.parse_args()

    plot_run(Path(args.log), Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
