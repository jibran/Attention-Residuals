"""Compare training curves across multiple CSV log files.

Overlays val loss and val accuracy for every CSV found in the specified
log directory (or a list of explicit files), making it easy to see which
variant (Full AttnRes, Block AttnRes, Baseline) performs best.

Usage::

    # Auto-discover all CSVs in the logs/ directory:
    python visualization/compare_models.py --logs logs/

    # Select specific files:
    python visualization/compare_models.py \\
        --logs logs/AttnResTransformer-202406011200.csv \\
               logs/Baseline-202406011300.csv \\
        --out reports/comparison.png

    # Also plot training speed (steps/second) from elapsed_s:
    python visualization/compare_models.py --logs logs/ --speed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COLORS = [
    "#6366f1",
    "#f59e0b",
    "#10b981",
    "#ef4444",
    "#8b5cf6",
    "#06b6d4",
    "#f97316",
    "#84cc16",
]


def _collect_logs(sources: list[str]) -> list[Path]:
    """Gather CSV paths from a mix of files and directories.

    Args:
        sources: List of file or directory paths.

    Returns:
        Sorted list of ``.csv`` :class:`~pathlib.Path` objects.
    """
    paths: list[Path] = []
    for s in sources:
        p = Path(s)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.csv")))
        elif p.suffix == ".csv":
            paths.append(p)
    return paths


def _load_epochs(path: Path) -> pd.DataFrame:
    """Load epoch-level rows from a log CSV.

    Args:
        path: CSV log file path.

    Returns:
        DataFrame filtered to rows where ``phase == "epoch"``.
    """
    df = pd.read_csv(path)
    return df[df["phase"] == "epoch"].copy()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def compare_models(
    log_sources: list[str],
    out_path: Path | None = None,
    show_speed: bool = False,
) -> None:
    """Overlay validation loss and accuracy (and optionally speed) for all runs.

    Args:
        log_sources: List of CSV paths or directories containing CSVs.
        out_path: If provided, save the figure here; otherwise show GUI.
        show_speed: If ``True``, add a third subplot showing throughput
            (steps / elapsed second) over epochs.
    """
    paths = _collect_logs(log_sources)
    if not paths:
        print("No CSV files found.")
        return

    n_plots = 3 if show_speed else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    fig.suptitle("Model comparison", fontsize=13, fontweight="bold")

    for idx, path in enumerate(paths):
        epochs = _load_epochs(path)
        if epochs.empty:
            continue

        color = _COLORS[idx % len(_COLORS)]
        label = path.stem

        x = epochs["epoch"].astype(int)
        vl = epochs["val_loss"].astype(float)
        va = epochs["val_acc"].astype(float) * 100

        # Val loss
        axes[0].plot(
            x, vl, color=color, linewidth=2, marker="o", markersize=4, label=label
        )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Val loss")
        axes[0].set_title("Validation loss")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        # Val accuracy
        axes[1].plot(
            x, va, color=color, linewidth=2, marker="s", markersize=4, label=label
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Val accuracy (%)")
        axes[1].set_title("Validation accuracy")
        axes[1].set_ylim(0, 100)
        axes[1].grid(True, linestyle="--", alpha=0.4)

        # Speed (steps / second)
        if show_speed and "step" in epochs.columns and "elapsed_s" in epochs.columns:
            steps = epochs["step"].astype(float)
            elapsed = epochs["elapsed_s"].astype(float)
            speed = steps / elapsed.clip(lower=1)
            axes[2].plot(
                x,
                speed,
                color=color,
                linewidth=2,
                marker="^",
                markersize=4,
                label=label,
            )
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Steps / second")
            axes[2].set_title("Training speed")
            axes[2].grid(True, linestyle="--", alpha=0.4)

    for ax in axes:
        ax.legend(fontsize=9)

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
    """Parse CLI args and produce the comparison plot."""
    parser = argparse.ArgumentParser(
        description="Compare multiple AttnRes training runs."
    )
    parser.add_argument(
        "--logs", nargs="+", required=True, help="CSV files or directories to compare."
    )
    parser.add_argument("--out", default=None, help="Output path (PNG/PDF).")
    parser.add_argument(
        "--speed", action="store_true", help="Add a training-speed subplot."
    )
    args = parser.parse_args()

    compare_models(
        args.logs, Path(args.out) if args.out else None, show_speed=args.speed
    )


if __name__ == "__main__":
    main()
