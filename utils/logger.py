"""Training logger: writes CSV logs and prints rich console output.

Each training run creates one CSV file at::

    logs/<model_name>-<YYYYMMDDHHMM>.csv

Columns
-------
epoch, step, train_loss, train_acc, val_loss, val_acc, lr, elapsed_s

Usage::

    logger = TrainingLogger(log_dir="logs/", model_name="AttnResTransformer")
    logger.log_step(epoch=1, step=50, train_loss=0.42, train_acc=0.87, lr=3e-4)
    logger.log_epoch(epoch=1, val_loss=0.35, val_acc=0.91)
    logger.close()
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

try:
    from rich import box as rich_box
    from rich.console import Console
    from rich.table import Table

    _RICH = True
except ImportError:
    _RICH = False


class TrainingLogger:
    """Logs training metrics to a timestamped CSV and prints to console.

    Attributes:
        log_path: Absolute path to the CSV file for this run.

    Args:
        log_dir: Directory where CSV files are written.
        model_name: Used as the prefix of the CSV filename.
    """

    _CSV_FIELDS = [
        "epoch",
        "step",
        "phase",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "lr",
        "elapsed_s",
    ]

    def __init__(self, log_dir: str | Path, model_name: str) -> None:
        self._start = time.time()
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d%H%M")
        self.log_path = log_dir / f"{model_name}-{stamp}.csv"

        self._fh = self.log_path.open("w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._CSV_FIELDS)
        self._writer.writeheader()
        self._fh.flush()

        self._console = Console() if _RICH else None
        self._print(f"[bold]Log file:[/bold] {self.log_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_step(
        self,
        epoch: int,
        step: int,
        train_loss: float,
        train_acc: float,
        lr: float,
    ) -> None:
        """Write a training-step row to CSV and print a compact console line.

        Args:
            epoch: Current epoch index (1-based).
            step: Global optimiser step count.
            train_loss: Loss on the current mini-batch.
            train_acc: Accuracy on the current mini-batch (0–1).
            lr: Current learning rate.
        """
        row = {
            "epoch": epoch,
            "step": step,
            "phase": "train",
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": "",
            "val_acc": "",
            "lr": f"{lr:.2e}",
            "elapsed_s": round(time.time() - self._start, 1),
        }
        self._writer.writerow(row)
        self._fh.flush()

        msg = (
            f"[dim]epoch[/dim] {epoch:>3}  "
            f"[dim]step[/dim] {step:>6}  "
            f"loss [yellow]{train_loss:.4f}[/yellow]  "
            f"acc [cyan]{train_acc*100:.2f}%[/cyan]  "
            f"lr {lr:.2e}"
        )
        self._print(msg)

    def log_epoch(
        self,
        epoch: int,
        step: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
    ) -> None:
        """Write an end-of-epoch summary row to CSV and print a table.

        Args:
            epoch: Current epoch index (1-based).
            step: Global step count at end of epoch.
            train_loss: Mean training loss over the epoch.
            train_acc: Mean training accuracy over the epoch.
            val_loss: Validation loss.
            val_acc: Validation accuracy (0–1).
            lr: Learning rate at end of epoch.
        """
        elapsed = round(time.time() - self._start, 1)
        row = {
            "epoch": epoch,
            "step": step,
            "phase": "epoch",
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr": f"{lr:.2e}",
            "elapsed_s": elapsed,
        }
        self._writer.writerow(row)
        self._fh.flush()

        if _RICH and self._console:
            table = Table(box=rich_box.SIMPLE_HEAVY, show_header=True, padding=(0, 1))
            table.add_column("Epoch", style="bold", justify="right")
            table.add_column("Train loss", justify="right")
            table.add_column("Train acc", justify="right")
            table.add_column("Val loss", justify="right")
            table.add_column("Val acc", justify="right")
            table.add_column("LR", justify="right")
            table.add_column("Elapsed", justify="right")
            table.add_row(
                str(epoch),
                f"{train_loss:.4f}",
                f"{train_acc*100:.2f}%",
                f"[green]{val_loss:.4f}[/green]",
                f"[bold green]{val_acc*100:.2f}%[/bold green]",
                f"{lr:.2e}",
                f"{elapsed}s",
            )
            self._console.print(table)
        else:
            print(
                f"Epoch {epoch} | "
                f"train loss {train_loss:.4f}  acc {train_acc*100:.2f}% | "
                f"val loss {val_loss:.4f}  acc {val_acc*100:.2f}% | "
                f"lr {lr:.2e} | {elapsed}s"
            )

    def close(self) -> None:
        """Flush and close the underlying CSV file handle."""
        self._fh.flush()
        self._fh.close()
        self._print(f"[dim]Logger closed → {self.log_path}[/dim]")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _print(self, msg: str) -> None:
        if _RICH and self._console:
            self._console.print(msg)
        else:
            import re

            print(re.sub(r"\[/?[^\]]+\]", "", msg))
