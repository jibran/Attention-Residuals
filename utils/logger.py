"""Training logger: writes CSV logs, prints rich console output, and forwards
metrics to an optional experiment tracker (W&B or MLflow).

Each training run creates one CSV file at::

    logs/<model_name>-<YYYYMMDDHHMM>.csv

Columns
-------
epoch, step, phase, train_loss, train_acc, val_loss, val_acc, lr, elapsed_s

Usage::

    from utils.tracker import ExperimentTracker

    tracker = ExperimentTracker.from_config(cfg, run_name="AttnResLM")
    logger  = TrainingLogger(log_dir="logs/", model_name="AttnResLM",
                             tracker=tracker)
    logger.log_step(epoch=1, step=50, train_loss=0.42, train_acc=0.87, lr=3e-4)
    logger.log_epoch(epoch=1, step=800, train_loss=0.38, train_acc=0.90,
                     val_loss=0.35, val_acc=0.91, lr=3e-4)
    logger.close()
    tracker.finish()
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.tracker import ExperimentTracker

try:
    from rich import box as rich_box
    from rich.console import Console
    from rich.table import Table

    _RICH = True
except ImportError:
    _RICH = False


class TrainingLogger:
    """Logs training metrics to a timestamped CSV, the console, and an optional
    experiment tracker (W&B / MLflow).

    Attributes:
        log_path: Absolute path to the CSV file for this run.

    Args:
        log_dir: Directory where CSV files are written.
        model_name: Used as the prefix of the CSV filename.
        tracker: Optional :class:`~utils.tracker.ExperimentTracker` instance.
            When supplied, every call to :meth:`log_step` and
            :meth:`log_epoch` also forwards the metrics to the tracker.
            Pass ``None`` (default) for CSV-only logging.
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

    def __init__(
        self,
        log_dir: str | Path,
        model_name: str,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self._start = time.time()
        self._tracker = tracker
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
        if tracker and tracker.is_active:
            self._print(f"[bold]Tracker:[/bold] {tracker.backend}")

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
        """Write a training-step row to CSV, print to console, and track.

        Args:
            epoch: Current epoch index (1-based).
            step: Global optimiser step count.
            train_loss: Loss on the current mini-batch.
            train_acc: Accuracy on the current mini-batch (0–1).
            lr: Current learning rate.
        """
        elapsed = round(time.time() - self._start, 1)
        row = {
            "epoch": epoch,
            "step": step,
            "phase": "train",
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": "",
            "val_acc": "",
            "lr": f"{lr:.2e}",
            "elapsed_s": elapsed,
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

        # Forward to tracker
        if self._tracker and self._tracker.is_active:
            self._tracker.log_metrics(
                {
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "train/lr": lr,
                },
                step=step,
            )

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
        """Write an end-of-epoch summary row to CSV, print a table, and track.

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

        # Forward to tracker
        if self._tracker and self._tracker.is_active:
            self._tracker.log_metrics(
                {
                    "epoch/train_loss": train_loss,
                    "epoch/train_acc": train_acc,
                    "epoch/val_loss": val_loss,
                    "epoch/val_acc": val_acc,
                    "epoch/lr": lr,
                    "epoch/elapsed_s": elapsed,
                },
                step=step,
            )

    def close(self) -> None:
        """Flush and close the underlying CSV file handle.

        Does **not** call :meth:`~utils.tracker.ExperimentTracker.finish` on
        the tracker — the caller is responsible for that so artifacts can be
        uploaded after :meth:`close` is called.
        """
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
