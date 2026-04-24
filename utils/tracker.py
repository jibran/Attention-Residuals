"""Unified experiment tracker for Weights & Biases and MLflow.

Provides a single :class:`ExperimentTracker` façade that routes metrics,
parameters, and artifacts to whichever backend is configured.  Both backends
are optional — if a library is not installed, a clear :exc:`ImportError` with
an install command is raised at construction time rather than silently failing
mid-run.

Supported backends
------------------
``"none"``
    No-op tracker (default).  Zero overhead; no external dependencies.

``"wandb"``
    Weights & Biases.  Requires ``pip install wandb``.
    Authenticate once with ``wandb login`` before training.

``"mlflow"``
    MLflow.  Requires ``pip install mlflow``.
    Defaults to a local ``mlruns/`` directory; point
    ``logging.mlflow_tracking_uri`` at a remote server for team logging.

Configuration
-------------
All settings live under ``logging:`` in the YAML config::

    logging:
      log_dir: logs/
      checkpoint_dir: checkpoints/
      tracker: wandb                   # none | wandb | mlflow
      wandb_project: attnres           # wandb only
      wandb_run_name: ""               # optional; auto-generated if empty
      mlflow_tracking_uri: mlruns      # mlflow only; use http://... for remote
      mlflow_experiment: AttnRes       # mlflow experiment name

CLI override::

    python train/train_lm.py --config config/tinystories.yaml \\
        --override logging.tracker=wandb logging.wandb_project=my-project

Usage inside training code::

    tracker = ExperimentTracker.from_config(cfg, run_name="AttnResLM", config_dict=cfg.to_dict())
    tracker.log_params(cfg.to_dict())
    tracker.log_metrics({"train/loss": 0.42, "train/ppl": 5.2}, step=100)
    tracker.log_artifact("checkpoints/best/best.pt")
    tracker.finish()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from utils.config import Config

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ExperimentTracker:
    """Uniform façade over W&B, MLflow, and a no-op null tracker.

    Do not instantiate directly — use :meth:`from_config` or
    :meth:`from_backend`.

    Args:
        backend: One of ``"none"``, ``"wandb"``, ``"mlflow"``.
        _run: Internal run object (wandb run or mlflow client wrapper).
    """

    def __init__(self, backend: str, _run: Any = None) -> None:
        self._backend = backend
        self._run = _run

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        run_name: str = "",
        config_dict: dict | None = None,
    ) -> ExperimentTracker:
        """Build a tracker from the experiment :class:`~utils.config.Config`.

        Args:
            cfg: Full experiment configuration.  The ``cfg.logging`` section
                controls which backend is initialised.
            run_name: Human-readable name for this run (shown in the UI).
            config_dict: Hyper-parameter dictionary logged at run start.
                Defaults to ``cfg.to_dict()`` if ``None``.

        Returns:
            Configured :class:`ExperimentTracker` instance.

        Raises:
            ImportError: If the requested backend library is not installed.
            ValueError: If ``cfg.logging.tracker`` is not a recognised value.
        """
        params = config_dict if config_dict is not None else cfg.to_dict()
        return cls.from_backend(
            backend=cfg.logging.tracker,
            run_name=run_name or cfg.model.name,
            config_dict=params,
            wandb_project=cfg.logging.wandb_project,
            wandb_run_name=cfg.logging.wandb_run_name,
            mlflow_tracking_uri=cfg.logging.mlflow_tracking_uri,
            mlflow_experiment=cfg.logging.mlflow_experiment,
        )

    @classmethod
    def from_backend(
        cls,
        backend: str = "none",
        run_name: str = "",
        config_dict: dict | None = None,
        *,
        wandb_project: str = "attnres",
        wandb_run_name: str = "",
        mlflow_tracking_uri: str = "mlruns",
        mlflow_experiment: str = "AttnRes",
    ) -> ExperimentTracker:
        """Construct a tracker for a specific backend.

        Args:
            backend: ``"none"``, ``"wandb"``, or ``"mlflow"``.
            run_name: Name for this run.
            config_dict: Hyper-parameters to log at start.
            wandb_project: W&B project name.
            wandb_run_name: W&B run name (overrides ``run_name`` in the UI).
            mlflow_tracking_uri: Local path or ``http://`` URI for the
                MLflow tracking server.
            mlflow_experiment: MLflow experiment name.

        Returns:
            :class:`ExperimentTracker` instance.

        Raises:
            ImportError: If the requested backend library is not installed.
            ValueError: If ``backend`` is not recognised.
        """
        backend = backend.lower().strip()

        if backend == "none":
            return cls("none")

        if backend == "wandb":
            return cls._init_wandb(
                run_name=wandb_run_name or run_name,
                project=wandb_project,
                config=config_dict or {},
            )

        if backend == "mlflow":
            return cls._init_mlflow(
                run_name=run_name,
                tracking_uri=mlflow_tracking_uri,
                experiment_name=mlflow_experiment,
                params=config_dict or {},
            )

        raise ValueError(
            f"Unknown tracker backend '{backend}'. "
            "Choose one of: 'none', 'wandb', 'mlflow'."
        )

    # ------------------------------------------------------------------
    # Backend initialisers
    # ------------------------------------------------------------------

    @classmethod
    def _init_wandb(
        cls,
        run_name: str,
        project: str,
        config: dict,
    ) -> ExperimentTracker:
        """Initialise a W&B run.

        Args:
            run_name: Display name in the W&B UI.
            project: W&B project to log into.
            config: Hyper-parameter dict shown in the run overview.

        Returns:
            :class:`ExperimentTracker` wrapping the wandb run.

        Raises:
            ImportError: If ``wandb`` is not installed.
        """
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Install it with:\n"
                "    pip install wandb\n"
                "Then authenticate with:\n"
                "    wandb login"
            ) from exc

        run = wandb.init(
            project=project or "attnres",
            name=run_name or None,
            config=config,
            resume="allow",
        )
        print(f"W&B run: {run.url}")
        return cls("wandb", _run=run)

    @classmethod
    def _init_mlflow(
        cls,
        run_name: str,
        tracking_uri: str,
        experiment_name: str,
        params: dict,
    ) -> ExperimentTracker:
        """Initialise an MLflow run.

        Args:
            run_name: MLflow run name.
            tracking_uri: Local directory path or remote ``http://`` URI.
            experiment_name: MLflow experiment to log into.
            params: Hyper-parameters logged via ``mlflow.log_params``.

        Returns:
            :class:`ExperimentTracker` wrapping an (mlflow_client, run_id) pair.

        Raises:
            ImportError: If ``mlflow`` is not installed.
        """
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "mlflow is not installed. Install it with:\n" "    pip install mlflow"
            ) from exc

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        active_run = mlflow.start_run(run_name=run_name or None)

        # Flatten nested param dict; MLflow requires string keys and values
        flat = _flatten_dict(params)
        if flat:
            mlflow.log_params(flat)

        tracking_uri_display = (
            tracking_uri
            if tracking_uri.startswith("http")
            else str(Path(tracking_uri).resolve())
        )
        print(f"MLflow run: {tracking_uri_display}  experiment: {experiment_name}")
        return cls("mlflow", _run=active_run)

    # ------------------------------------------------------------------
    # Uniform logging API
    # ------------------------------------------------------------------

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log a dictionary of scalar metrics.

        Args:
            metrics: Mapping of metric name to float value.
                Use ``"train/loss"``, ``"val/ppl"``, etc. for namespacing.
            step: Global training step.  Passed to the backend so the x-axis
                of charts aligns with step count.  ``None`` lets the backend
                use its own counter.
        """
        if self._backend == "none":
            return

        if self._backend == "wandb":
            import wandb

            wandb.log(metrics, step=step)

        elif self._backend == "mlflow":
            import mlflow

            mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dictionary of hyper-parameters.

        Best called once at run start.  For MLflow, params are flattened and
        truncated to 250 chars per value (MLflow's limit).

        Args:
            params: Nested or flat mapping of parameter names to values.
        """
        if self._backend == "none":
            return

        if self._backend == "wandb":
            import wandb

            wandb.config.update(params, allow_val_change=True)

        elif self._backend == "mlflow":
            import mlflow

            flat = _flatten_dict(params)
            if flat:
                mlflow.log_params(flat)

    def log_artifact(self, path: str | Path) -> None:
        """Upload a file (checkpoint, config, etc.) to the tracking server.

        For W&B, the file is saved to the run's ``files`` section.
        For MLflow, the file is stored in the artifact store.
        No-op for ``"none"``.

        Args:
            path: Path to the local file to upload.
        """
        if self._backend == "none":
            return

        path = str(path)
        if not os.path.isfile(path):
            return

        if self._backend == "wandb":
            import wandb

            wandb.save(path, policy="now")

        elif self._backend == "mlflow":
            import mlflow

            mlflow.log_artifact(path)

    def finish(self) -> None:
        """Mark the run as complete and flush any pending data.

        Should be called at the very end of training, even on error (wrap in
        a ``try/finally`` block).  Safe to call multiple times.
        """
        if self._backend == "none":
            return

        if self._backend == "wandb":
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

        elif self._backend == "mlflow":
            try:
                import mlflow

                mlflow.end_run()
            except Exception:
                pass

        self._backend = "none"  # make subsequent calls no-ops

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> ExperimentTracker:
        """Support ``with ExperimentTracker.from_config(cfg) as tracker:``."""
        return self

    def __exit__(self, *_) -> None:
        """Call :meth:`finish` on exit (including on exception)."""
        self.finish()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        """Active backend name (``"none"``, ``"wandb"``, or ``"mlflow"``)."""
        return self._backend

    @property
    def is_active(self) -> bool:
        """``True`` if a real tracking backend is configured."""
        return self._backend != "none"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatten_dict(
    d: dict,
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, str]:
    """Flatten a nested dict into a single-level dict with dot-separated keys.

    MLflow ``log_params`` requires flat string keys and string values ≤ 250
    chars.  This helper handles arbitrary nesting and converts values to
    strings.

    Args:
        d: Nested dictionary.
        parent_key: Key prefix for recursive calls.
        sep: Separator character between key segments.

    Returns:
        Flat ``{str: str}`` dictionary safe for ``mlflow.log_params``.
    """
    items: dict[str, str] = {}
    for k, v in d.items():
        full_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten_dict(v, full_key, sep=sep))
        else:
            items[full_key] = str(v)[:250]
    return items
