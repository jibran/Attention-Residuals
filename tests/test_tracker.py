"""Unit tests for ExperimentTracker and TrainingLogger tracker integration.

All tests run fully offline — no W&B or MLflow connection is made.  The
backend libraries are mocked at the module level.

Covers:
  * :class:`~utils.tracker.ExperimentTracker` null backend
  * W&B backend (mocked ``wandb`` module)
  * MLflow backend (mocked ``mlflow`` module)
  * :func:`~utils.tracker._flatten_dict` helper
  * :class:`~utils.logger.TrainingLogger` with and without tracker
  * Config flag ``logging.tracker`` wired through YAML and CLI overrides
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import Config, LoggingConfig
from utils.logger import TrainingLogger
from utils.tracker import ExperimentTracker, _flatten_dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _null_tracker() -> ExperimentTracker:
    return ExperimentTracker.from_backend("none")


def _mock_wandb():
    """Return a mock wandb module wired to sys.modules."""
    mod = types.ModuleType("wandb")
    run = MagicMock()
    run.url = "https://wandb.ai/test/run"
    mod.init = MagicMock(return_value=run)
    mod.log = MagicMock()
    mod.finish = MagicMock()
    mod.save = MagicMock()
    mod.config = MagicMock()
    mod.config.update = MagicMock()
    sys.modules["wandb"] = mod
    return mod


def _mock_mlflow():
    """Return a mock mlflow module wired to sys.modules."""
    mod = types.ModuleType("mlflow")
    run = MagicMock()
    run.info.run_id = "test-run-id"
    mod.set_tracking_uri = MagicMock()
    mod.set_experiment = MagicMock()
    mod.start_run = MagicMock(return_value=run)
    mod.log_params = MagicMock()
    mod.log_metrics = MagicMock()
    mod.log_artifact = MagicMock()
    mod.end_run = MagicMock()
    sys.modules["mlflow"] = mod
    return mod


# ---------------------------------------------------------------------------
# _flatten_dict
# ---------------------------------------------------------------------------


class TestFlattenDict:
    def test_flat_dict_unchanged(self):
        d = {"a": 1, "b": "hello"}
        assert _flatten_dict(d) == {"a": "1", "b": "hello"}

    def test_nested_dict_flattened(self):
        d = {"model": {"dim": 256, "depth": 8}, "lr": 3e-4}
        flat = _flatten_dict(d)
        assert flat["model.dim"] == "256"
        assert flat["model.depth"] == "8"
        assert "lr" in flat

    def test_deeply_nested(self):
        d = {"a": {"b": {"c": 42}}}
        assert _flatten_dict(d) == {"a.b.c": "42"}

    def test_values_truncated_to_250(self):
        long_val = "x" * 300
        flat = _flatten_dict({"k": long_val})
        assert len(flat["k"]) == 250

    def test_empty_dict(self):
        assert _flatten_dict({}) == {}


# ---------------------------------------------------------------------------
# Null tracker
# ---------------------------------------------------------------------------


class TestNullTracker:
    def test_backend_is_none(self):
        t = _null_tracker()
        assert t.backend == "none"

    def test_is_not_active(self):
        assert not _null_tracker().is_active

    def test_log_metrics_no_error(self):
        _null_tracker().log_metrics({"loss": 0.5}, step=1)

    def test_log_params_no_error(self):
        _null_tracker().log_params({"lr": 3e-4})

    def test_log_artifact_no_error(self):
        _null_tracker().log_artifact("/nonexistent/path.pt")

    def test_finish_no_error(self):
        _null_tracker().finish()

    def test_finish_idempotent(self):
        t = _null_tracker()
        t.finish()
        t.finish()  # second call must not raise

    def test_context_manager(self):
        with ExperimentTracker.from_backend("none") as t:
            t.log_metrics({"loss": 0.1})

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown tracker backend"):
            ExperimentTracker.from_backend("tensorboard")


# ---------------------------------------------------------------------------
# W&B backend (mocked)
# ---------------------------------------------------------------------------


class TestWandbTracker:
    def setup_method(self):
        self.wandb = _mock_wandb()

    def teardown_method(self):
        sys.modules.pop("wandb", None)

    def test_init_calls_wandb_init(self):
        t = ExperimentTracker.from_backend(
            "wandb",
            run_name="test_run",
            wandb_project="my-project",
            config_dict={"lr": 3e-4},
        )
        assert t.backend == "wandb"
        assert t.is_active
        self.wandb.init.assert_called_once()
        call_kwargs = self.wandb.init.call_args.kwargs
        assert call_kwargs["project"] == "my-project"
        assert call_kwargs["name"] == "test_run"

    def test_backend_is_wandb(self):
        t = ExperimentTracker.from_backend("wandb", wandb_project="p")
        assert t.backend == "wandb"
        assert t.is_active

    def test_log_metrics_calls_wandb_log(self):
        t = ExperimentTracker.from_backend("wandb", wandb_project="p")
        t.log_metrics({"train/loss": 0.42, "train/acc": 0.88}, step=100)
        self.wandb.log.assert_called_once_with(
            {"train/loss": 0.42, "train/acc": 0.88}, step=100
        )

    def test_log_params_calls_config_update(self):
        t = ExperimentTracker.from_backend("wandb", wandb_project="p")
        t.log_params({"lr": 1e-3})
        self.wandb.config.update.assert_called_with({"lr": 1e-3}, allow_val_change=True)

    def test_log_artifact_calls_wandb_save(self, tmp_path):
        f = tmp_path / "model.pt"
        f.write_bytes(b"fake")
        t = ExperimentTracker.from_backend("wandb", wandb_project="p")
        t.log_artifact(f)
        self.wandb.save.assert_called_once_with(str(f), policy="now")

    def test_log_artifact_nonexistent_file_no_error(self):
        t = ExperimentTracker.from_backend("wandb", wandb_project="p")
        t.log_artifact("/does/not/exist.pt")  # must not raise

    def test_finish_calls_wandb_finish(self):
        t = ExperimentTracker.from_backend("wandb", wandb_project="p")
        t.finish()
        self.wandb.finish.assert_called_once()

    def test_finish_sets_backend_to_none(self):
        t = ExperimentTracker.from_backend("wandb", wandb_project="p")
        t.finish()
        assert t.backend == "none"

    def test_context_manager_calls_finish(self):
        with ExperimentTracker.from_backend("wandb", wandb_project="p"):
            pass
        self.wandb.finish.assert_called_once()

    def test_missing_wandb_raises_import_error(self):
        sys.modules.pop("wandb", None)
        with patch.dict(sys.modules, {"wandb": None}):
            with pytest.raises((ImportError, TypeError)):
                ExperimentTracker.from_backend("wandb")


# ---------------------------------------------------------------------------
# MLflow backend (mocked)
# ---------------------------------------------------------------------------


class TestMLflowTracker:
    def setup_method(self):
        self.mlflow = _mock_mlflow()

    def teardown_method(self):
        sys.modules.pop("mlflow", None)

    def test_init_sets_tracking_uri(self):
        ExperimentTracker.from_backend(
            "mlflow", mlflow_tracking_uri="mlruns", mlflow_experiment="Exp"
        )
        self.mlflow.set_tracking_uri.assert_called_once_with("mlruns")

    def test_init_sets_experiment(self):
        ExperimentTracker.from_backend("mlflow", mlflow_experiment="MyExp")
        self.mlflow.set_experiment.assert_called_once_with("MyExp")

    def test_backend_is_mlflow(self):
        t = ExperimentTracker.from_backend("mlflow")
        assert t.backend == "mlflow"
        assert t.is_active

    def test_init_logs_params(self):
        ExperimentTracker.from_backend("mlflow", config_dict={"model": {"dim": 256}})
        self.mlflow.log_params.assert_called()
        flat = self.mlflow.log_params.call_args.args[0]
        assert "model.dim" in flat

    def test_log_metrics_calls_mlflow_log_metrics(self):
        t = ExperimentTracker.from_backend("mlflow")
        t.log_metrics({"val/loss": 0.31}, step=500)
        self.mlflow.log_metrics.assert_called_once_with({"val/loss": 0.31}, step=500)

    def test_log_artifact_calls_mlflow_log_artifact(self, tmp_path):
        f = tmp_path / "ckpt.pt"
        f.write_bytes(b"fake")
        t = ExperimentTracker.from_backend("mlflow")
        t.log_artifact(f)
        self.mlflow.log_artifact.assert_called_once_with(str(f))

    def test_finish_calls_end_run(self):
        t = ExperimentTracker.from_backend("mlflow")
        t.finish()
        self.mlflow.end_run.assert_called_once()

    def test_context_manager_calls_end_run(self):
        with ExperimentTracker.from_backend("mlflow"):
            pass
        self.mlflow.end_run.assert_called_once()

    def test_missing_mlflow_raises_import_error(self):
        sys.modules.pop("mlflow", None)
        with patch.dict(sys.modules, {"mlflow": None}):
            with pytest.raises((ImportError, TypeError)):
                ExperimentTracker.from_backend("mlflow")


# ---------------------------------------------------------------------------
# from_config factory
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_none_backend_from_config(self):
        cfg = Config()
        cfg.logging.tracker = "none"
        t = ExperimentTracker.from_config(cfg)
        assert t.backend == "none"

    def test_wandb_backend_from_config(self):
        _mock_wandb()
        try:
            cfg = Config()
            cfg.logging.tracker = "wandb"
            cfg.logging.wandb_project = "test-proj"
            t = ExperimentTracker.from_config(cfg, run_name="my_run")
            assert t.backend == "wandb"
        finally:
            sys.modules.pop("wandb", None)

    def test_mlflow_backend_from_config(self):
        _mock_mlflow()
        try:
            cfg = Config()
            cfg.logging.tracker = "mlflow"
            cfg.logging.mlflow_experiment = "TestExp"
            t = ExperimentTracker.from_config(cfg)
            assert t.backend == "mlflow"
        finally:
            sys.modules.pop("mlflow", None)


# ---------------------------------------------------------------------------
# LoggingConfig
# ---------------------------------------------------------------------------


class TestLoggingConfig:
    def test_default_tracker_is_none(self):
        assert LoggingConfig().tracker == "none"

    def test_default_wandb_project(self):
        assert LoggingConfig().wandb_project == "attnres"

    def test_default_mlflow_uri(self):
        assert LoggingConfig().mlflow_tracking_uri == "mlruns"

    def test_default_mlflow_experiment(self):
        assert LoggingConfig().mlflow_experiment == "AttnRes"

    def test_serialises_in_to_dict(self):
        cfg = Config()
        cfg.logging.tracker = "wandb"
        d = cfg.to_dict()
        assert d["logging"]["tracker"] == "wandb"
        assert "wandb_project" in d["logging"]
        assert "mlflow_tracking_uri" in d["logging"]

    def test_yaml_override_tracker(self, tmp_path):
        from utils.config import load_config

        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(
            "model:\n  name: test\n"
            "training:\n  device: cpu\n"
            "data:\n  dataset: shakespeare\n"
            "logging:\n"
            "  tracker: none\n"
            "  log_dir: logs/\n"
            "  checkpoint_dir: checkpoints/\n"
            "generation:\n  max_new_tokens: 100\n  temperature: 1.0\n  top_k: 0\n"
        )
        cfg = load_config(yaml_path, overrides=["logging.tracker=wandb"])
        assert cfg.logging.tracker == "wandb"

    def test_yaml_override_wandb_project(self, tmp_path):
        from utils.config import load_config

        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(
            "model:\n  name: test\n"
            "training:\n  device: cpu\n"
            "data:\n  dataset: shakespeare\n"
            "logging:\n"
            "  tracker: none\n"
            "  log_dir: logs/\n"
            "  checkpoint_dir: checkpoints/\n"
            "generation:\n  max_new_tokens: 100\n  temperature: 1.0\n  top_k: 0\n"
        )
        cfg = load_config(yaml_path, overrides=["logging.wandb_project=my-experiments"])
        assert cfg.logging.wandb_project == "my-experiments"


# ---------------------------------------------------------------------------
# TrainingLogger + tracker integration
# ---------------------------------------------------------------------------


class TestTrainingLoggerWithTracker:
    def test_step_forwarded_to_tracker(self, tmp_path):
        tracker = MagicMock(spec=ExperimentTracker)
        tracker.is_active = True
        logger = TrainingLogger(log_dir=tmp_path, model_name="test", tracker=tracker)
        logger.log_step(epoch=1, step=10, train_loss=0.5, train_acc=0.8, lr=1e-3)
        tracker.log_metrics.assert_called_once()
        metrics, kwargs = (
            tracker.log_metrics.call_args.args[0],
            tracker.log_metrics.call_args.kwargs,
        )
        assert "train/loss" in metrics
        assert metrics["train/loss"] == pytest.approx(0.5)
        assert kwargs.get("step") == 10
        logger.close()

    def test_epoch_forwarded_to_tracker(self, tmp_path):
        tracker = MagicMock(spec=ExperimentTracker)
        tracker.is_active = True
        logger = TrainingLogger(log_dir=tmp_path, model_name="test", tracker=tracker)
        logger.log_epoch(
            epoch=1,
            step=100,
            train_loss=0.4,
            train_acc=0.85,
            val_loss=0.38,
            val_acc=0.87,
            lr=1e-3,
        )
        tracker.log_metrics.assert_called_once()
        metrics = tracker.log_metrics.call_args.args[0]
        assert "epoch/val_loss" in metrics
        assert metrics["epoch/val_loss"] == pytest.approx(0.38)
        logger.close()

    def test_inactive_tracker_not_called(self, tmp_path):
        tracker = MagicMock(spec=ExperimentTracker)
        tracker.is_active = False
        logger = TrainingLogger(log_dir=tmp_path, model_name="test", tracker=tracker)
        logger.log_step(epoch=1, step=1, train_loss=0.5, train_acc=0.8, lr=1e-3)
        tracker.log_metrics.assert_not_called()
        logger.close()

    def test_no_tracker_still_writes_csv(self, tmp_path):
        logger = TrainingLogger(log_dir=tmp_path, model_name="test", tracker=None)
        logger.log_step(epoch=1, step=5, train_loss=0.6, train_acc=0.75, lr=2e-4)
        logger.close()
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 1
        content = csv_files[0].read_text()
        assert "0.6" in content

    def test_step_logged_with_step_number(self, tmp_path):
        tracker = MagicMock(spec=ExperimentTracker)
        tracker.is_active = True
        logger = TrainingLogger(log_dir=tmp_path, model_name="test", tracker=tracker)
        logger.log_step(epoch=2, step=42, train_loss=0.3, train_acc=0.9, lr=5e-4)
        _, kwargs = tracker.log_metrics.call_args.args[0], tracker.log_metrics.call_args
        assert kwargs.kwargs.get("step") == 42 or kwargs.args[1] == 42
        logger.close()
