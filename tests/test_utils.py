"""Unit tests for utility modules.

Tests cover:
  * :func:`~utils.config.load_config` and override parsing
  * :class:`~utils.logger.TrainingLogger`
  * :class:`~utils.checkpoint.CheckpointManager`
  * :func:`~utils.device.resolve_device` and :func:`~utils.device.seed_everything`
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.checkpoint import CheckpointManager
from utils.config import Config, load_config
from utils.device import resolve_device, seed_everything
from utils.logger import TrainingLogger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp(tmp_path):
    """Provide a fresh temporary directory for each test."""
    return tmp_path


@pytest.fixture()
def base_yaml(tmp):
    """Write a minimal valid YAML config and return its path."""
    content = """\
model:
  name: "TestModel"
  dim: 64
  depth: 4
  heads: 2
  head_dim: 32
  mlp_multiplier: 2
  dropout: 0.0
  use_block_attn_res: true
  block_size: 2
  norm_eps: 1.0e-6
  max_seq_len: 64

training:
  epochs: 5
  batch_size: 8
  lr: 1.0e-3
  weight_decay: 0.01
  grad_clip: 1.0
  warmup_steps: 10
  log_every: 5
  save_every: 1
  seed: 0
  device: "cpu"

data:
  dataset: "mnist"
  data_dir: "data/"
  num_workers: 0
  pin_memory: false
  val_split: 0.1

logging:
  log_dir: "logs/"
  checkpoint_dir: "checkpoints/"
"""
    p = tmp / "test_config.yaml"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for the YAML config loader."""

    def test_loads_successfully(self, base_yaml):
        """A valid YAML file should load without errors."""
        cfg = load_config(base_yaml)
        assert isinstance(cfg, Config)

    def test_model_fields(self, base_yaml):
        """Loaded model fields must match the YAML values."""
        cfg = load_config(base_yaml)
        assert cfg.model.dim == 64
        assert cfg.model.depth == 4
        assert cfg.model.use_block_attn_res is True

    def test_training_fields(self, base_yaml):
        """Loaded training fields must match the YAML values."""
        cfg = load_config(base_yaml)
        assert cfg.training.epochs == 5
        assert pytest.approx(cfg.training.lr) == 1e-3

    def test_override_int(self, base_yaml):
        """Integer overrides must be applied correctly."""
        cfg = load_config(base_yaml, overrides=["model.dim=128"])
        assert cfg.model.dim == 128

    def test_override_float(self, base_yaml):
        """Float overrides must be applied correctly."""
        cfg = load_config(base_yaml, overrides=["training.lr=5e-4"])
        assert pytest.approx(cfg.training.lr) == 5e-4

    def test_override_bool_true(self, base_yaml):
        """Boolean true override must work."""
        cfg = load_config(base_yaml, overrides=["model.use_block_attn_res=false"])
        assert cfg.model.use_block_attn_res is False

    def test_override_string(self, base_yaml):
        """String overrides must be applied correctly."""
        cfg = load_config(base_yaml, overrides=["data.dataset=cifar10"])
        assert cfg.data.dataset == "cifar10"

    def test_multiple_overrides(self, base_yaml):
        """Multiple overrides must all be applied."""
        cfg = load_config(base_yaml, overrides=["model.dim=256", "training.epochs=10"])
        assert cfg.model.dim == 256
        assert cfg.training.epochs == 10

    def test_missing_file_raises(self, tmp):
        """FileNotFoundError must be raised for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp / "does_not_exist.yaml")

    def test_malformed_override_raises(self, base_yaml):
        """A missing '=' in an override string must raise ValueError."""
        with pytest.raises(ValueError):
            load_config(base_yaml, overrides=["model.dim128"])

    def test_to_dict_round_trips(self, base_yaml):
        """to_dict() must return a nested dict matching the config values."""
        cfg = load_config(base_yaml)
        d = cfg.to_dict()
        assert d["model"]["dim"] == cfg.model.dim
        assert d["training"]["epochs"] == cfg.training.epochs


# ---------------------------------------------------------------------------
# TrainingLogger
# ---------------------------------------------------------------------------


class TestTrainingLogger:
    """Tests for the CSV training logger."""

    def test_creates_csv(self, tmp):
        """Logger must create a CSV file on initialisation."""
        logger = TrainingLogger(log_dir=tmp, model_name="MyModel")
        logger.close()
        csvs = list(tmp.glob("MyModel-*.csv"))
        assert len(csvs) == 1

    def test_log_step_writes_row(self, tmp):
        """log_step must append a row with phase='train' to the CSV."""
        logger = TrainingLogger(log_dir=tmp, model_name="M")
        logger.log_step(epoch=1, step=50, train_loss=0.5, train_acc=0.8, lr=1e-3)
        logger.close()
        rows = list(csv.DictReader(logger.log_path.open()))
        train_rows = [r for r in rows if r["phase"] == "train"]
        assert len(train_rows) == 1
        assert float(train_rows[0]["train_loss"]) == pytest.approx(0.5)

    def test_log_epoch_writes_row(self, tmp):
        """log_epoch must append a row with phase='epoch' to the CSV."""
        logger = TrainingLogger(log_dir=tmp, model_name="M")
        logger.log_epoch(
            epoch=1,
            step=100,
            train_loss=0.4,
            train_acc=0.85,
            val_loss=0.35,
            val_acc=0.90,
            lr=1e-3,
        )
        logger.close()
        rows = list(csv.DictReader(logger.log_path.open()))
        epoch_rows = [r for r in rows if r["phase"] == "epoch"]
        assert len(epoch_rows) == 1
        assert float(epoch_rows[0]["val_acc"]) == pytest.approx(0.90)

    def test_multiple_steps_accumulate(self, tmp):
        """Multiple log_step calls must produce multiple rows."""
        logger = TrainingLogger(log_dir=tmp, model_name="M")
        for step in range(1, 6):
            logger.log_step(
                epoch=1, step=step * 10, train_loss=0.5, train_acc=0.8, lr=1e-3
            )
        logger.close()
        rows = list(csv.DictReader(logger.log_path.open()))
        train_rows = [r for r in rows if r["phase"] == "train"]
        assert len(train_rows) == 5

    def test_csv_has_header(self, tmp):
        """CSV must have the expected column headers."""
        logger = TrainingLogger(log_dir=tmp, model_name="M")
        logger.close()
        with logger.log_path.open() as fh:
            header = fh.readline().strip().split(",")
        assert "epoch" in header
        assert "val_loss" in header
        assert "train_acc" in header

    def test_tokens_per_sec_logged(self, tmp):
        """log_step must write tokens_per_sec to CSV when provided."""
        logger = TrainingLogger(log_dir=tmp, model_name="M")
        logger.log_step(
            epoch=1,
            step=10,
            train_loss=0.5,
            train_acc=0.8,
            lr=1e-3,
            tokens_per_sec=45_000.0,
            step_ms=22.4,
        )
        logger.close()
        rows = list(csv.DictReader(logger.log_path.open()))
        row = [r for r in rows if r["phase"] == "train"][0]
        assert row["tokens_per_sec"] != ""
        assert float(row["tokens_per_sec"]) == pytest.approx(45_000, rel=0.01)
        assert float(row["step_ms"]) == pytest.approx(22.4, rel=0.01)

    def test_step_no_timing_leaves_columns_empty(self, tmp):
        """log_step without timing args must leave timing columns empty."""
        logger = TrainingLogger(log_dir=tmp, model_name="M")
        logger.log_step(epoch=1, step=10, train_loss=0.5, train_acc=0.8, lr=1e-3)
        logger.close()
        rows = list(csv.DictReader(logger.log_path.open()))
        row = [r for r in rows if r["phase"] == "train"][0]
        assert row["tokens_per_sec"] == ""
        assert row["step_ms"] == ""

    def test_epoch_time_s_logged(self, tmp):
        """log_epoch must write epoch_time_s to CSV when provided."""
        logger = TrainingLogger(log_dir=tmp, model_name="M")
        logger.log_epoch(
            epoch=1,
            step=100,
            train_loss=0.4,
            train_acc=0.85,
            val_loss=0.35,
            val_acc=0.90,
            lr=1e-3,
            epoch_time_s=142.7,
        )
        logger.close()
        rows = list(csv.DictReader(logger.log_path.open()))
        row = [r for r in rows if r["phase"] == "epoch"][0]
        assert float(row["epoch_time_s"]) == pytest.approx(142.7, rel=0.01)

    def test_epoch_no_timing_leaves_column_empty(self, tmp):
        """log_epoch without epoch_time_s must leave that column empty."""
        logger = TrainingLogger(log_dir=tmp, model_name="M")
        logger.log_epoch(
            epoch=1,
            step=100,
            train_loss=0.4,
            train_acc=0.85,
            val_loss=0.35,
            val_acc=0.90,
            lr=1e-3,
        )
        logger.close()
        rows = list(csv.DictReader(logger.log_path.open()))
        row = [r for r in rows if r["phase"] == "epoch"][0]
        assert row["epoch_time_s"] == ""

    def test_csv_timing_columns_in_header(self, tmp):
        """CSV header must include the new timing columns."""
        logger = TrainingLogger(log_dir=tmp, model_name="M")
        logger.close()
        header = logger.log_path.read_text().splitlines()[0].split(",")
        assert "tokens_per_sec" in header
        assert "step_ms" in header
        assert "epoch_time_s" in header
        assert "elapsed_s" in header


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


def _tiny_model():
    """Return a minimal linear model for checkpoint tests."""
    return torch.nn.Linear(4, 2)


def _dummy_config():
    """Return a default Config for checkpoint tests."""
    return Config()


class TestCheckpointManager:
    """Tests for the checkpoint save/load manager."""

    def test_saves_latest(self, tmp):
        """save() must create checkpoints/latest/latest.pt."""
        mgr = CheckpointManager(tmp / "ckpts", _dummy_config())
        model = _tiny_model()
        opt = torch.optim.AdamW(model.parameters())
        mgr.save(model, opt, None, epoch=1, step=10, val_loss=0.5, val_acc=0.8)
        assert (tmp / "ckpts" / "latest" / "latest.pt").exists()

    def test_saves_best_on_improvement(self, tmp):
        """save() must create best.pt when val_loss improves."""
        mgr = CheckpointManager(tmp / "ckpts", _dummy_config())
        model = _tiny_model()
        opt = torch.optim.AdamW(model.parameters())
        saved = mgr.save(model, opt, None, epoch=1, step=10, val_loss=0.5, val_acc=0.8)
        assert "best" in saved
        assert (tmp / "ckpts" / "best" / "best.pt").exists()

    def test_does_not_overwrite_best_if_no_improvement(self, tmp):
        """best.pt must not be overwritten when val_loss does not improve."""
        mgr = CheckpointManager(tmp / "ckpts", _dummy_config())
        model = _tiny_model()
        opt = torch.optim.AdamW(model.parameters())
        mgr.save(model, opt, None, epoch=1, step=10, val_loss=0.3, val_acc=0.9)
        best_mtime = (tmp / "ckpts" / "best" / "best.pt").stat().st_mtime
        saved2 = mgr.save(
            model, opt, None, epoch=2, step=20, val_loss=0.5, val_acc=0.85
        )
        assert "best" not in saved2
        assert (tmp / "ckpts" / "best" / "best.pt").stat().st_mtime == best_mtime

    def test_load_restores_weights(self, tmp):
        """load() must restore model weights that were previously saved."""
        mgr = CheckpointManager(tmp / "ckpts", _dummy_config())
        model = _tiny_model()
        opt = torch.optim.AdamW(model.parameters())
        # Modify weights so they're non-default
        with torch.no_grad():
            model.weight.fill_(3.14)
        mgr.save(model, opt, None, epoch=1, step=10, val_loss=0.4, val_acc=0.9)

        # Load into a fresh model
        model2 = _tiny_model()
        mgr.load(model2)
        assert torch.allclose(model.weight, model2.weight)

    def test_load_returns_epoch_and_step(self, tmp):
        """load() must return (epoch, step) matching the saved values."""
        mgr = CheckpointManager(tmp / "ckpts", _dummy_config())
        model = _tiny_model()
        opt = torch.optim.AdamW(model.parameters())
        mgr.save(model, opt, None, epoch=7, step=350, val_loss=0.2, val_acc=0.95)
        epoch, step = mgr.load(model)
        assert epoch == 7
        assert step == 350

    def test_missing_checkpoint_raises(self, tmp):
        """load() must raise FileNotFoundError if the checkpoint is absent."""
        mgr = CheckpointManager(tmp / "ckpts", _dummy_config())
        with pytest.raises(FileNotFoundError):
            mgr.load(_tiny_model())

    def test_best_val_loss_tracked(self, tmp):
        """best_val_loss property must track the minimum seen so far."""
        mgr = CheckpointManager(tmp / "ckpts", _dummy_config())
        model = _tiny_model()
        opt = torch.optim.AdamW(model.parameters())
        mgr.save(model, opt, None, epoch=1, step=10, val_loss=0.8, val_acc=0.7)
        mgr.save(model, opt, None, epoch=2, step=20, val_loss=0.5, val_acc=0.8)
        mgr.save(model, opt, None, epoch=3, step=30, val_loss=0.6, val_acc=0.75)
        assert pytest.approx(mgr.best_val_loss) == 0.5


# ---------------------------------------------------------------------------
# resolve_device / seed_everything
# ---------------------------------------------------------------------------


class TestDeviceUtils:
    """Tests for device resolution and seeding utilities."""

    def test_auto_returns_device(self):
        """resolve_device('auto') must return a valid torch.device."""
        dev = resolve_device("auto")
        assert isinstance(dev, torch.device)

    def test_cpu_explicit(self):
        """resolve_device('cpu') must return torch.device('cpu')."""
        assert resolve_device("cpu") == torch.device("cpu")

    def test_seed_reproducibility(self):
        """Two calls with the same seed must produce identical random tensors."""
        seed_everything(42)
        t1 = torch.randn(10)
        seed_everything(42)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2)

    def test_different_seeds_differ(self):
        """Different seeds must (almost certainly) produce different tensors."""
        seed_everything(1)
        t1 = torch.randn(100)
        seed_everything(2)
        t2 = torch.randn(100)
        assert not torch.allclose(t1, t2)
