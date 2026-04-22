"""Integration tests for the full AttnRes and Baseline transformer models.

Tests cover:
  * :class:`~models.transformer.AttnResTransformer` (Full + Block variants)
  * :class:`~models.transformer.BaselineTransformer`
  * :func:`~models.build_model` factory
  * :class:`~models.transformer.PatchEmbedding`
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import AttnResTransformer, BaselineTransformer, build_model
from models.transformer import PatchEmbedding
from utils.config import ModelConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _base_cfg(**overrides) -> ModelConfig:
    """Return a minimal ModelConfig for fast tests."""
    defaults = dict(
        name="TestModel",
        dim=32,
        depth=4,
        heads=2,
        head_dim=16,
        mlp_multiplier=2,
        dropout=0.0,
        use_block_attn_res=True,
        block_size=4,
        norm_eps=1e-6,
        max_seq_len=64,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


MNIST_KWARGS = dict(num_classes=10, img_size=28, patch_size=4, in_channels=1)
CIFAR_KWARGS = dict(num_classes=10, img_size=32, patch_size=4, in_channels=3)


# ---------------------------------------------------------------------------
# PatchEmbedding
# ---------------------------------------------------------------------------


class TestPatchEmbedding:
    """Tests for the patch-embedding projection layer."""

    def test_mnist_shape(self):
        """MNIST 28×28 with patch 4 → 49 patches."""
        pe = PatchEmbedding(img_size=28, patch_size=4, in_channels=1, dim=32)
        x = torch.randn(2, 1, 28, 28)
        out = pe(x)
        assert out.shape == (2, 49, 32)

    def test_cifar_shape(self):
        """CIFAR-10 32×32 with patch 4 → 64 patches."""
        pe = PatchEmbedding(img_size=32, patch_size=4, in_channels=3, dim=32)
        x = torch.randn(2, 3, 32, 32)
        assert pe(x).shape == (2, 64, 32)

    def test_invalid_patch_raises(self):
        """img_size not divisible by patch_size must raise AssertionError."""
        with pytest.raises(AssertionError):
            PatchEmbedding(img_size=28, patch_size=5, in_channels=1, dim=32)

    def test_gradient_flows(self):
        """Gradients must flow from output back to input pixels."""
        pe = PatchEmbedding(img_size=28, patch_size=4, in_channels=1, dim=32)
        x = torch.randn(2, 1, 28, 28, requires_grad=True)
        pe(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# AttnResTransformer — Block variant
# ---------------------------------------------------------------------------


class TestAttnResTransformerBlock:
    """Integration tests for AttnResTransformer with Block AttnRes."""

    def test_mnist_output_shape(self):
        """Logits shape must be (B, num_classes) for MNIST input."""
        model = AttnResTransformer(_base_cfg(use_block_attn_res=True), **MNIST_KWARGS)
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_cifar_output_shape(self):
        """Logits shape must be (B, num_classes) for CIFAR-10 input."""
        model = AttnResTransformer(_base_cfg(use_block_attn_res=True), **CIFAR_KWARGS)
        x = torch.randn(4, 3, 32, 32)
        assert model(x).shape == (4, 10)

    def test_batch_size_one(self):
        """Must work with batch size 1."""
        model = AttnResTransformer(_base_cfg(), **MNIST_KWARGS)
        x = torch.randn(1, 1, 28, 28)
        assert model(x).shape == (1, 10)

    def test_no_nan_in_output(self):
        """Output logits must be finite for random inputs."""
        model = AttnResTransformer(_base_cfg(), **MNIST_KWARGS)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flows_end_to_end(self):
        """Loss must back-prop from logits all the way to input pixels."""
        model = AttnResTransformer(_base_cfg(), **MNIST_KWARGS)
        x = torch.randn(2, 1, 28, 28, requires_grad=True)
        loss = model(x).sum()
        loss.backward()
        assert x.grad is not None

    def test_parameter_count_positive(self):
        """Model must have a non-zero number of trainable parameters."""
        model = AttnResTransformer(_base_cfg(), **MNIST_KWARGS)
        assert model.num_parameters > 0

    def test_train_vs_eval_mode(self):
        """Switching between train and eval must not change output shape."""
        model = AttnResTransformer(_base_cfg(), **MNIST_KWARGS)
        x = torch.randn(2, 1, 28, 28)
        model.train()
        out_train = model(x)
        model.eval()
        with torch.no_grad():
            out_eval = model(x)
        assert out_train.shape == out_eval.shape

    def test_deeper_model(self):
        """A deeper model (depth=8) must still produce correct output shape."""
        model = AttnResTransformer(_base_cfg(depth=8, block_size=4), **MNIST_KWARGS)
        x = torch.randn(2, 1, 28, 28)
        assert model(x).shape == (2, 10)


# ---------------------------------------------------------------------------
# AttnResTransformer — Full variant
# ---------------------------------------------------------------------------


class TestAttnResTransformerFull:
    """Integration tests for AttnResTransformer with Full AttnRes."""

    def test_output_shape(self):
        """Full AttnRes variant must produce (B, num_classes) logits."""
        model = AttnResTransformer(_base_cfg(use_block_attn_res=False), **MNIST_KWARGS)
        x = torch.randn(3, 1, 28, 28)
        assert model(x).shape == (3, 10)

    def test_gradient_flows(self):
        """Gradients must reach input for Full AttnRes."""
        model = AttnResTransformer(_base_cfg(use_block_attn_res=False), **MNIST_KWARGS)
        x = torch.randn(2, 1, 28, 28, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None

    def test_no_nan(self):
        """Full AttnRes output must be finite."""
        model = AttnResTransformer(_base_cfg(use_block_attn_res=False), **MNIST_KWARGS)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# BaselineTransformer
# ---------------------------------------------------------------------------


class TestBaselineTransformer:
    """Integration tests for the standard-residual baseline model."""

    def test_output_shape(self):
        """Baseline must produce (B, num_classes) logits."""
        model = BaselineTransformer(_base_cfg(), **MNIST_KWARGS)
        x = torch.randn(2, 1, 28, 28)
        assert model(x).shape == (2, 10)

    def test_gradient_flows(self):
        """Gradients must flow end-to-end for baseline."""
        model = BaselineTransformer(_base_cfg(), **MNIST_KWARGS)
        x = torch.randn(2, 1, 28, 28, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None

    def test_parameter_count_comparable(self):
        """Baseline and AttnRes models should have similar parameter counts."""
        attnres = AttnResTransformer(_base_cfg(), **MNIST_KWARGS)
        baseline = BaselineTransformer(_base_cfg(), **MNIST_KWARGS)
        ratio = attnres.num_parameters / baseline.num_parameters
        # AttnRes adds one d-dim query vec per sublayer — should stay < 2×
        assert 0.9 < ratio < 2.0, f"Unexpected param ratio: {ratio:.2f}"


# ---------------------------------------------------------------------------
# build_model factory
# ---------------------------------------------------------------------------


class TestBuildModel:
    """Tests for the build_model factory function."""

    def test_builds_attnres(self):
        """build_model without baseline=True must return AttnResTransformer."""
        model = build_model(_base_cfg(), **MNIST_KWARGS, baseline=False)
        assert isinstance(model, AttnResTransformer)

    def test_builds_baseline(self):
        """build_model with baseline=True must return BaselineTransformer."""
        model = build_model(_base_cfg(), **MNIST_KWARGS, baseline=True)
        assert isinstance(model, BaselineTransformer)

    def test_factory_output_shape(self):
        """Factory-built model must produce correct output shape."""
        model = build_model(_base_cfg(), **MNIST_KWARGS)
        x = torch.randn(2, 1, 28, 28)
        assert model(x).shape == (2, 10)
