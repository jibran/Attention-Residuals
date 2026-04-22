"""Inference script: load a trained checkpoint and run predictions.

Supports three modes:

1. **Single image** — pass ``--input path/to/image.png``.
2. **Test-set evaluation** — pass ``--eval`` to run over the full test split and
   print accuracy.
3. **Batch directory** — pass ``--input path/to/dir/`` to predict on every image
   in a directory.

Usage::

    # Evaluate on the test set:
    python inference/inference.py \\
        --checkpoint checkpoints/best/best.pt \\
        --eval

    # Predict a single image:
    python inference/inference.py \\
        --checkpoint checkpoints/best/best.pt \\
        --input data/sample.png

    # Override dataset (if saved config doesn't match):
    python inference/inference.py \\
        --checkpoint checkpoints/best/best.pt \\
        --eval --dataset mnist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from dataset.image_datasets import get_dataset_class
from models import build_model
from utils.config import Config, ModelConfig, TrainingConfig, DataConfig, LoggingConfig
from utils.device import resolve_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
_MNIST_CLASSES = [str(i) for i in range(10)]


def _class_names(dataset: str) -> list[str]:
    """Return human-readable class names for a dataset.

    Args:
        dataset: ``"mnist"`` or ``"cifar10"``.

    Returns:
        List of class name strings.
    """
    return _MNIST_CLASSES if dataset == "mnist" else _CIFAR10_CLASSES


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load a ``.pt`` checkpoint dict.

    Args:
        path: Path to ``.pt`` file.
        device: Device to map tensors onto.

    Returns:
        Checkpoint dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def _rebuild_config(ckpt_dict: dict) -> Config:
    """Reconstruct a :class:`~utils.config.Config` from a checkpoint dict.

    Args:
        ckpt_dict: Checkpoint dictionary containing a ``"config"`` key.

    Returns:
        Populated :class:`~utils.config.Config`.
    """
    raw = ckpt_dict.get("config", {})
    return Config(
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        data=DataConfig(**raw.get("data", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
    )


def _get_image_transform(dataset: str):
    """Return the normalisation transform for a given dataset.

    Args:
        dataset: ``"mnist"`` or ``"cifar10"``.

    Returns:
        A :class:`~torchvision.transforms.Compose` pipeline.
    """
    if dataset == "mnist":
        return transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    dataset: str,
    device: torch.device,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Predict the top-k classes for a single image file.

    Args:
        model: Trained model in eval mode.
        image_path: Path to an image file.
        dataset: Dataset name (used to select the correct transform).
        device: Inference device.
        top_k: Number of top predictions to return.

    Returns:
        List of ``(class_name, probability)`` tuples sorted by probability
        descending.
    """
    transform = _get_image_transform(dataset)
    img = Image.open(image_path).convert("RGB" if dataset == "cifar10" else "L")
    tensor = transform(img).unsqueeze(0).to(device)  # (1, C, H, W)

    logits = model(tensor)
    probs = F.softmax(logits, dim=-1).squeeze(0)

    k = min(top_k, probs.size(0))
    top_probs, top_idx = probs.topk(k)
    names = _class_names(dataset)
    return [(names[i], top_probs[j].item()) for j, i in enumerate(top_idx.tolist())]


@torch.no_grad()
def evaluate_test_set(
    model: torch.nn.Module,
    dataset: str,
    data_dir: str,
    batch_size: int,
    device: torch.device,
) -> float:
    """Run the model over the full test split and return accuracy.

    Args:
        model: Trained model in eval mode.
        dataset: Dataset identifier.
        data_dir: Root data directory.
        batch_size: Batch size for the test loader.
        device: Inference device.

    Returns:
        Test accuracy in ``[0, 1]``.
    """
    from torch.utils.data import DataLoader

    ds_cls = get_dataset_class(dataset)
    ds_meta = {
        "mnist": dict(img_size=28, in_channels=1),
        "cifar10": dict(img_size=32, in_channels=3),
    }[dataset]

    test_ds = ds_cls(root=data_dir, train=False, augment=False)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run inference."""
    parser = argparse.ArgumentParser(description="AttnRes inference / evaluation.")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .pt checkpoint file."
    )
    parser.add_argument(
        "--input", default=None, help="Image file or directory to predict."
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate on the test set.")
    parser.add_argument(
        "--dataset", default=None, help="Override dataset (mnist|cifar10)."
    )
    parser.add_argument("--device", default="auto", help="Device override.")
    parser.add_argument(
        "--top_k", type=int, default=5, help="Top-k predictions to show."
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    ckpt = _load_checkpoint(Path(args.checkpoint), device)
    cfg = _rebuild_config(ckpt)

    dataset = args.dataset or cfg.data.dataset
    ds_info = {
        "mnist": dict(img_size=28, in_channels=1, num_classes=10),
        "cifar10": dict(img_size=32, in_channels=3, num_classes=10),
    }[dataset]

    model = build_model(cfg=cfg.model, **ds_info).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    vacc = ckpt.get("val_acc", 0.0)
    print(f"Loaded checkpoint — epoch {epoch}, val acc {vacc*100:.2f}%")

    # ---- Test-set evaluation
    if args.eval:
        acc = evaluate_test_set(
            model,
            dataset,
            cfg.data.data_dir,
            cfg.training.batch_size,
            device,
        )
        print(f"Test accuracy ({dataset}): {acc*100:.2f}%")

    # ---- Single image or directory
    if args.input:
        input_path = Path(args.input)
        paths = list(input_path.glob("*")) if input_path.is_dir() else [input_path]
        for p in paths:
            try:
                results = predict_image(model, p, dataset, device, args.top_k)
                print(f"\n{p.name}")
                for name, prob in results:
                    bar = "█" * int(prob * 20)
                    print(f"  {name:12s} {prob*100:5.1f}%  {bar}")
            except Exception as exc:
                print(f"  [skip] {p.name}: {exc}")


if __name__ == "__main__":
    main()
