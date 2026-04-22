"""Language-model inference: generate Shakespeare text from a checkpoint.

Loads a trained :class:`~models.AttnResLM` (or :class:`~models.BaselineLM`)
and generates text autoregressively from a user-supplied prompt.

Usage::

    # Generate 500 characters from the best checkpoint:
    python inference/inference_lm.py \\
        --checkpoint checkpoints/best/best.pt \\
        --prompt "HAMLET: To be, or not"

    # Adjust temperature and length:
    python inference/inference_lm.py \\
        --checkpoint checkpoints/best/best.pt \\
        --prompt "KING LEAR:" \\
        --max_new_tokens 800 \\
        --temperature 0.6 \\
        --top_k 50

    # Evaluate perplexity on the test split:
    python inference/inference_lm.py \\
        --checkpoint checkpoints/best/best.pt \\
        --eval
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dataset.shakespeare_dataset import ShakespeareDataset
from dataset.tokenizer import CharTokenizer
from models.lm_transformer import AttnResLM, BaselineLM
from utils.config import (
    Config,
    DataConfig,
    GenerationConfig,
    LoggingConfig,
    ModelConfig,
    TrainingConfig,
)
from utils.device import resolve_device

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load a ``.pt`` checkpoint file.

    Args:
        path: Path to the checkpoint.
        device: Device to map tensors onto.

    Returns:
        Checkpoint dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def _rebuild_config(ckpt_dict: dict) -> Config:
    """Reconstruct a :class:`~utils.config.Config` from a checkpoint dict.

    Args:
        ckpt_dict: Checkpoint dictionary with a ``"config"`` key.

    Returns:
        Populated :class:`~utils.config.Config`.
    """
    raw = ckpt_dict.get("config", {})
    return Config(
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        data=DataConfig(**raw.get("data", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
    )


def _load_tokenizer(cfg: Config) -> CharTokenizer:
    """Load the tokeniser that was saved during training.

    Args:
        cfg: Config containing the data directory path.

    Returns:
        Fitted :class:`~dataset.tokenizer.CharTokenizer`.

    Raises:
        FileNotFoundError: If the vocabulary file has not been created yet.
    """
    vocab_path = Path(cfg.data.data_dir) / "processed" / "vocab.json"
    return CharTokenizer.load(vocab_path)


@torch.no_grad()
def _evaluate_perplexity(
    model: torch.nn.Module,
    cfg: Config,
    tokenizer: CharTokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[float, float]:
    """Compute loss and perplexity on the Shakespeare test split.

    Args:
        model: Language model in eval mode.
        cfg: Experiment config (for data settings).
        tokenizer: Fitted tokeniser.
        device: Inference device.
        batch_size: Batch size for the test loader.

    Returns:
        Tuple ``(mean_loss, perplexity)``.
    """
    loaders, _ = ShakespeareDataset.get_loaders(
        data_dir=cfg.data.data_dir,
        seq_len=cfg.data.seq_len,
        stride=cfg.data.seq_len,  # non-overlapping for eval
        val_split=0.0,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        seed=0,
    )
    loader = loaders["test"]
    total_loss, total_tokens = 0.0, 0
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
    mean_loss = total_loss / max(total_tokens, 1)
    return mean_loss, math.exp(mean_loss)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run LM inference / generation."""
    parser = argparse.ArgumentParser(
        description="Generate Shakespeare text with AttnResLM."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .pt checkpoint file."
    )
    parser.add_argument("--prompt", default="ROMEO: ", help="Seed text for generation.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=None, help="Tokens to generate."
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature."
    )
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filter.")
    parser.add_argument(
        "--eval", action="store_true", help="Compute test-set perplexity."
    )
    parser.add_argument("--device", default="auto", help="Device override.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    ckpt = _load_checkpoint(Path(args.checkpoint), device)
    cfg = _rebuild_config(ckpt)
    tok = _load_tokenizer(cfg)

    # Build model and load weights
    is_baseline = "Baseline" in ckpt.get("config", {}).get("model", {}).get("name", "")
    model_cls = BaselineLM if is_baseline else AttnResLM
    model = model_cls(
        cfg=cfg.model, vocab_size=tok.vocab_size, seq_len=cfg.data.seq_len
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(
        f"Loaded checkpoint — epoch {epoch}  val_loss {val_loss:.4f}"
        f"  ppl {math.exp(min(val_loss, 20)):.1f}"
    )
    print(f"Vocab size: {tok.vocab_size}  |  Params: {model.num_parameters:,}\n")

    # --- Perplexity evaluation
    if args.eval:
        loss, ppl = _evaluate_perplexity(model, cfg, tok, device)
        print(f"Test — loss: {loss:.4f}  perplexity: {ppl:.2f}\n")

    # --- Text generation
    temperature = args.temperature or cfg.generation.temperature
    top_k = args.top_k if args.top_k is not None else cfg.generation.top_k
    max_new_tok = args.max_new_tokens or cfg.generation.max_new_tokens

    prompt_ids = torch.tensor(
        tok.encode(args.prompt), dtype=torch.long, device=device
    ).unsqueeze(0)
    out_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tok,
        temperature=temperature,
        top_k=top_k if top_k > 0 else None,
    )
    generated = tok.decode(out_ids[0].tolist())

    print("─" * 60)
    print(generated)
    print("─" * 60)


if __name__ == "__main__":
    main()
