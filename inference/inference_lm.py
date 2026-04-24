"""Language-model inference: generate text and evaluate perplexity.

Supports both the character-level Shakespeare model and the BPE TinyStories
model.  The dataset type is detected automatically from the saved config.

Usage::

    # Generate text (TinyStories):
    python inference/inference_lm.py \\
        --checkpoint checkpoints/best/best.pt \\
        --prompt "Once upon a time"

    # Evaluate val-set perplexity:
    python inference/inference_lm.py \\
        --checkpoint checkpoints/best/best.pt \\
        --eval

    # Override sampling parameters:
    python inference/inference_lm.py \\
        --checkpoint checkpoints/best/best.pt \\
        --prompt "Once upon a time" \\
        --max_new_tokens 400 \\
        --temperature 0.7 \\
        --top_k 50
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

import torch

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


def _load_tokenizer(cfg: Config) -> Any:
    """Load the tokeniser that matches the checkpoint's dataset.

    For Shakespeare: loads the ``CharTokenizer`` from ``vocab.json``.
    For TinyStories: loads the ``EleutherAI/gpt-neo-125M`` HF tokeniser.

    Args:
        cfg: Config containing dataset name and data directory.

    Returns:
        Fitted tokeniser (``CharTokenizer`` or HuggingFace tokeniser).
    """
    dataset = cfg.data.dataset.lower()
    if dataset == "shakespeare":
        from dataset.tokenizer import CharTokenizer

        vocab_path = Path(cfg.data.data_dir) / "processed" / "vocab.json"
        return CharTokenizer.load(vocab_path)

    if dataset == "tinystories":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    raise ValueError(f"Unknown dataset '{cfg.data.dataset}' in checkpoint config.")


def _encode(prompt: str, tokenizer: Any, device: torch.device) -> torch.Tensor:
    """Encode a prompt string to a ``(1, T)`` token-id tensor.

    Args:
        prompt: Seed text string.
        tokenizer: Fitted tokeniser.
        device: Target device.

    Returns:
        Token ids tensor ``(1, T)``.
    """
    if hasattr(tokenizer, "input_ids"):
        # HF tokeniser with return_tensors
        return tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    # Both CharTokenizer.encode and HF tokeniser.encode return a list of ints
    ids = tokenizer.encode(prompt)
    if isinstance(ids, list):
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    return ids.to(device)


def _decode(ids: list[int], tokenizer: Any) -> str:
    """Decode a list of token ids to a string.

    Args:
        ids: List of integer token ids.
        tokenizer: Fitted tokeniser.

    Returns:
        Decoded text string.
    """
    return tokenizer.decode(ids)


@torch.no_grad()
def _evaluate_perplexity(
    model: torch.nn.Module,
    cfg: Config,
    tokenizer: Any,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[float, float]:
    """Compute loss and perplexity on the dataset's validation split.

    Args:
        model: Language model in eval mode.
        cfg: Experiment config.
        tokenizer: Fitted tokeniser.
        device: Inference device.
        batch_size: Batch size for the eval loader.

    Returns:
        Tuple ``(mean_loss, perplexity)``.
    """
    dataset = cfg.data.dataset.lower()
    if dataset == "shakespeare":
        from dataset.shakespeare_dataset import ShakespeareDataset

        loaders, _ = ShakespeareDataset.get_loaders(
            data_dir=cfg.data.data_dir,
            seq_len=cfg.data.seq_len,
            stride=cfg.data.seq_len,
            val_split=0.1,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            seed=0,
        )
        loader = loaders["val"]
    elif dataset == "tinystories":
        from dataset.tinystories_dataset import TinyStoriesDataset

        loaders, _ = TinyStoriesDataset.get_loaders(
            data_dir=cfg.data.data_dir,
            seq_len=cfg.data.seq_len,
            stride=cfg.data.seq_len,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            seed=0,
        )
        loader = loaders["val"]
    else:
        raise ValueError(f"Unknown dataset '{cfg.data.dataset}'.")

    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
    mean_loss = total_loss / max(total_tokens, 1)
    return mean_loss, math.exp(min(mean_loss, 20))


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run LM inference / generation."""
    parser = argparse.ArgumentParser(
        description="Generate text / evaluate perplexity with AttnResLM."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .pt checkpoint file."
    )
    parser.add_argument("--prompt", default=None, help="Seed text for generation.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=None, help="Tokens to generate."
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature."
    )
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filter.")
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Enable KV-cache for faster generation.",
    )
    parser.add_argument(
        "--eval", action="store_true", help="Compute validation-set perplexity."
    )
    parser.add_argument("--device", default="auto", help="Device override.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    ckpt = _load_checkpoint(Path(args.checkpoint), device)
    cfg = _rebuild_config(ckpt)
    tok = _load_tokenizer(cfg)

    vocab_size = tok.vocab_size if hasattr(tok, "vocab_size") else len(tok)

    # Build model and load weights
    is_baseline = "Baseline" in ckpt.get("config", {}).get("model", {}).get("name", "")
    model_cls = BaselineLM if is_baseline else AttnResLM
    model = model_cls(
        cfg=cfg.model, vocab_size=vocab_size, seq_len=cfg.data.seq_len
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(
        f"Loaded checkpoint — epoch {epoch}  "
        f"val_loss {val_loss:.4f}  ppl {math.exp(min(val_loss, 20)):.1f}"
    )
    print(f"Vocab size: {vocab_size}  |  Params: {model.num_parameters:,}\n")

    # --- Perplexity evaluation
    if args.eval:
        t_eval0 = time.perf_counter()
        loss, ppl = _evaluate_perplexity(model, cfg, tok, device)
        eval_time_s = time.perf_counter() - t_eval0
        print(f"Val — loss: {loss:.4f}  perplexity: {ppl:.2f}  ({eval_time_s:.1f}s)\n")

    # --- Text generation
    temperature = args.temperature or cfg.generation.temperature
    top_k = args.top_k if args.top_k is not None else cfg.generation.top_k
    max_new_tok = args.max_new_tokens or cfg.generation.max_new_tokens
    prompt = args.prompt or (
        "Once upon a time" if cfg.data.dataset.lower() == "tinystories" else "ROMEO: "
    )
    use_cache = args.use_kv_cache or cfg.model.use_kv_cache

    prompt_ids = _encode(prompt, tok, device)
    prompt_len = prompt_ids.shape[1]
    cache_label = "KV-cache" if use_cache else "no cache"

    t0 = time.perf_counter()
    out_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tok,
        temperature=temperature,
        top_k=top_k if top_k > 0 else None,
        use_kv_cache=use_cache,
    )
    gen_time_s = time.perf_counter() - t0
    new_tokens = out_ids.shape[1] - prompt_len
    tok_per_sec = new_tokens / max(gen_time_s, 1e-9)
    ms_per_tok = gen_time_s * 1000 / max(new_tokens, 1)

    generated = _decode(out_ids[0].tolist(), tok)

    print("─" * 60)
    print(generated)
    print("─" * 60)
    print(
        f"Generated {new_tokens} tokens in {gen_time_s:.2f}s  "
        f"({tok_per_sec:.1f} tok/s  {ms_per_tok:.1f} ms/tok)  [{cache_label}]"
    )


if __name__ == "__main__":
    main()
