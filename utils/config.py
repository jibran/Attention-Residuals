"""Configuration loader for AttnRes experiments.

Loads a YAML file and exposes typed dataclasses for each config section.
Provides a single :func:`load_config` entry-point that merges a base file
with any CLI overrides supplied as ``key.subkey=value`` strings.

Example::

    cfg = load_config("config/base.yaml", overrides=["training.lr=1e-3"])
    print(cfg.model.dim)   # 256
    print(cfg.training.lr) # 0.001
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Hyper-parameters that define the model architecture.

    Attributes:
        name: Human-readable model name used in checkpoint filenames and logs.
        dim: Hidden / embedding dimension ``d``.
        depth: Total number of transformer layers (each layer = Attn + MLP).
        heads: Number of self-attention heads.
        head_dim: Dimension per attention head.
        mlp_multiplier: Hidden-dim multiplier for the SwiGLU MLP block.
        dropout: Dropout probability applied in attention and MLP.
        use_block_attn_res: If ``True`` use Block AttnRes; if ``False`` use
            Full AttnRes.
        block_size: Number of transformer layers per block (Block AttnRes only).
        norm_eps: Epsilon for RMSNorm.
        max_seq_len: Maximum sequence length (used for positional embeddings).
        use_kv_cache: If True, :class:`~models.components.CausalSelfAttention`
            accepts an optional :class:`~models.components.KVCache` during inference,
            avoiding re-computation of past keys and values.  Must be False
            during training.  Enable via model.use_kv_cache: true in YAML or
            --override model.use_kv_cache=true at the CLI.
    """

    name: str = "AttnResTransformer"
    dim: int = 256
    depth: int = 8
    heads: int = 4
    head_dim: int = 64
    mlp_multiplier: int = 4
    dropout: float = 0.1
    use_block_attn_res: bool = True
    block_size: int = 4
    norm_eps: float = 1e-6
    max_seq_len: int = 512
    use_kv_cache: bool = False


@dataclass
class TrainingConfig:
    """Training loop hyper-parameters.

    Attributes:
        epochs: Total training epochs.
        batch_size: Mini-batch size.
        lr: Peak learning rate (used after warm-up).
        weight_decay: AdamW weight decay coefficient.
        grad_clip: Gradient norm clipping threshold (0 = disabled).
        warmup_steps: Number of linear warm-up steps before cosine decay.
        log_every: Log a training line every this many steps.
        save_every: Save a checkpoint every this many epochs.
        seed: Global random seed for reproducibility.
        device: ``"auto"`` | ``"cpu"`` | ``"cuda"`` | ``"mps"``.
    """

    epochs: int = 20
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    warmup_steps: int = 200
    log_every: int = 50
    save_every: int = 1
    seed: int = 42
    device: str = "auto"


@dataclass
class DataConfig:
    """Dataset and data-loading parameters.

    Attributes:
        dataset: Dataset identifier — ``"mnist"`` | ``"cifar10"`` | ``"shakespeare"``.
        data_dir: Root directory where datasets are downloaded / cached.
        num_workers: DataLoader worker processes.
        pin_memory: Whether to use pinned (page-locked) memory in DataLoader.
        val_split: Fraction of the training set held out for validation.
        seq_len: Token-sequence length for language-model datasets.
        stride: Sliding-window stride for language-model datasets.
    """

    dataset: str = "mnist"
    data_dir: str = "data/"
    num_workers: int = 4
    pin_memory: bool = True
    val_split: float = 0.1
    seq_len: int = 256
    stride: int = 128


@dataclass
class GenerationConfig:
    """Text-generation sampling parameters (language-model tasks only).

    Attributes:
        max_new_tokens: Number of tokens to sample during generation.
        temperature: Softmax temperature; lower = more deterministic.
        top_k: Restrict sampling to the top-k logits (0 = disabled).
    """

    max_new_tokens: int = 500
    temperature: float = 0.8
    top_k: int = 40


@dataclass
class LoggingConfig:
    """Paths for logs and checkpoints.

    Attributes:
        log_dir: Directory where CSV training logs are written.
        checkpoint_dir: Root directory; ``best/`` and ``latest/``
            subdirectories are created automatically.
    """

    log_dir: str = "logs/"
    checkpoint_dir: str = "checkpoints/"


@dataclass
class Config:
    """Top-level experiment configuration.

    Attributes:
        model: Model architecture settings.
        training: Training loop settings.
        data: Dataset and loading settings.
        logging: Logging and checkpoint path settings.
        generation: Text-generation sampling settings (LM tasks only).
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return the full config as a plain nested dictionary.

        Returns:
            Nested ``dict`` mirroring the dataclass hierarchy.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dict_to_config(raw: dict[str, Any]) -> Config:
    """Instantiate a :class:`Config` from a raw nested dictionary.

    Args:
        raw: Nested dictionary parsed from YAML.

    Returns:
        Populated :class:`Config` instance.
    """
    return Config(
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        data=DataConfig(**raw.get("data", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
    )


def _apply_override(cfg_dict: dict[str, Any], dotkey: str, value: str) -> None:
    """Set a value inside a nested dict using a dot-separated key path.

    Performs best-effort type coercion: tries ``int``, ``float``, ``bool``
    (``"true"`` / ``"false"``), then falls back to ``str``.

    Args:
        cfg_dict: Mutable nested dictionary to update in-place.
        dotkey: Dot-separated key path, e.g. ``"training.lr"``.
        value: String representation of the new value.

    Raises:
        KeyError: If any intermediate key does not exist.
    """
    keys = dotkey.split(".")
    node = cfg_dict
    for k in keys[:-1]:
        node = node[k]
    leaf = keys[-1]

    # Type coercion
    if value.lower() == "true":
        node[leaf] = True
    elif value.lower() == "false":
        node[leaf] = False
    else:
        for cast in (int, float):
            try:
                node[leaf] = cast(value)
                return
            except ValueError:
                pass
        node[leaf] = value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config file and apply optional CLI overrides.

    Args:
        path: Path to the ``.yaml`` config file.
        overrides: List of ``"section.key=value"`` strings that overwrite
            values from the file, e.g. ``["training.lr=1e-3", "model.dim=512"]``.

    Returns:
        Fully populated :class:`Config` instance.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If an override string is malformed (missing ``=``).

    Example::

        cfg = load_config("config/base.yaml", overrides=["training.epochs=5"])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    raw = copy.deepcopy(raw)

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(
                f"Override '{override}' is malformed — expected 'section.key=value'."
            )
        dotkey, _, value = override.partition("=")
        _apply_override(raw, dotkey.strip(), value.strip())

    return _dict_to_config(raw)
