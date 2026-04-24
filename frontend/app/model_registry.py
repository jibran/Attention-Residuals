"""Model registry: discovers checkpoint files and manages loaded models.

The registry scans ``settings.checkpoints_dir`` (recursively) for ``*.pt``
files on startup and whenever ``/api/models`` is called.  Models are loaded
lazily on the first generation request and kept in memory for subsequent
requests.  Only one model is kept loaded at a time on CPU-only deployments to
limit memory usage; on GPU deployments all discovered models are cached.

Thread safety
-------------
Loading is protected by an ``asyncio.Lock`` so that concurrent requests for
the same model do not trigger duplicate loads.

Usage::

    from app.model_registry import registry

    models = registry.list_models()
    result = await registry.generate(
        model_id="best",
        prompt="Once upon a time",
        max_new_tokens=200,
        temperature=0.8,
        top_k=40,
        use_kv_cache=True,
    )
"""

from __future__ import annotations

import asyncio
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.config import settings

# ---------------------------------------------------------------------------
# Ensure the attnres source tree is importable
# ---------------------------------------------------------------------------

_attnres_src = settings.attnres_src_dir.resolve()
if str(_attnres_src) not in sys.path:
    sys.path.insert(0, str(_attnres_src))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """Metadata about a discovered checkpoint file.

    Attributes:
        model_id: Short identifier derived from the checkpoint filename (stem).
        path: Absolute path to the ``.pt`` file.
        name: Display name shown in the UI (same as ``model_id`` by default).
        val_loss: Validation loss recorded at checkpoint time, or NaN.
        val_ppl: Validation perplexity (``exp(val_loss)``).
        epoch: Training epoch at which the checkpoint was saved.
        params: Total trainable parameter count, or 0 if not yet computed.
        architecture: Human-readable architecture string, e.g.
            ``"AttnResLM (Block, XSA)"``.
        dataset: Dataset name from the saved config.
    """

    model_id: str
    path: Path
    name: str = ""
    val_loss: float = float("nan")
    val_ppl: float = float("nan")
    epoch: int = 0
    params: int = 0
    architecture: str = ""
    dataset: str = ""

    def __post_init__(self) -> None:
        """Set ``name`` to ``model_id`` when not provided."""
        if not self.name:
            self.name = self.model_id

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary for the REST API.

        Returns:
            Plain dictionary with string keys and JSON-serialisable values.
        """
        return {
            "model_id": self.model_id,
            "name": self.name,
            "val_loss": (
                round(self.val_loss, 4) if not math.isnan(self.val_loss) else None
            ),
            "val_ppl": round(self.val_ppl, 2) if not math.isnan(self.val_ppl) else None,
            "epoch": self.epoch,
            "params": self.params,
            "params_fmt": _fmt_params(self.params),
            "architecture": self.architecture,
            "dataset": self.dataset,
        }


@dataclass
class GenerationResult:
    """Output of a single generation call.

    Attributes:
        prompt: The original input prompt string.
        generated: The full text (prompt + continuation).
        new_tokens: Number of tokens generated (excluding the prompt).
        elapsed_s: Wall-clock seconds for the generation call.
        tok_per_sec: Tokens per second throughput.
        ms_per_tok: Mean milliseconds per generated token.
        model_id: ID of the model that produced this result.
        use_kv_cache: Whether KV caching was active.
    """

    prompt: str
    generated: str
    new_tokens: int
    elapsed_s: float
    tok_per_sec: float
    ms_per_tok: float
    model_id: str
    use_kv_cache: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fmt_params(n: int) -> str:
    """Format a parameter count as a human-readable string.

    Args:
        n: Raw parameter count.

    Returns:
        String like ``"125.0 M"`` or ``"512 K"``.
    """
    if n == 0:
        return "unknown"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} M"
    return f"{n / 1_000:.0f} K"


def _architecture_label(ckpt_config: dict) -> str:
    """Build a short architecture description from the saved model config.

    Args:
        ckpt_config: The ``"model"`` sub-dict from the checkpoint ``"config"``
            key.

    Returns:
        Human-readable string, e.g. ``"AttnResLM (Block, XSA)"``.
    """
    name = ckpt_config.get("name", "")
    parts: list[str] = []

    if "Baseline" in name or "baseline" in name:
        return "BaselineLM (fixed residuals)"

    if ckpt_config.get("use_block_attn_res", True):
        parts.append("Block AttnRes")
    else:
        parts.append("Full AttnRes")

    if ckpt_config.get("use_xsa", False):
        parts.append("XSA")

    return f"AttnResLM ({', '.join(parts)})"


def _probe_checkpoint(path: Path) -> ModelInfo:
    """Read metadata from a checkpoint file without loading the model weights.

    Args:
        path: Path to the ``.pt`` checkpoint file.

    Returns:
        :class:`ModelInfo` populated from the checkpoint's metadata fields.
    """
    model_id = path.stem
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return ModelInfo(model_id=model_id, path=path)

    cfg_dict = ckpt.get("config", {})
    model_cfg = cfg_dict.get("model", {})
    data_cfg = cfg_dict.get("data", {})

    val_loss = float(ckpt.get("val_loss", float("nan")))
    val_ppl = math.exp(min(val_loss, 20)) if not math.isnan(val_loss) else float("nan")

    return ModelInfo(
        model_id=model_id,
        path=path,
        val_loss=val_loss,
        val_ppl=val_ppl,
        epoch=int(ckpt.get("epoch", 0)),
        architecture=_architecture_label(model_cfg),
        dataset=data_cfg.get("dataset", ""),
    )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Discovers, caches, and serves AttnRes language model checkpoints.

    Args:
        checkpoints_dir: Root directory scanned recursively for ``*.pt`` files.
        device_str: PyTorch device string (``"auto"``, ``"cpu"``, ``"cuda"``).
    """

    def __init__(self, checkpoints_dir: Path, device_str: str = "auto") -> None:
        self._ckpt_dir = checkpoints_dir
        self._device = self._resolve_device(device_str)
        self._infos: dict[str, ModelInfo] = {}
        self._loaded: dict[str, tuple[Any, Any, Any]] = (
            {}
        )  # model_id → (model, tokenizer, cfg)
        self._lock = asyncio.Lock()
        self._scan()

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """Resolve an ``"auto"`` device string to a concrete device.

        Args:
            device_str: One of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``.

        Returns:
            :class:`torch.device` instance.
        """
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _scan(self) -> None:
        """Scan ``checkpoints_dir`` for ``*.pt`` files and populate ``_infos``.

        Existing entries are preserved; new files are probed for metadata.
        """
        if not self._ckpt_dir.exists():
            return
        for pt_file in sorted(self._ckpt_dir.rglob("*.pt")):
            model_id = pt_file.stem
            if model_id not in self._infos:
                self._infos[model_id] = _probe_checkpoint(pt_file)

    def list_models(self) -> list[dict[str, Any]]:
        """Return metadata for all discovered checkpoints.

        Rescans the directory on every call so newly added files are picked up
        without restarting the server.

        Returns:
            List of dictionaries suitable for JSON serialisation.
        """
        self._scan()
        return [info.to_dict() for info in self._infos.values()]

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    async def _ensure_loaded(self, model_id: str) -> tuple[Any, Any, Any]:
        """Load ``model_id`` into memory if not already cached.

        Protected by an asyncio lock to prevent duplicate loads from
        concurrent requests.

        Args:
            model_id: Identifier of the checkpoint to load.

        Returns:
            Tuple ``(model, tokenizer, cfg)`` where ``cfg`` is the
            reconstructed :class:`~utils.config.Config`.

        Raises:
            KeyError: If ``model_id`` is not a known checkpoint.
            FileNotFoundError: If the checkpoint file has been deleted.
        """
        async with self._lock:
            if model_id in self._loaded:
                return self._loaded[model_id]

            if model_id not in self._infos:
                self._scan()
            if model_id not in self._infos:
                raise KeyError(f"Unknown model: {model_id!r}")

            info = self._infos[model_id]
            model, tokenizer, cfg = await asyncio.get_event_loop().run_in_executor(
                None, self._load_sync, info
            )
            # Count parameters now that the model is instantiated
            info.params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self._loaded[model_id] = (model, tokenizer, cfg)
            return model, tokenizer, cfg

    def _load_sync(self, info: ModelInfo) -> tuple[Any, Any, Any]:
        """Synchronous model loading (runs in a thread pool executor).

        Args:
            info: Checkpoint metadata including the file path.

        Returns:
            Tuple ``(model, tokenizer, cfg)``.
        """
        from models.lm_transformer import AttnResLM, BaselineLM
        from utils.config import (
            Config,
            DataConfig,
            GenerationConfig,
            LoggingConfig,
            ModelConfig,
            TrainingConfig,
        )

        ckpt = torch.load(info.path, map_location=self._device, weights_only=False)
        raw = ckpt.get("config", {})

        cfg = Config(
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(**raw.get("data", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            generation=GenerationConfig(**raw.get("generation", {})),
        )

        # Tokeniser
        dataset = cfg.data.dataset.lower()
        if dataset == "shakespeare":
            from dataset.tokenizer import CharTokenizer

            vocab_path = Path(cfg.data.data_dir) / "processed" / "vocab.json"
            tokenizer = CharTokenizer.load(vocab_path)
            vocab_size = len(tokenizer)
        elif dataset == "tinystories":
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            vocab_size = tokenizer.vocab_size
        else:
            raise ValueError(f"Unknown dataset '{cfg.data.dataset}'")

        # Model
        model_name = raw.get("model", {}).get("name", "")
        is_baseline = "Baseline" in model_name or "baseline" in model_name
        ModelClass = BaselineLM if is_baseline else AttnResLM
        model = ModelClass(cfg.model, vocab_size=vocab_size, seq_len=cfg.data.seq_len)
        model.load_state_dict(ckpt["model_state"])
        model.eval().to(self._device)
        return model, tokenizer, cfg

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
        use_kv_cache: bool = False,
    ) -> GenerationResult:
        """Generate a text continuation for ``prompt``.

        Args:
            model_id: Checkpoint identifier.
            prompt: Input text to continue.
            max_new_tokens: Number of new tokens to generate.
            temperature: Softmax temperature (higher = more random).
            top_k: Top-k logit filter; 0 disables it.
            use_kv_cache: Whether to enable KV caching for faster decoding.

        Returns:
            :class:`GenerationResult` containing the generated text and
            timing statistics.

        Raises:
            KeyError: If ``model_id`` is unknown.
            ValueError: If ``prompt`` is empty.
        """
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        max_new_tokens = min(max_new_tokens, settings.max_new_tokens_limit)
        model, tokenizer, cfg = await self._ensure_loaded(model_id)

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self._generate_sync,
            model,
            tokenizer,
            cfg,
            model_id,
            prompt,
            max_new_tokens,
            temperature,
            top_k,
            use_kv_cache,
        )
        return result

    @staticmethod
    def _generate_sync(
        model: Any,
        tokenizer: Any,
        cfg: Any,
        model_id: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        use_kv_cache: bool,
    ) -> GenerationResult:
        """Synchronous generation (runs in a thread pool executor).

        Args:
            model: Loaded language model.
            tokenizer: Fitted tokeniser.
            cfg: Model configuration.
            model_id: Checkpoint identifier (for the result payload).
            prompt: Input text.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filter value; 0 = disabled.
            use_kv_cache: Enable KV caching.

        Returns:
            :class:`GenerationResult`.
        """
        device = next(model.parameters()).device

        # Encode prompt
        ids = tokenizer.encode(prompt)
        if isinstance(ids, list):
            prompt_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        else:
            prompt_ids = ids.to(device)
        prompt_len = prompt_ids.shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
                use_kv_cache=use_kv_cache,
            )
        elapsed = time.perf_counter() - t0

        new_tokens = out_ids.shape[1] - prompt_len
        generated = tokenizer.decode(out_ids[0].tolist())

        return GenerationResult(
            prompt=prompt,
            generated=generated,
            new_tokens=new_tokens,
            elapsed_s=round(elapsed, 3),
            tok_per_sec=round(new_tokens / max(elapsed, 1e-9), 1),
            ms_per_tok=round(elapsed * 1000 / max(new_tokens, 1), 1),
            model_id=model_id,
            use_kv_cache=use_kv_cache,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

registry = ModelRegistry(
    checkpoints_dir=settings.checkpoints_dir,
    device_str=settings.device,
)
