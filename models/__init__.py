"""Model definitions for AttnRes experiments."""

from models.components import RMSNorm, SwiGLU, RotaryEmbedding, CausalSelfAttention
from models.attn_res import FullAttnResOp, BlockAttnResOp, AttnResTransformerLayer
from models.transformer import AttnResTransformer, BaselineTransformer, PatchEmbedding
from models.lm_transformer import AttnResLM, BaselineLM
from utils.config import ModelConfig


def build_model(
    cfg, num_classes=10, img_size=28, patch_size=4, in_channels=1, baseline=False
):
    """Factory: returns AttnResTransformer or BaselineTransformer (vision)."""
    cls = BaselineTransformer if baseline else AttnResTransformer
    return cls(
        cfg=cfg,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
    )


def build_lm(cfg, vocab_size: int, seq_len: int = 256, baseline: bool = False):
    """Factory: returns AttnResLM or BaselineLM (language model).

    Args:
        cfg: ModelConfig instance.
        vocab_size: Tokeniser vocabulary size.
        seq_len: Maximum context window length.
        baseline: If True, return BaselineLM (standard residuals).

    Returns:
        Initialised language model nn.Module.
    """
    cls = BaselineLM if baseline else AttnResLM
    return cls(cfg=cfg, vocab_size=vocab_size, seq_len=seq_len)


__all__ = [
    "RMSNorm",
    "SwiGLU",
    "RotaryEmbedding",
    "CausalSelfAttention",
    "FullAttnResOp",
    "BlockAttnResOp",
    "AttnResTransformerLayer",
    "AttnResTransformer",
    "BaselineTransformer",
    "PatchEmbedding",
    "AttnResLM",
    "BaselineLM",
    "build_model",
    "build_lm",
]
