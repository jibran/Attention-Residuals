"""Full Attention Residuals Transformer model.

Assembles :class:`~models.attn_res.AttnResTransformerLayer` layers into a
complete sequence classification model suitable for MNIST / CIFAR-10.

Architecture overview::

    Input pixels → patch embedding → [AttnResLayer × depth] → CLS pool → head

The image is flattened into a sequence of non-overlapping patches and linearly
projected into ``dim``-dimensional vectors (similar to ViT but without the
class token — we use mean-pooling instead for simplicity).

Two public model classes are provided:

* :class:`AttnResTransformer` — main model; uses Full or Block AttnRes.
* :class:`BaselineTransformer` — identical architecture but with standard
  residual connections; used as the comparison baseline.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.attn_res import AttnResTransformerLayer
from models.components import RMSNorm, CausalSelfAttention, SwiGLU
from utils.config import ModelConfig


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------


class PatchEmbedding(nn.Module):
    """Linearly project flattened image patches into ``dim``-d vectors.

    Args:
        img_size: Spatial side length of the (square) input image.
        patch_size: Spatial side length of each (square) patch.
        in_channels: Number of input channels (1 for MNIST, 3 for CIFAR-10).
        dim: Output embedding dimension.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        dim: int,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size."
        self.n_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patches and project them.

        Args:
            x: Image tensor ``(B, C, H, W)``.

        Returns:
            Patch sequence ``(B, n_patches, dim)``.
        """
        B, C, H, W = x.shape
        p = self.patch_size
        # Reshape into patches: (B, n_patches, patch_dim)
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, -1, C * p * p)
        return self.proj(x)


# ---------------------------------------------------------------------------
# AttnRes Transformer
# ---------------------------------------------------------------------------


class AttnResTransformer(nn.Module):
    """Vision transformer with Attention Residuals for image classification.

    Args:
        cfg: :class:`~utils.config.ModelConfig` instance.
        num_classes: Number of output classes.
        img_size: Input image spatial size (height == width assumed).
        patch_size: Patch size for :class:`PatchEmbedding`.
        in_channels: Input image channels.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        num_classes: int = 10,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_block = cfg.use_block_attn_res

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, cfg.dim)
        n_patches = self.patch_embed.n_patches

        # Learnable position embedding (added to patch embedding)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, cfg.dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                AttnResTransformerLayer(
                    dim=cfg.dim,
                    heads=cfg.heads,
                    head_dim=cfg.head_dim,
                    mlp_multiplier=cfg.mlp_multiplier,
                    dropout=cfg.dropout,
                    max_seq_len=n_patches,
                    layer_number=i + 1,
                    block_size=cfg.block_size,
                    use_block_attn_res=cfg.use_block_attn_res,
                    norm_eps=cfg.norm_eps,
                )
                for i in range(cfg.depth)
            ]
        )

        # Final norm + classification head
        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise linear layers with truncated normal; biases to zero."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full forward pass.

        Args:
            x: Image batch ``(B, C, H, W)``.

        Returns:
            Logits ``(B, num_classes)``.
        """
        # Patch embedding + positional encoding
        h = self.patch_embed(x) + self.pos_embed  # (B, T, d)

        if self.use_block:
            # Block AttnRes: block_reps starts with the embedding as "block 0"
            block_reps: list[torch.Tensor] = [h]
            partial_block = torch.zeros_like(h)
            for layer in self.layers:
                block_reps, partial_block = layer(block_reps, partial_block)
            # Final hidden state = sum of all block reps + remaining partial
            out = sum(block_reps[1:], partial_block)  # skip embedding block 0
        else:
            # Full AttnRes: accumulate all sublayer outputs
            layer_outputs: list[torch.Tensor] = [h]
            for layer in self.layers:
                layer_outputs = layer(layer_outputs)
            out = sum(layer_outputs[1:])  # sum all sublayer outs

        # Mean-pool over sequence, normalise, classify
        out = self.norm(out.mean(dim=1))  # (B, d)
        return self.head(out)  # (B, num_classes)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Baseline Transformer (standard residuals — for comparison)
# ---------------------------------------------------------------------------


class _BaselineLayer(nn.Module):
    """Standard pre-norm transformer layer with fixed unit residuals.

    Args:
        dim: Hidden dimension.
        heads: Number of attention heads.
        head_dim: Dimension per head.
        mlp_multiplier: SwiGLU inner-dim multiplier.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length.
        norm_eps: RMSNorm epsilon.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.mlp_norm = RMSNorm(dim, eps=norm_eps)
        self.attn = CausalSelfAttention(dim, heads, head_dim, max_seq_len, dropout)
        self.mlp = SwiGLU(dim, hidden_dim=dim * mlp_multiplier, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard residual update: ``h = h + attn(norm(h)); h = h + mlp(norm(h))``.

        Args:
            x: Hidden state ``(B, T, d)``.

        Returns:
            Updated hidden state ``(B, T, d)``.
        """
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class BaselineTransformer(nn.Module):
    """Baseline vision transformer with standard residual connections.

    Identical architecture to :class:`AttnResTransformer` but uses fixed unit
    residuals instead of learned attention over depth.  Used for ablation /
    comparison experiments.

    Args:
        cfg: :class:`~utils.config.ModelConfig` instance.
        num_classes: Number of output classes.
        img_size: Input image spatial size.
        patch_size: Patch size.
        in_channels: Input channels.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        num_classes: int = 10,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, cfg.dim)
        n_patches = self.patch_embed.n_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, cfg.dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList(
            [
                _BaselineLayer(
                    dim=cfg.dim,
                    heads=cfg.heads,
                    head_dim=cfg.head_dim,
                    mlp_multiplier=cfg.mlp_multiplier,
                    dropout=cfg.dropout,
                    max_seq_len=n_patches,
                    norm_eps=cfg.norm_eps,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run baseline transformer forward pass.

        Args:
            x: Image batch ``(B, C, H, W)``.

        Returns:
            Logits ``(B, num_classes)``.
        """
        h = self.patch_embed(x) + self.pos_embed
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h.mean(dim=1))
        return self.head(h)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
