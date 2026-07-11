"""
Vision Encoder — DINOv2-S/14 (pretrained)
==========================================

WHY PRETRAINED INSTEAD OF FROM-SCRATCH
---------------------------------------
Every OCR system in our compute class initializes its encoder from pretrained
weights (Donut/Nougat: ImageNet Swin). Pix2Struct trained vision from scratch
but paid for it with 80M screenshots on 64 TPUs. On two T4s with synthetic
crops, a from-scratch encoder is the riskiest part of the whole system — it
must learn what edges, strokes, and paper textures look like from nothing.

DINOv2-S/14 (facebook/dinov2-small, 21M params) is self-supervised on 142M
images and has excellent dense features out of the box. The encoder starts
knowing vision; training only has to teach it *Devanagari*.

WHY A WRAPPER OVER transformers.Dinov2Model
--------------------------------------------
Reimplementing DINOv2 (LayerScale, exact LayerNorm placement, …) invites
weight-loading bugs. `transformers` is already a project dependency and its
Dinov2Model handles position-embedding interpolation for any input size
natively. This file stays ~100 lines.

GEOMETRY
--------
448×448 input, patch 14 → 32×32 = 1024 patch tokens, dim 384.
DINOv2 was pretrained at 518px (37×37 grid); `interpolate_pos_encoding=True`
resizes its position embeddings to our grid on the fly.

Decoder budget check: 1024 visual + 512 text = 1536 = 3 × max_seq_len(512),
exactly what the decoder's RoPE tables and causal mask are sized for.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """
    DINOv2-S/14 wrapper. Returns patch tokens only (CLS dropped — OCR reads
    spatial tokens, not a global summary).

    Args:
        img_size:   square canvas size (448 → 32×32 grid at patch 14)
        patch_size: must be 14 for DINOv2 (kept as arg for config plumbing)
        embed_dim:  384 for dinov2-small (checked against the loaded model)
        pretrained: load facebook/dinov2-small weights (True for training;
                    False builds the same architecture randomly initialized —
                    used when a checkpoint will overwrite the weights anyway)
    """

    MODEL_NAME = "facebook/dinov2-small"

    def __init__(
        self,
        img_size:    int = 448,
        patch_size:  int = 14,
        in_channels: int = 3,
        embed_dim:   int = 384,
        pretrained:  bool = True,
        # accepted-and-ignored (legacy config keys from the custom ViT)
        n_layers:    int = 12,
        n_heads:     int = 6,
    ):
        super().__init__()
        assert patch_size == 14, "DINOv2 uses patch 14; adjust img_size instead"
        assert img_size % patch_size == 0, f"{img_size} not divisible by {patch_size}"

        from transformers import Dinov2Config, Dinov2Model
        if pretrained:
            self.backbone = Dinov2Model.from_pretrained(self.MODEL_NAME)
        else:
            self.backbone = Dinov2Model(Dinov2Config.from_pretrained(self.MODEL_NAME))

        assert self.backbone.config.hidden_size == embed_dim, (
            f"config embed_dim {embed_dim} != dinov2-small hidden {self.backbone.config.hidden_size}"
        )
        self.embed_dim = embed_dim
        self.grid = img_size // patch_size

        # Same accessor the rest of the codebase uses (vlm.py, train scripts)
        class _PE:  # ponytail: minimal shim for .patch_embed.n_patches
            pass
        self.patch_embed = _PE()
        self.patch_embed.n_patches = self.grid ** 2

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, img_size, img_size) — ImageNet-normalized
                    (DINOv2 uses the same mean/std as our datasets)
        Returns:
            (B, n_patches, embed_dim) — patch tokens, CLS dropped
        """
        out = self.backbone(pixel_values=images, interpolate_pos_encoding=True)
        return out.last_hidden_state[:, 1:, :]  # drop CLS

    def set_gradient_checkpointing(self, enabled: bool):
        if enabled:
            self.backbone.gradient_checkpointing_enable()
        else:
            self.backbone.gradient_checkpointing_disable()


# ── self-check (downloads ~85MB on first run) ─────────────────────────────────
if __name__ == "__main__":
    model = VisionEncoder(img_size=448, pretrained=True)
    total = sum(p.numel() for p in model.parameters())
    print(f"VisionEncoder (DINOv2-S/14) self-check")
    print(f"  params    : {total/1e6:.1f}M")
    print(f"  n_patches : {model.patch_embed.n_patches}  ({model.grid}×{model.grid} grid)")

    x = torch.randn(2, 3, 448, 448)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1024, 384), f"shape: {out.shape}"
    print(f"  output    : {tuple(out.shape)}  ✅")
