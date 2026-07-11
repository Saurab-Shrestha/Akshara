"""
Vision Encoder — ViT-S/16
==========================

WHY THIS EXISTS
---------------
The decoder knows text. But it can't see images. The vision encoder converts
a document image into a sequence of patch tokens — the same shape as text
tokens — so the decoder can read image content like it reads text.

HOW IT WORKS (ViT — Vision Transformer, Dosovitskiy et al. 2020)
-----------------------------------------------------------------
Step 1: PATCH EMBEDDING
  - Input image: (batch, 3, H, W) — 3 RGB channels
  - Cut into non-overlapping 16×16 patches
  - For 224×224 image: 14×14 = 196 patches
  - Each patch: 16×16×3 = 768 raw values
  - Project each patch to embed_dim (384 for ViT-S) via a linear layer
  - This is equivalent to a Conv2d with kernel=16, stride=16

Step 2: CLS TOKEN
  - Prepend a learned [CLS] token to the sequence
  - Final sequence: [CLS, patch_1, patch_2, ..., patch_196] = 197 tokens
  - [CLS] aggregates global image information (useful for classification)
  - For OCR, we use ALL patch tokens (not just CLS), so the decoder sees
    every spatial region of the image

Step 3: POSITION EMBEDDING
  - Add a learned 2D position embedding to each patch token
  - Unlike text where RoPE works, images have 2D spatial structure
  - Learned embeddings (not RoPE) are standard for ViTs — the model
    learns that patch (row=3, col=5) is spatially near (row=3, col=6)

Step 4: TRANSFORMER BLOCKS
  - Standard transformer blocks (attention + MLP)
  - We use our own TransformerBlock components
  - n_layers=12, n_heads=6, embed_dim=384 for ViT-S/16

Step 5: OUTPUT
  - Shape: (batch, n_patches, embed_dim) = (batch, 196, 384)
  - These 196 vectors are the "visual tokens"
  - The connector (Stage 09) maps them from 384 → 768 (decoder's dim)

WHY ViT-S/16 SPECIFICALLY?
---------------------------
  ViT-Ti/16:  192 dim, too small — expressiveness loss
  ViT-S/16:   384 dim, 22M params — fits within our Kaggle budget
  ViT-B/16:   768 dim, 86M params — matches decoder dim, but heavy for T4

With ViT-S, the connector (2-layer MLP, 384→768) does the dim matching.
The mismatch is intentional — the encoder learns a compact visual representation,
the connector learns to "translate" it into the decoder's language.

IMAGE SIZE FOR OCR
------------------
Standard ViT uses 224×224. For OCR of Nepali text, we may use larger images
(e.g. 448×448) to preserve small character details (matras, dots, conjuncts).

At 448×448 with patch size 16: (448/16)² = 784 patches instead of 196.
The model handles this without retraining by interpolating position embeddings.

This file's DEFAULT_CONFIG uses 224×224 for speed. Swap to 448 for real OCR.

PARAMETER COUNT (ViT-S/16, 224×224)
--------------------------------------
  Patch embed (conv): 16×16×3 × 384       =    294,912
  CLS token:                               =        384
  Pos embed: 197 × 384                    =     75,648
  12 × TransformerBlock (~263k each)       =  3,158,016
  Final LayerNorm                          =        768
  Total                                    ≈   3.5M  (+ embeddings)
  Actually: standard ViT-S/16 = ~22M total — most in attention matrices
"""

import torch
import torch.nn as nn

from src.models.rms_norm import RMSNorm
from src.models.swiglu import SwiGLU


class PatchEmbed(nn.Module):
    """
    Convert image into a sequence of patch embeddings.

    Uses Conv2d with kernel_size=patch_size, stride=patch_size — each
    convolution window corresponds to exactly one patch. Equivalent to
    flattening each patch and running a linear projection, but faster.
    """

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        """
        Args:
            img_size:    image height == width (e.g. 224)
            patch_size:  patch height == width (e.g. 16)
            in_channels: 3 for RGB
            embed_dim:   output embedding dimension per patch
        """
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size {img_size} must be divisible by patch_size {patch_size}"

        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size

        # One conv that extracts and projects all patches in one pass
        # kernel=patch_size, stride=patch_size → no overlap → exact patches
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, img_size, img_size)
        Returns:
            (batch, n_patches, embed_dim)
        """
        # Conv2d: (B, C, H, W) → (B, embed_dim, H/p, W/p)
        x = self.proj(x)
        # Flatten spatial dims: (B, embed_dim, H/p, W/p) → (B, embed_dim, n_patches)
        x = x.flatten(2)
        # Rearrange: (B, embed_dim, n_patches) → (B, n_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class ViTBlock(nn.Module):
    """
    One ViT transformer block.

    Same pre-norm + residual structure as our text TransformerBlock.
    Uses standard multi-head attention (not GQA, not GDN) — the vision
    encoder processes all patches bidirectionally (no causal mask needed:
    patch 50 can see patch 100, images aren't sequential).
    """

    def __init__(self, embed_dim: int, n_heads: int):
        """
        Args:
            embed_dim: hidden dimension
            n_heads:   attention heads (must divide embed_dim evenly)
        """
        super().__init__()
        assert embed_dim % n_heads == 0

        self.norm1 = RMSNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True,   # expects (batch, seq, dim) — matches our convention
            bias=False,
        )

        self.norm2 = RMSNorm(embed_dim)
        self.mlp   = SwiGLU(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            (batch, seq_len, embed_dim)
        """
        # Attention sub-layer: bidirectional (no mask) — every patch sees all others
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out

        # MLP sub-layer
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """
    ViT-S/16 vision encoder.

    Takes a batch of images, returns a sequence of patch embeddings that
    the connector (Stage 09) will project into the decoder's embedding space.
    """

    def __init__(
        self,
        img_size:    int = 224,
        patch_size:  int = 16,
        in_channels: int = 3,
        embed_dim:   int = 384,   # ViT-S uses 384
        n_layers:    int = 12,
        n_heads:     int = 6,
    ):
        """
        Args:
            img_size:    square image dimension (224 or 448 for OCR)
            patch_size:  16×16 patches (ViT-16 standard)
            in_channels: 3 for RGB images
            embed_dim:   384 for ViT-S, 768 for ViT-B
            n_layers:    12 transformer blocks
            n_heads:     6 for ViT-S (head_dim = 384/6 = 64)
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Patch embedding: image → patch sequence
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.grid = img_size // patch_size  # patches per side (28 at 448/16)

        # Factorized 2D position embedding (Pix2Struct-style): a row table and
        # a column table instead of one flat table. pos(r, c) = row[r] + col[c].
        # Grid-shape-agnostic: a future variable-resolution input (patch budget
        # instead of fixed canvas) reuses the same tables for any H×W grid.
        self.row_embed = nn.Parameter(torch.zeros(self.grid, embed_dim))
        self.col_embed = nn.Parameter(torch.zeros(self.grid, embed_dim))

        # No CLS token: OCR reads all patch tokens; a global summary token
        # is dead weight here.

        # Transformer blocks — bidirectional attention (no causal mask)
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, n_heads)
            for _ in range(n_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch, 3, img_size, img_size) — normalized RGB images
                    (aspect-preserving resize + white padding done by the dataset)

        Returns:
            (batch, n_patches, embed_dim) — one token per patch, row-major order
        """
        # 1. Embed patches: (B, 3, H, W) → (B, n_patches, embed_dim)
        x = self.patch_embed(images)

        # 2. Add factorized 2D position: pos(r, c) = row[r] + col[c]
        # row-major patch order matches PatchEmbed's flatten
        g = self.grid
        pos = (self.row_embed.unsqueeze(1) + self.col_embed.unsqueeze(0))  # (g, g, dim)
        x = x + pos.reshape(1, g * g, -1)

        # 3. Transformer blocks (bidirectional)
        for block in self.blocks:
            x = block(x)

        return self.norm(x)  # (B, n_patches, embed_dim)

    def resize_grid(self, new_img_size: int, patch_size: int = 16):
        """
        Resize the row/col position tables for a different canvas size.
        Call BEFORE building the optimizer (replaces parameters).
        """
        new_grid = new_img_size // patch_size
        if new_grid == self.grid:
            return
        def _interp(table):
            t = table.t().unsqueeze(0)  # (1, dim, grid)
            t = torch.nn.functional.interpolate(t, size=new_grid, mode="linear", align_corners=False)
            return nn.Parameter(t.squeeze(0).t().contiguous())
        self.row_embed = _interp(self.row_embed.data)
        self.col_embed = _interp(self.col_embed.data)
        self.grid = new_grid
        self.patch_embed.n_patches = new_grid ** 2
        print(f"  Resized pos tables → {new_grid}×{new_grid} grid")


# ── default config (ViT-S/16 at 448) ──────────────────────────────────────────
DEFAULT_CONFIG = dict(
    img_size    = 448,
    patch_size  = 16,
    in_channels = 3,
    embed_dim   = 384,
    n_layers    = 12,
    n_heads     = 6,
)


# ── self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = VisionEncoder(**DEFAULT_CONFIG)

    total  = sum(p.numel() for p in model.parameters())
    print(f"VisionEncoder (ViT-S/16) self-check")
    print(f"  params       : {total/1e6:.1f}M")

    n_patches = model.patch_embed.n_patches
    print(f"  n_patches    : {n_patches}  ({model.grid}×{model.grid} grid)")

    B = 2
    images = torch.randn(B, 3, DEFAULT_CONFIG["img_size"], DEFAULT_CONFIG["img_size"])
    out    = model(images)

    expected_shape = (B, n_patches, DEFAULT_CONFIG["embed_dim"])
    assert out.shape == expected_shape, f"shape mismatch: {out.shape} != {expected_shape}"
    print(f"  input shape  : {tuple(images.shape)}")
    print(f"  output shape : {tuple(out.shape)}  ✅")

    print(f"\n  Testing grid resize (448→224)...")
    model.resize_grid(new_img_size=224)
    out_small = model(torch.randn(B, 3, 224, 224))
    assert out_small.shape == (B, 14 * 14, DEFAULT_CONFIG["embed_dim"])
    print(f"  224×224 output: {tuple(out_small.shape)}  ✅")
