# 08 · Vision Encoder (ViT-S/16)

> **⚠️ Partly superseded.** `src/models/vit.py` now wraps **pretrained
> DINOv2-S/14 at 448px** (14×14 patches → 1024 tokens, dim 384), not a
> from-scratch ViT-S/16 at 224px/16px. The *concepts* below (patch embedding,
> the vision→language bridge) still apply; the specific arch/params don't. See
> **[ARCHITECTURE.md §2.2](ARCHITECTURE.md)** for the current encoder.

Converting a document image into a sequence of patch tokens.

**File:** [`src/models/vit.py`](../src/models/vit.py)

---

## The core idea

Text is a sequence of tokens. We make images a sequence of tokens too — by
cutting the image into a grid of non-overlapping patches and projecting each
patch into a vector.

```
224×224 image
      ↓  cut into 16×16 patches
14×14 grid = 196 patches
      ↓  flatten each patch (16×16×3 = 768 values)
      ↓  project to embed_dim=384 via Conv2d
196 × 384 patch vectors
      ↓  run through 12 transformer blocks
196 × 384 visual tokens
```

This is **ViT** (Vision Transformer, Dosovitskiy et al., 2020). The insight:
if image patches are "words", a transformer can reason over them the same way
it reasons over text.

---

## Why ViT-S specifically?

| Variant | Embed | Heads | Layers | Params |
|---|---|---|---|---|
| ViT-Ti/16 | 192 | 3 | 12 | 5M |
| **ViT-S/16** | **384** | **6** | **12** | **22M** |
| ViT-B/16 | 768 | 12 | 12 | 86M |

ViT-B's 768 dim matches our decoder exactly — no connector needed. But 86M
just for the encoder is expensive on T4. ViT-S at 22M + a 0.9M connector
bridge is more efficient: same final shape, 4× less encoder cost.

---

## PatchEmbed — the key component

```python
self.proj = nn.Conv2d(
    in_channels=3, out_channels=embed_dim,
    kernel_size=patch_size, stride=patch_size,
    bias=False,
)
```

A `Conv2d` with `kernel_size=stride=patch_size=16` extracts exactly one patch
per window, with no overlap. It's equivalent to flattening each patch and
running a linear layer, but faster because CUDA Conv2d is heavily optimized.

After conv: `(B, 384, 14, 14)` → flatten spatial → transpose → `(B, 196, 384)`

---

## CLS token

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
x = torch.cat([cls_token.expand(B, -1, -1), patches], dim=1)
# shape: (B, 197, 384)
```

A learned vector prepended to the patch sequence. In classification ViTs, the
final CLS output is fed to a classification head — it aggregates global info.

For OCR, we **drop CLS** and return all 196 patch tokens. The decoder's
attention naturally aggregates what it needs; we want spatial diversity, not
a global summary. Returning `x[:, 1:, :]` drops the CLS position.

---

## Position embedding

Unlike text (where RoPE handles position inside attention), images have 2D
spatial structure. A learned position table `(197, 384)` is added to all
tokens (CLS + patches) after patch embedding.

Why learned and not RoPE? RoPE is designed for 1D sequences with relative
distance. Image patches have 2D spatial relationships (patch at row 3, col 5
is adjacent to patch at row 3, col 6 AND row 4, col 5). Learned 2D position
embeddings capture this structure more naturally.

---

## Bidirectional attention (no causal mask)

Text decoders use causal masking: token at position t can't see position t+1.

Images have no temporal direction. Patch 50 (middle of the image) should be
able to see patch 100 (bottom of the image) — they might contain related
characters (e.g. a vowel sign above, a consonant below).

The `ViTBlock` uses `nn.MultiheadAttention` with no mask argument.

---

## Position interpolation for different image sizes

Trained at 224×224 (196 patches), want to inference at 448×448 (784 patches)?

```python
model.interpolate_pos_embed(new_img_size=448)
```

This bicubically resizes the learned position grid from 14×14 to 28×28.
The model can then process larger images without retraining — just the
position embeddings change shape.

This is how production OCR systems handle variable document sizes.

---

## Self-check results

```
VisionEncoder (ViT-S/16) self-check
  params       : 21.6M
  n_patches    : 196  (14×14 grid)
  input shape  : (2, 3, 224, 224)
  output shape : (2, 196, 384)  ✅
  CLS dropped  : ✅  (196 patch tokens, not 197)

  Testing position interpolation (224→448)...
  Interpolated pos_embed: 14² → 28² patches
  448×448 output: (2, 784, 384)  ✅
```

---

**Next:** [09 · Connector](09_connector.md)
