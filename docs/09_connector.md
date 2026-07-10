# 09 · Connector

A 2-layer MLP that bridges the vision encoder (384 dim) to the decoder (768 dim).

**File:** [`src/models/connector.py`](../src/models/connector.py)

---

## Why is this needed?

ViT-S outputs 384-dim vectors. The HybridDecoder expects 768-dim inputs. The
connector maps between them.

```
Vision encoder out:   (B, 196, 384)
         ↓  Linear(384 → 768) + GeLU + Linear(768 → 768)
Decoder visual prefix: (B, 196, 768)
```

---

## Why a 2-layer MLP and not just one linear layer?

A single linear layer can only learn affine transformations of the visual
features. A 2-layer MLP with a non-linearity can learn a non-linear mapping
— translating "visual feature space" language into "decoder language space."

Think of it like a translator who doesn't just replace words one-for-one, but
understands the meaning and rephrases it in the target language's idioms.

---

## Why GeLU here, not SwiGLU?

SwiGLU uses 3 weight matrices (gate, hidden, out) and is ~1.5× larger for
the same hidden dim. For a small bridge of 0.9M params, the extra complexity
has no meaningful benefit. GeLU with 2 matrices is simpler and sufficient.

---

## The code

```python
self.net = nn.Sequential(
    nn.Linear(vision_dim, hidden_dim, bias=True),   # 384 → 768
    nn.GELU(),
    nn.Linear(hidden_dim, decoder_dim, bias=True),  # 768 → 768
)
```

`hidden_dim = vision_dim * 2 = 768` — standard 2× expansion ratio.

Note: the connector uses `bias=True` (unlike most of our model which uses
`bias=False`). This helps the connector shift and scale visual features
without needing to learn it implicitly through the weights.

---

## Self-check results

```
Connector self-check
  params       : 886,272  (0.89M)
  input shape  : (2, 196, 384)
  output shape : (2, 196, 768)  ✅
  dim bridge   : 384 → 768
```

---

## During training

**Stage 1 (language pretraining):** Connector is frozen (along with encoder).
Only the decoder trains. Visual features don't need translating yet — the
decoder is just learning Nepali.

**Stage 2 (OCR fine-tuning):** All three components train together. The
connector learns the specific translation needed for OCR feature maps.

```python
model.freeze_encoder()    # freezes both encoder and connector
# ... train decoder on text corpus ...

model.unfreeze_all()
# ... train everything end-to-end on (image, text) pairs ...
```

---

**Next:** [10 · Full VLM](10_vlm.md)
