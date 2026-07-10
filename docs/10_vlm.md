# 10 · Full VLM Assembly

Wiring the three components into a single end-to-end OCR model.

**File:** [`src/models/vlm.py`](../src/models/vlm.py)

---

## Architecture overview

```
Image (B, 3, 224, 224)
        │
        ▼
  VisionEncoder       21.6M params
  (ViT-S/16)
        │  (B, 196, 384)
        ▼
  Connector            0.9M params
  (2-layer MLP)
        │  (B, 196, 768)  ← visual prefix
        ▼
  HybridDecoder       273.0M params
  (3:1 GDN, 12 layers)
        │  text token IDs (B, T)
        ▼
  Logits (B, T, 248044)  → decoded Nepali/English text

Total unique params: 295.5M
```

---

## Forward pass

```python
def forward(self, images, token_ids, targets=None):
    # 1. Encode image into patch tokens
    patch_tokens  = self.encoder(images)        # (B, 196, 384)
    visual_prefix = self.connector(patch_tokens) # (B, 196, 768)

    # 2. Decoder reads visual tokens, then generates text
    logits, loss = self.decoder(
        token_ids     = token_ids,
        targets       = targets,
        vision_prefix = visual_prefix,
    )

    # 3. Return text logits only (slice off visual prefix positions)
    n_visual = visual_prefix.shape[1]
    logits   = logits[:, n_visual:, :]  # (B, T, vocab_size)

    return logits, loss
```

The decoder internally sees a sequence of length `196 + T` (visual tokens
prepended to text tokens). We slice back to `T` for output — callers only
care about text predictions.

---

## Generation

```python
@torch.no_grad()
def generate(self, images, bos_token_id, eos_token_id, max_new_tokens=256, temperature=0.1):
    # Encode image once
    patch_tokens  = self.encoder(images)
    visual_prefix = self.connector(patch_tokens)

    # Start with BOS token
    token_ids = torch.full((B, 1), bos_token_id, ...)

    for _ in range(max_new_tokens):
        logits, _ = self.decoder(token_ids, vision_prefix=visual_prefix)
        next_id   = sample(logits[:, -1, :], temperature)
        token_ids = torch.cat([token_ids, next_id], dim=1)

        if (next_id == eos_token_id).all():
            break

    return token_ids
```

**Key optimization:** the image is encoded **once** before the generation
loop. Each autoregressive step reuses `visual_prefix` — only the text tokens
extend. Without this, you'd re-run ViT-S for every generated token (wasteful).

**Temperature = 0.1** for OCR: near-greedy, strongly biased toward the
highest-probability token. OCR text should be verbatim — we don't want
creative variation.

---

## Two-stage training

### Stage 1: Language pretraining

```python
model.freeze_encoder()   # freeze VisionEncoder + Connector

# Training loop — text only, no images
for batch in nepali_corpus:
    ids, targets = tokenize(batch)
    _, loss = model.decoder(ids, targets)   # call decoder directly
    loss.backward()
    optimizer.step()
```

Why? A decoder that already understands Nepali grammar and character patterns
learns OCR 10× faster in Stage 2. The visual encoder hasn't been seen yet —
it can be frozen without loss.

### Stage 2: OCR fine-tuning

```python
model.unfreeze_all()
model.set_gradient_checkpointing(True)   # essential for T4 VRAM

for batch in ocr_dataset:
    images, texts = batch
    ids, targets  = tokenize(texts)
    _, loss = model(images, ids, targets)   # full forward pass
    loss.backward()
    optimizer.step()
```

All three components train together. The visual encoder learns OCR-specific
features (sharp stroke edges, conjunct detection). The connector learns the
translation. The decoder learns to use visual context for text prediction.

---

## Self-check results

```
Total unique params: 295.5M

Forward pass:
  images shape  : (2, 3, 224, 224)
  token_ids     : (2, 16)
  visual prefix : (B, 196, 768)
  logits shape  : (2, 16, 248044)  ✅
  loss          : 12.68

Freeze encoder:
  frozen params : 92/229 param groups  ✅
  after unfreeze: 0 frozen  ✅

Gradient checkpointing: enabled  ✅
```

---

## What comes next

The model is complete. What remains:
- **Stage 11** — synthetic data pipeline: render Nepali text as images for training
- **Stage 12** — training script: pretraining + OCR fine-tuning on Kaggle
- **Stage 13** — inference script: run OCR on a real Nepali document

---

**Next:** Stage 11 · Synthetic Data *(coming)*
