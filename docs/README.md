# Nepali OCR VLM — Documentation

A step-by-step guide to building a Surya v2-style Vision Language Model for
Nepali (Devanagari) + English OCR from scratch, designed to run on Kaggle free GPUs.

---

## What we're building

A **~270M parameter hybrid VLM** that reads Nepali and English document images
and outputs the text as Unicode. The architecture mirrors Surya v2:

```
Document Image
      │
      ▼
 ┌─────────────────────┐
 │  Vision Encoder     │  ViT-S/16 — splits image into 16×16 patches,
 │  (~22M params)      │  encodes each patch into a 768-dim vector
 └──────────┬──────────┘
            │ patch tokens (N × 768)
            ▼
 ┌─────────────────────┐
 │  Connector          │  2-layer MLP — bridges encoder and decoder
 │  (~5M params)       │  dimensions
 └──────────┬──────────┘
            │ visual prefix
            ▼
 ┌─────────────────────┐
 │  Hybrid Decoder     │  3:1 GDN + Attention — reads visual tokens,
 │  (~270M params)     │  generates Nepali/English text autoregressively
 └──────────┬──────────┘
            │
            ▼
    "नेपाल राम्रो देश हो"
```

---

## Architecture: 3:1 Gated DeltaNet Hybrid

The decoder is not a standard transformer. It alternates between two layer types:

```
Layer  0: GDNBlock          ← fast, O(T), recurrent memory
Layer  1: GDNBlock
Layer  2: GDNBlock
Layer  3: TransformerBlock  ← exact attention, O(T²), precise recall
Layer  4: GDNBlock
...repeats every 4 layers
```

This is the same pattern used in **Surya v2** (via Qwen3.5-style architecture),
**Qwen3-Next**, and **Kimi Linear**. 75% of layers are fast GDN (good for long
documents), 25% are exact attention (ensures verbatim character recall for OCR).

---

## Documentation map

### Foundations (read these first)
| Doc | Covers |
|---|---|
| [Tokenization](foundations/tokenization.md) | BPE, Devanagari Unicode, why we use Qwen3.5 tokenizer |
| [Attention](foundations/attention.md) | Self-attention, causal masking, multi-head, GQA |
| [Transformer](foundations/transformer.md) | Residual connections, pre-norm, training loop |
| [Gated DeltaNet](foundations/gdn.md) | Linear attention, delta rule, gating — the key innovation |

### Build stages
| Doc | What we build | Files |
|---|---|---|
| [01 · Setup](01_setup.md) | Project structure, venv, dependencies | `requirements.txt`, `tokenizer/verify.py` |
| [02 · Tokenizer](02_tokenizer.md) | Qwen3.5 tokenizer verification for Nepali | `tokenizer/verify.py` |
| [03 · Building Blocks](03_building_blocks.md) | RMSNorm, RoPE, SwiGLU | `src/models/rms_norm.py`, `rope.py`, `swiglu.py` |
| [04 · Attention](04_attention.md) | Multi-head attention + GQA + causal mask | `src/models/attention.py` |
| [05 · Transformer Block](05_transformer_block.md) | One full standard layer | `src/models/transformer_block.py` |
| [06 · GDN Block](06_gdn_block.md) | Gated DeltaNet recurrent layer | `src/models/gdn_block.py` |
| [07 · Hybrid Decoder](07_hybrid_decoder.md) | 3:1 hybrid — the full text model | `src/models/hybrid_decoder.py` |
| [08 · Vision Encoder](08_vision_encoder.md) | ViT-S/16 patch encoding | `src/models/vit.py` |
| [09 · Connector](09_connector.md) | Vision→language bridge | `src/models/connector.py` |
| [10 · Full VLM](10_vlm.md) | Assembled end-to-end model | `src/models/vlm.py` |
| 11 · Synthetic Data *(coming)* | SynthTIGER Nepali text rendering | `src/data/` |
| 12 · Training *(coming)* | Pretraining + OCR fine-tuning on Kaggle | `scripts/` |
| 13 · Inference *(coming)* | Running OCR on a real Nepali document | `scripts/generate.py` |

---

## Quick start

```bash
git clone <this-repo>
cd nepali-ocr-vlm
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Verify the tokenizer handles Nepali correctly
PYTHONPATH=. python tokenizer/verify.py

# Run all component self-checks
PYTHONPATH=. python src/models/rms_norm.py
PYTHONPATH=. python src/models/rope.py
PYTHONPATH=. python src/models/swiglu.py
PYTHONPATH=. python src/models/attention.py
PYTHONPATH=. python src/models/transformer_block.py
PYTHONPATH=. python src/models/gdn_block.py
PYTHONPATH=. python src/models/hybrid_decoder.py
```

---

## Project structure

```
nepali-ocr-vlm/
├── docs/                    ← you are here
│   ├── foundations/         ← conceptual background
│   └── diagrams/            ← mermaid architecture diagrams
├── src/
│   ├── models/
│   │   ├── rms_norm.py      ← normalization (faster than LayerNorm)
│   │   ├── rope.py          ← rotary position encoding
│   │   ├── swiglu.py        ← gated feed-forward MLP
│   │   ├── attention.py     ← multi-head attention with GQA
│   │   ├── transformer_block.py ← standard transformer layer
│   │   ├── gdn_block.py     ← Gated DeltaNet recurrent layer
│   │   ├── decoder.py       ← pure transformer decoder (reference)
│   │   └── hybrid_decoder.py ← 3:1 hybrid decoder (main model)
│   └── data/                ← dataset classes (coming)
├── tokenizer/
│   └── verify.py            ← Devanagari tokenizer verification
├── scripts/                 ← training + inference scripts (coming)
├── data/
│   ├── corpus/              ← Nepali text for pretraining
│   └── synthetic/           ← rendered (image, text) pairs
└── requirements.txt
```

---

## Hardware target

Designed to train on **Kaggle free tier** (T4 GPU, 16GB VRAM, 30h/week):
- Enable gradient checkpointing: `model.set_gradient_checkpointing(True)`
- Use gradient accumulation to simulate larger batches
- The GDN layers reduce memory at inference vs pure attention

For production training: 4× A100 80GB, ~$10–15k cloud cost for the full pipeline.
