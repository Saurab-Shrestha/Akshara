# Akshara — Component Walkthroughs (learning notes)

A step-by-step, build-from-scratch tutorial for the model's components.

> **⚠️ This tutorial series predates two big changes.** The canonical, current
> design is **[ARCHITECTURE.md](ARCHITECTURE.md)** (+ **[OCR_FINETUNE_PLAN.md](OCR_FINETUNE_PLAN.md)**) — read those first.
> Two shifts happened after these notes were written:
> 1. **Vision encoder** is now **pretrained DINOv2-S/14 at 448px** (1024 patches), not a from-scratch ViT-S/16.
> 2. **Full-page OCR → crop recognition**: Surya finds structure, we only read crops. So `11_detection.md` is **superseded**.
>
> The component notes 03–07, 09, 10 are still accurate; 08 (vision) and 11 (detection) are stale.

---

## What we're building

A **~308M parameter hybrid VLM** that reads Devanagari + English document
*crops* and outputs Unicode text:

```
Crop image (448×448)
      │
      ▼
 ┌─────────────────────┐
 │  Vision Encoder     │  DINOv2-S/14 (pretrained) — 14×14 patches at 448px
 │  (~22M params)      │  → 32×32 = 1024 tokens, dim 384
 └──────────┬──────────┘
            │ patch tokens (1024 × 384)
            ▼
 ┌─────────────────────┐
 │  Connector          │  2-layer MLP + RMSNorm — 384 → 768
 │  (~0.9M params)     │
 └──────────┬──────────┘
            │ visual prefix
            ▼
 ┌─────────────────────┐
 │  Hybrid Decoder     │  16 layers, 3:1 GDN + Attention (FLA on GPU),
 │  (~308M params)     │  generates Devanagari/Latin text autoregressively
 └──────────┬──────────┘
            │
            ▼
    "नेपाल राम्रो देश हो"
```

**Status:** Stage 1 language pretrain ✅ done — prior on HF (`Saurab0/akshara-pretrain`).

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
| ~~11 · Detection~~ | **superseded** — full-page detection was replaced by Surya + crop recognition (see [ARCHITECTURE.md](ARCHITECTURE.md)) | — |

Current design, curriculum, data strategy, and training/infra all live in
**[ARCHITECTURE.md](ARCHITECTURE.md)** and **[OCR_FINETUNE_PLAN.md](OCR_FINETUNE_PLAN.md)**.

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

## Hardware

Stage 1 trained on a **Lightning.ai A100** (bf16 + FLA kernel, ~20k tok/s). See
[ARCHITECTURE.md §6](ARCHITECTURE.md) for the ephemeral-machine playbook (HF
backup, FLA reinstall, corpus regen, allowance-fit) — the practical details that
actually keep a cloud run from being lost.
