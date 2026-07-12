# Akshara — OCR Fine-Tune Plan (Stage 2 → 3)

Status as of this doc: **Stage 1 (language pretrain) is DONE.** The decoder
prior lives at `Saurab0/akshara-pretrain` on HF (step 5400, ~540M tokens, dev
ppl ≈ 6.7; strong Nepali, functional English — see the language-eval results).
This document covers turning that language prior into a crop **recognizer**.

**Two framing decisions that shape this plan:**
- **Learning-first.** The goal is to understand how OCR systems are built, not
  to beat PaddleOCR-VL / Nemotron-OCR. So we build the recognizer ourselves, and
  use those models as *baselines* (a reference CER to measure against) and
  optional *teachers* (pseudo-labeling), never as replacements.
- **Real-data-first.** Enough real labeled data now exists (ocr-mlt-50m, Mozhi-LR)
  that it carries the bulk of training. Synthetic drops to a *small supplement*
  (paragraph layouts + a run to understand the technique), which means the
  font/libraqm work only has to hold at small scale — not as a 300k-crop factory.

---

## 1. What we're building

Recall the pivot (see `docs/ARCHITECTURE.md`): we do **not** OCR a full page in
one shot. Structure is found by pretrained Surya layout/table models; *reading*
is the only trained part.

```
Page image
  └─ Surya layout + table structure (pretrained)   → region / cell crops
        └─ Akshara recognizer (WE TRAIN THIS)        crop image → plain text
              └─ HTML assembler (plain Python)        region class → <p>/<h1>/<table>
```

The recognizer is a vision-language model:

```
crop (448×448, aspect-preserving pad)
  → DINOv2-S/14 encoder (pretrained, frozen at first)   1024 patch tokens, dim 384
  → Connector (MLP 384→768 + RMSNorm)                   NEW, random init
  → HybridDecoder (GDN:attn 3:1, 16 layers)             ← LOAD Stage-1 prior
  → plain text (running text, no layout tokens)
```

Only the **Connector** is fully new. The **decoder** starts from our pretrained
prior; the **vision encoder** starts from DINOv2 weights. So Stage 2 mostly has
to teach the Connector to map visual features into the decoder's token space —
the decoder already knows what valid Nepali/English *looks like*.

---

## 2. Curriculum (the reason Stage 1 existed)

Pix2Struct's result: a reading warmup before full training matters. Our knob is
`--max_lines` in `src/data/synth_data.py`.

| Stage | Data | `max_lines` | Vision encoder | Goal |
|---|---|---|---|---|
| **2A — line warmup** | synthetic single-line crops | 1 | **frozen** ~first 1k steps, then unfrozen | glyph-level reading: conjuncts, matras, digits, Latin |
| **2B — paragraph** | synthetic multi-line crops | ~12 | unfrozen | multi-line reading, longer context |
| **3 — real fine-tune** | real printed Devanagari + real docs | n/a | unfrozen, low LR | close the synthetic→real gap |

**Why freeze vision first** (handled in `train_ocr.py`): a random-init Connector
sends garbage gradients into the pretrained DINOv2 and decoder. Freeze the
encoder for ~1k steps so the Connector aligns first, then unfreeze.

**Primary metric is CER (character error rate) on greedy decode**, not
perplexity — ppl rewards confident-but-wrong. Eval on a held-out synthetic set
every N steps; final gate is the **real** held-out set (heiDATA, never trained).

---

## 3. Data preparation (the real work of this phase)

### 3.1 Real data (primary Stage 2 source)
| Dataset | Use | Note |
|---|---|---|
| interfaze-ai/ocr-mlt-50m | **bulk 2A lines** | on HF; filter `lang ∈ {ne,hi,en}` → our JSONL. Start here. |
| Mozhi-LR (CVIT/IIIT) | real Nepali 2A/3 | request form; real printed word images |
| oscar-corpus/mOSCAR | teacher-pseudo-labeled pairs | image source only (not OCR-aligned) — see 3.3 |
| heiDATA printed Devanagari | **eval only** | 5,139 real lines — NEVER train on it |

First move: write the ocr-mlt-50m → JSONL converter (filter ne/hi/en) and
eyeball a sample. This is the low-friction bulk of Stage 2A.

### 3.2 Synthetic supplement (small — to understand the technique + paragraph gap)
`src/data/synth_data.py` renders (crop, text) pairs from the Stage-1 corpus.
Kept **small** now that real data carries the bulk:

```bash
PYTHONPATH=. python src/data/synth_data.py \
    --corpus data/corpus_v4/train.jsonl --fonts fonts/ \
    --out data/crops/lines --n 20000 --max_lines 1
```

- **Font open-item (small scale now):** mixed-script crops need fonts covering
  BOTH Devanagari and Latin (or per-script fallback), else English/numbers
  render as tofu and corrupt labels. Only has to hold for the small supplement.
- **libraqm is mandatory** (correct matra/conjunct shaping) — verify
  `features.check("raqm") == True` on the box.

### 3.3 Baseline + teacher (existing models)
Run **PaddleOCR-VL** (or Nemotron-OCR) on the eval crops for a reference CER —
this tells us whether our own recognizer is working, and whether the model is
good enough at Nepali to **pseudo-label mOSCAR** images into real training
pairs (`mOSCAR image → Surya crop → teacher transcription`). If the teacher is
weak on Nepali, skip the mOSCAR path and lean on ocr-mlt-50m + Mozhi.

### 3.3 Format
`CropOCRDataset` expects JSONL `{"image": "path.png", "text": "..."}` (dir mode
also supports `.txt`/`.html` siblings). Targets pad with `-100` after the first
EOS (already implemented).

---

## 4. Training

Entry point `scripts/train_ocr.py` (already wired: DataParallel-safe loss,
`evaluate_cer`, vision-freeze warmup, `--pretrain_ckpt`, HF-style checkpointing).

```bash
# Stage 2A — load the Stage-1 decoder prior, freeze vision, warm up on lines
PYTHONPATH=. python scripts/train_ocr.py \
    --config configs/ocr_finetune.json \
    --pretrain_ckpt checkpoints/pretrain_a100_v4.pt \
    --train_path data/crops/lines/data.jsonl \
    --hf_repo Saurab0/akshara-ocr \
    --device cuda

# Stage 2B — continue from 2A on paragraph crops
PYTHONPATH=. python scripts/train_ocr.py \
    --config configs/ocr_finetune.json \
    --pretrain_ckpt checkpoints/ocr_2a.pt \
    --train_path data/crops/paras/data.jsonl \
    --hf_repo Saurab0/akshara-ocr --device cuda

# Stage 3 — real data, low LR
PYTHONPATH=. python scripts/train_ocr.py \
    --config configs/ocr_finetune.json \
    --pretrain_ckpt checkpoints/ocr_2b.pt \
    --train_path data/crops/real/data.jsonl \
    --lr 3e-5 --hf_repo Saurab0/akshara-ocr --device cuda
```

**Carry over the lessons from Stage 1:**
- **HF backup from step 0** — `--hf_repo Saurab0/akshara-ocr` + `huggingface-cli
  login` on the box. Lightning wipes the disk on session end; only HF survives.
- **Reinstall `flash-linear-attention==0.5.1` every session** (conda env is per
  machine, not persistent) — the decoder's GDN weights need FLA to load.
- **Set the HF repo namespace exactly `Saurab0`** (case-sensitive) with a
  **write**-scoped token.
- **Fit the free A100 allowance** — measure tok/s early, trim step count for a
  clean cosine landing rather than a mid-schedule kill.
- Generate the corpus/crops **on the training box before the run** (survives
  within a session; regenerate each new machine).

---

## 5. Open items / risks

1. **Font coverage (small-scale open item, no longer a blocker)** — mixed-script
   synthetic crops need bilingual fonts, else labels corrupt. Only matters for
   the small synthetic supplement now that real data carries the bulk.
2. **English is the weak side of the prior** — if English-doc OCR lags, add
   English to Stage 3 (data-driven, not preemptive).
3. **FLA → portable inference** — the trained model needs FLA (CUDA) to run.
   For CPU/Mac inference and the Surya pipeline, we'll need a validated
   FLA-GDN → pure-Python-GDN weight converter. Build this at the inference/
   deployment stage, verify outputs match before trusting it.
4. **`use_short_conv=False`** in the GDN layer (FLA warns it hurts quality) —
   a candidate improvement for any future from-scratch retrain, not this phase.
5. **Surya integration** (`src/pipeline.py`) — wire layout/table crops into the
   recognizer once the recognizer clears CER on the real eval set.

---

## 6. Order of work

1. **ocr-mlt-50m → JSONL converter** (filter ne/hi/en) + eyeball a sample —
   real training data flowing, unblocks Stage 2A. Start here.
2. **PaddleOCR-VL baseline** CER on eval crops — reference number, and decides
   whether the mOSCAR pseudo-label path is open.
3. Request **Mozhi-LR**; download **heiDATA** (held-out eval).
4. **Stage 2A run** on real lines → compare CER to the baseline.
5. **Small synthetic supplement** (fonts only at small scale) + paragraph crops
   → **Stage 2B run**.
6. **Stage 3** on real corrected data → gate on **heiDATA** CER.
7. *(optional)* mOSCAR pseudo-labeling if the teacher proved good at Nepali.
8. Wire the **Surya pipeline**; end-to-end page → HTML.
