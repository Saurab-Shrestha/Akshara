# Akshara (अक्षर)

**अक्षर** — Sanskrit for *letter* and *imperishable*. Devanagari + English document OCR: page image in, structured `<p>`, `<h1>`, `<table>` HTML out.

Pretrained layout/table-structure models (Surya) find the *structure*; a custom-trained crop recognizer does the *reading*. Structure is geometry and language-agnostic — reading the glyphs is the only part that needs training, so it's the only part we train.

> **This is a learning project** — the goal is to understand how OCR systems are built end-to-end, not to out-compete existing models (PaddleOCR-VL, Nemotron-OCR). We build the recognizer ourselves and use those models/datasets as baselines and teachers.

> **Status:** Stage 1 (language pretrain) ✅ **done** — decoder prior on HF at [`Saurab0/akshara-pretrain`](https://huggingface.co/Saurab0/akshara-pretrain) (~540M tokens, dev ppl ≈ 6.7; strong Nepali, functional English). Next: Stage 2 (crop recognizer).

**Full design doc: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** · **OCR plan: [docs/OCR_FINETUNE_PLAN.md](docs/OCR_FINETUNE_PLAN.md)**

---

## Architecture

```
Page image
    │
    ▼
Surya layout + table structure (pretrained)   — paragraphs, headings, table cells
    │
    ▼  region / cell crops (448×448, aspect-preserving pad)
    │
Akshara recognizer (trained by us):
    DINOv2-S/14 encoder    — pretrained, 1024 patches
    Connector (MLP+norm)   — 384 → 768
    Hybrid decoder         — 16 layers, GDN:attention 3:1, GQA, RoPE
    │
    ▼  plain text per crop (Devanagari / Latin)
    │
HTML assembler (plain Python)                 — region class → <p>/<h1>/<table>
```

The decoder alternates Gated DeltaNet (GDN) recurrence layers with full-attention "exact recall" layers at a 3:1 ratio. On GPU the GDN runs via the FLA Triton kernel; a pure-Python fallback exists for CPU/Mac.

**Recognizer size**: ~308M parameters (190M is the 248k-vocab embedding table)

---

## Training

Curriculum: learn the language, learn to read a line, learn to read a paragraph (the line→paragraph warmup follows [Pix2Struct](https://arxiv.org/abs/2210.03347)'s reading-curriculum result):

| Stage | What trains | Data | Goal | Status |
|---|---|---|---|---|
| 1 — Language pretrain | Decoder only | 965k-doc mix: 70% FineWeb-2 Devanagari (ne/hi/mr/sa) + 20% FineWeb-Edu English + 10% Wikipedia | Learn script + grammar prior | ✅ done (ppl ≈ 6.7) |
| 2A — Line warmup | Full model | Real lines (ocr-mlt-50m, Mozhi-LR) + small synthetic | Glyph-level reading (conjuncts, matras) | next |
| 2B — Paragraph OCR | Full model | Multi-line crops | Multi-line reading | |
| 3 — Real fine-tune | Full model | Real corrected crops | Domain adaptation (gate on heiDATA) | |

---

## Setup

```bash
git clone git@github.com:Saurab-Shrestha/Akshara.git
cd Akshara

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**For GPU training**, also install the FLA kernel and log into HF (checkpoint backup). On an ephemeral cloud box (Lightning/Kaggle) the conda env resets each session — reinstall FLA every time:

```bash
pip install flash-linear-attention==0.5.1   # GDN Triton kernel (CUDA only)
huggingface-cli login                        # write-scoped token → checkpoint backup
```

Download the Noto Devanagari font (required for synthetic data generation):

```bash
mkdir -p fonts
wget -q 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf' \
     -O fonts/NotoSansDevanagari-Regular.ttf
```

**⚠️ libraqm — required before generating synthetic data.** PIL needs libraqm to correctly position Devanagari matras and conjuncts. Without it, every training label is silently corrupt.

```bash
# macOS
brew install harfbuzz fribidi
pip install pillow --no-binary :all:

# Linux
sudo apt install libraqm-dev
pip install pillow --no-binary :all:

# Verify
python -c "from PIL import features; print(features.check('raqm'))"   # → True
```

---

## Data Preparation

**Stage 1 corpus** (multilingual mix, ~965k docs):

```bash
PYTHONPATH=. python scripts/prepare_data.py --max_samples 1000000 --out_dir data/corpus_v4
```

**Stage 2 data — real-first (see [docs/OCR_FINETUNE_PLAN.md](docs/OCR_FINETUNE_PLAN.md)).**
Real labeled data now carries the bulk; synthetic is a small supplement, so the
font/libraqm work only has to hold at small scale.

| Dataset | Use |
|---|---|
| [interfaze-ai/ocr-mlt-50m](https://huggingface.co/datasets/interfaze-ai/ocr-mlt-50m) | filter `ne`/`hi`/`en` → **primary bulk** Stage 2A (Apache 2.0, on HF) |
| [Mozhi-LR (CVIT/IIIT)](https://cvit.iiit.ac.in/images/ConferencePapers/2024/Printed-OCR-for-Extremely-Low-resource-Indic-Languages.pdf) | real **printed Nepali** words — Stage 2A/3 |
| [oscar-corpus/mOSCAR](https://huggingface.co/datasets/oscar-corpus/mOSCAR) | image source to pseudo-label with a teacher model (not OCR-aligned itself) |
| [heiDATA printed Devanagari](https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/EGOKEI) | 5,139 real lines — held-out **eval only** |

Small synthetic supplement (understand the technique / paragraph layouts):

```bash
PYTHONPATH=. python src/data/synth_data.py \
    --corpus data/corpus_v4/train.jsonl --fonts fonts/ \
    --out data/crops/lines --n 20000 --max_lines 1
```

---

## Training

**Stage 1 — Language pretraining** ✅ *done* (~9h on an A100). Checkpoint on HF at `Saurab0/akshara-pretrain`. To reproduce / resume:

```bash
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/pretrain.py \
    --config configs/a100_pretrain.json \
    --hf_repo Saurab0/akshara-pretrain   # backs up every checkpoint to HF
# resume:  --resume checkpoints/pretrain_a100_v4.pt
```

**Stage 2 — OCR fine-tune** (crop recognizer; see [OCR plan](docs/OCR_FINETUNE_PLAN.md)):

```bash
PYTHONPATH=. python scripts/train_ocr.py \
    --config configs/ocr_finetune.json \
    --pretrain_ckpt checkpoints/pretrain_a100_v4.pt \
    --train_path data/crops/lines/data.jsonl \
    --hf_repo Saurab0/akshara-ocr \
    --device cuda
```

**Verify Stage 1 language quality:**

```bash
PYTHONPATH=. python scripts/lm_eval.py --ckpt checkpoints/pretrain_a100_v4.pt --device cuda
```

**Smoke test** (runs in ~2 minutes on CPU, verifies the full pipeline):

```bash
PYTHONPATH=. python scripts/train_ocr.py --config configs/smoke/ocr_finetune.json --device cpu
```

---

## Inference

> **Note:** the trained model uses the FLA GDN kernel, which is **CUDA-only** — inference currently needs a GPU box. CPU/Mac inference (and off-GPU Surya) will need an FLA-GDN → pure-Python-GDN weight converter (planned for the inference stage).

```bash
PYTHONPATH=. python scripts/generate.py \
    --checkpoint checkpoints/ocr_nepali.pt \
    --image path/to/document.png
```

Output is structured HTML:

```html
<h1>नेपालको इतिहास</h1>
<p>नेपाल एक सुन्दर देश हो जहाँ...</p>
<table><tr><th>जिल्ला</th><th>जनसंख्या</th></tr>...</table>
```

---

## Cloud training (Lightning.ai)

Stage 1 was trained on a Lightning.ai A100. Cloud boxes are **ephemeral** — the
disk and installed packages do **not** survive a session ending. The playbook
that keeps a run safe (learned after losing a 9h run):

- **Back up every checkpoint to HF** (`--hf_repo Saurab0/akshara-pretrain`). The
  namespace is **case-sensitive** and the token must be **write**-scoped.
- **Reinstall `flash-linear-attention==0.5.1`** on each new machine (the GDN
  weights won't load without it).
- **Regenerate the corpus on-box** before each run (it isn't persisted).
- On a free-tier allowance, **measure tok/s early and trim `train_steps`** so the
  cosine schedule lands cleanly inside the budget instead of getting killed.

Pull the trained model anywhere:

```bash
huggingface-cli download Saurab0/akshara-pretrain pretrain_a100_v4.pt --local-dir checkpoints
```

---

## Apple Silicon (M4/M3/M2)

Replace `--device cuda` with `--device mps`. Reduce batch size for OCR fine-tune:

```bash
PYTHONPATH=. python scripts/train_ocr.py \
    --config configs/ocr_finetune.json \
    --batch_size 1 --grad_accum 32 \
    --device mps
```

---

## Project Structure

```
akshara/
├── src/
│   ├── models/
│   │   ├── vit.py              # DINOv2-S/14 encoder wrapper (pretrained)
│   │   ├── hybrid_decoder.py   # Transformer + GDN hybrid decoder
│   │   ├── attention.py        # Multi-head attention (GQA + RoPE)
│   │   ├── gdn_block.py        # GDN block
│   │   ├── connector.py        # Vision → decoder bridge (MLP)
│   │   └── vlm.py              # Full Akshara model assembly
│   ├── data/
│   │   ├── ocr_dataset.py      # CropOCRDataset (aspect-preserving pad, -100 mask)
│   │   ├── text_dataset.py     # Stage 1 corpus loader
│   │   ├── synth_data.py       # Synthetic crop generator (being rebuilt)
│   │   └── hf_dataset.py       # HuggingFace dataset loaders
│   └── pipeline.py             # Surya layout → recognizer → HTML assembly
├── scripts/
│   ├── prepare_data.py         # Data download + preparation
│   ├── pretrain.py             # Stage 1: language pretraining
│   ├── train_ocr.py            # Stage 2/3: OCR fine-tuning
│   └── generate.py             # Inference script
├── configs/
│   ├── base.json               # Shared model architecture
│   ├── pretrain.json           # Stage 1 training config
│   ├── ocr_finetune.json       # Stage 2/3 training config
│   └── smoke/                  # Fast smoke-test configs
├── config/
│   ├── config.py               # Dataclass configs
│   └── loader.py               # JSON → dataclass loader
├── fonts/                      # Noto Sans Devanagari
├── kaggle_run.py               # Kaggle kernel entry point
└── kernel-metadata.json        # Kaggle kernel metadata
```

---

## Name

**अक्षर** (Akshara) has two meanings in Sanskrit:

1. *Letter* — the atomic unit of written language, which is what OCR reads
2. *Imperishable / eternal* — that which does not decay

In Hindu mythology, Chitragupta (the divine scribe of Yama) records every deed in the language of akshara.
