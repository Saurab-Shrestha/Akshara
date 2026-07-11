# Akshara (अक्षर)

**अक्षर** — Sanskrit for *letter* and *imperishable*. Nepali/Devanagari document OCR: page image in, structured `<p>`, `<h1>`, `<table>` HTML out.

Pretrained layout/table-structure models (Surya) find the *structure*; a custom-trained crop recognizer does the *reading*. Structure is geometry and language-agnostic — reading Devanagari is the only part that needs training, so it's the only part we train.

**Full design doc: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** (flowcharts, rationale, data formats).

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
    Hybrid decoder         — 12 layers, GDN:attention 3:1, GQA, RoPE
    │
    ▼  plain Nepali text per crop
    │
HTML assembler (plain Python)                 — region class → <p>/<h1>/<table>
```

The decoder alternates Gated DeltaNet (GDN) recurrence layers with full-attention "exact recall" layers at a 3:1 ratio.

**Recognizer size**: ~296M parameters (190M is the 248k-vocab embedding table)

---

## Training

Curriculum: learn the language, learn to read a line, learn to read a paragraph (the line→paragraph warmup follows [Pix2Struct](https://arxiv.org/abs/2210.03347)'s reading-curriculum result):

| Stage | What trains | Data | Goal |
|---|---|---|---|
| 1 — Language pretrain | Decoder only | Nepali Wikipedia / CulturaX | Learn Devanagari token patterns |
| 2A — Line warmup | Full model | Synthetic single-line crops | Glyph-level reading (conjuncts, matras) |
| 2B — Paragraph OCR | Full model | Synthetic multi-line crops | Multi-line reading |
| 3 — Nepali fine-tune | Full model | Synthetic Nepali pages | Adapt to Devanagari documents |

---

## Setup

```bash
git clone git@github.com:Saurab-Shrestha/Akshara.git
cd Akshara

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the Noto Devanagari font (required for synthetic data generation):

```bash
mkdir -p fonts
wget -q 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf' \
     -O fonts/NotoSansDevanagari-Regular.ttf
```

---

## Data Preparation

```bash
# Nepali Wikipedia corpus (Stage 1 language pretrain)
PYTHONPATH=. python scripts/prepare_data.py --stage corpus --max_samples 200000

# Stage 2A: synthetic single-line crops (rendered from the corpus)
PYTHONPATH=. python src/data/synth_data.py \
    --corpus data/corpus/train.jsonl --fonts fonts/ \
    --out data/crops/lines --n 200000 --max_lines 1

# Stage 2B: synthetic paragraph crops
PYTHONPATH=. python src/data/synth_data.py \
    --corpus data/corpus/train.jsonl --fonts fonts/ \
    --out data/crops/paras --n 100000 --max_lines 12
```

External datasets worth mixing in (see docs/ARCHITECTURE.md §4):

| Dataset | Use |
|---|---|
| [interfaze-ai/ocr-mlt-50m](https://huggingface.co/datasets/interfaze-ai/ocr-mlt-50m) | filter `ne`/`hi` → bulk Stage 2A data (Apache 2.0) |
| [Mozhi / IIIT printed lines](https://cvit.iiit.ac.in/usodi/tdocrmil.php) | real printed Hindi lines — Stage 2A supplement |
| [heiDATA printed Devanagari](https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/EGOKEI) | 5,139 real lines — held-out **eval only** |

---

## Training

**Stage 1 — Language pretraining** (~8h on T4, ~12h on M4)

```bash
PYTHONPATH=. python scripts/pretrain.py \
    --config configs/pretrain.json \
    --device cuda   # or mps for Apple Silicon
```

**Stage 2 — OCR fine-tune on English documents** (~4h on T4)

```bash
PYTHONPATH=. python scripts/train_ocr.py \
    --config configs/ocr_finetune.json \
    --pretrain_ckpt checkpoints/pretrain.pt \
    --device cuda
```

**Stage 3 — Nepali fine-tune** (~2h on T4)

```bash
PYTHONPATH=. python scripts/train_ocr.py \
    --config configs/ocr_finetune.json \
    --pretrain_ckpt checkpoints/ocr.pt \
    --train_path data/documents/nepali_synth/data_train.jsonl \
    --dev_path   data/documents/nepali_synth/data_val.jsonl \
    --lr 3e-5 --train_steps 10000 \
    --out_ckpt checkpoints/ocr_nepali.pt \
    --device cuda
```

**Smoke test** (runs in ~2 minutes on CPU, verifies the full pipeline):

```bash
PYTHONPATH=. python scripts/train_ocr.py --config configs/smoke/ocr_finetune.json --device cpu
```

---

## Inference

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

## Kaggle Training

The full pipeline is set up to run as a Kaggle script kernel (T4 GPU):

```bash
# Push to Kaggle (one command — runs data prep + all 3 training stages)
kaggle kernels push -p .

# Check progress
open https://www.kaggle.com/code/saurabstha5/akshara

# Download checkpoints when done
kaggle kernels output saurabstha5/akshara -p ./checkpoints
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
