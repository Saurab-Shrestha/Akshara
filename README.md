# Akshara (अक्षर)

**अक्षर** — Sanskrit for *letter* and *imperishable*. An OCR model that reads Devanagari script and outputs structured HTML.

Full-page document understanding: a single image in, clean `<p>`, `<h1>`, `<table>` HTML out — no bounding boxes, no post-processing pipeline.

---

## Architecture

```
Image (448×448)
    │
    ▼
ViT-S/16 Encoder          — 12 layers, 384-dim, 28×28 = 784 patches
    │
    ▼
Connector (MLP)            — projects 384 → 768
    │
    ▼
Hybrid Decoder (12 layers) — alternates Transformer ↔ GDN blocks
    │                        GQA (12 query / 3 KV heads), RoPE, bf16
    ▼
HTML output tokens         — <p>नेपाल...</p>, <h1>...</h1>, <table>...</table>
```

The decoder is a hybrid of standard Transformer blocks and GDN (Generalized Divisive Normalization) blocks, which model multiplicative interactions between features — useful for script with dense conjunct characters.

**Model size**: ~300M parameters (embedding-heavy due to 248k vocab)

---

## Training

Three stages, each building on the last:

| Stage | What trains | Data | Goal |
|---|---|---|---|
| 1 — Language pretrain | Decoder only | Nepali Wikipedia / CulturaX | Learn Devanagari token patterns |
| 2 — OCR fine-tune | Full model | RenderedText + CORD + IAM | Learn image → HTML mapping |
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
# Nepali Wikipedia corpus (language pretrain)
PYTHONPATH=. python scripts/prepare_data.py --stage corpus --max_samples 200000

# English OCR datasets (RenderedText + CORD receipts + IAM handwriting)
PYTHONPATH=. python scripts/prepare_data.py --stage rendered --max_samples 100000
PYTHONPATH=. python scripts/prepare_data.py --stage cord
PYTHONPATH=. python scripts/prepare_data.py --stage iam

# Synthetic Nepali document images
PYTHONPATH=. python scripts/prepare_data.py --stage synth \
    --font_path fonts/NotoSansDevanagari-Regular.ttf --n_synth 50000

# Merge all OCR datasets into train/val
PYTHONPATH=. python scripts/prepare_data.py --stage merge
```

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
│   │   ├── vit.py              # ViT-S/16 vision encoder
│   │   ├── hybrid_decoder.py   # Transformer + GDN hybrid decoder
│   │   ├── attention.py        # Multi-head attention (GQA + RoPE)
│   │   ├── gdn_block.py        # GDN block
│   │   ├── connector.py        # Vision → decoder bridge (MLP)
│   │   └── vlm.py              # Full Akshara model assembly
│   ├── data/
│   │   ├── ocr_dataset.py      # JSONL dataset loader
│   │   ├── synth_data.py       # Synthetic Nepali page generator
│   │   └── hf_dataset.py       # HuggingFace dataset loaders
│   └── detection/
│       ├── surya_detector.py   # Surya-based text region detection
│       └── pipeline.py         # Full detection → recognition pipeline
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
