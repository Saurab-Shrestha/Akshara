# Akshara (अक्षर)

**अक्षर** — Sanskrit for *letter* and *imperishable*. A small **multimodal LLM built from scratch**: image + text in, text out. It reads Devanagari + English documents, captions images, and answers questions about them.

> **This is a learning project.** The goal is to understand every stage of how a multimodal LLM is built: tokenizer, language pretraining, vision alignment, instruction tuning. We build and train everything ourselves except the vision backbone (pretrained DINOv2). Existing models (PaddleOCR-VL, Qwen-VL) are baselines and teachers, not replacements.

> **Status:** v1 language pretrain ✅ done (`Saurab0/akshara-pretrain`, dev ppl ≈ 6.7). It showed that Qwen's borrowed 248k vocab ate most of the compute and that the model was under-trained on repeated data. **v2 is in progress:** own 48k tokenizer, 443M decoder, Muon, QK-norm, and a 3B+ token corpus. Code for stages 0–3 is done and smoke-tested; next step is building the corpus and running the pretrain.

**Plan: [docs/ROADMAP_V2.md](docs/ROADMAP_V2.md)** · Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · OCR data plan: [docs/OCR_FINETUNE_PLAN.md](docs/OCR_FINETUNE_PLAN.md)

---

## Architecture

```
image ──► DINOv2-S/14 encoder (pretrained, 22M) ──► 1024 patch tokens × 384
                                                        │
                                          Connector (MLP + RMSNorm) 384 → 1024
                                                        │  visual tokens replace <|image|>
                                                        ▼
text  ──► akshara-bpe tokenizer (ours, 48k) ──► HybridDecoder (ours, 443M) ──► text
                                                32 layers, 1024 wide
                                                24 Gated DeltaNet + 8 attention (3:1)
                                                GQA 16q/4kv, RoPE, QK-norm
```

The decoder alternates Gated DeltaNet (GDN) recurrence layers with full-attention "exact recall" layers at a 3:1 ratio. On GPU the GDN runs via the FLA Triton kernel; a pure-Python fallback exists for CPU/Mac. Attention uses the fused SDPA (FlashAttention) kernel.

For full documents, pretrained Surya layout/table models split the page into regions and the model reads each crop (`src/pipeline.py`, region class → `<p>`/`<h1>`/`<table>`).

**Size:** 443M decoder (50M of it embedding) + 22M vision + ~1.1M connector ≈ 466M total.

---

## Training stages

| Stage | Script | What trains | Data | Status |
|---|---|---|---|---|
| 0 — Corpus | `prepare_data.py` | — | FineWeb-2 ne/hi/mr/sa 60% · FineWeb-Edu 30% · FineMath 3% · Wikipedia 7%, ~12B chars | code ✅ |
| 1 — Tokenizer | `train_tokenizer.py` | 48k byte-level BPE | corpus sample | code ✅ |
| 2 — Pack | `tokenize_corpus.py` | — | corpus → uint16 `.bin` | code ✅ |
| 3 — Pretrain | `pretrain.py` | decoder | ~3.2B tokens (Chinchilla: 8.5B) | code ✅, **run next** |
| 4 — Align | to build | connector | image–caption pairs (LLaVA-Pretrain) | designed |
| 5 — Instruct | to build | connector + decoder | VQA, doc QA, OCR (ocr-mlt-50m, Mozhi-LR) | designed |

v1 (finished): 308M decoder with the Qwen vocab, ~540M tokens, AdamW. Pinned in `configs/a100_pretrain.json`, and v1 checkpoints still load.

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

## Data & pretraining (v2)

```bash
# 0. corpus: ~12B chars ≈ 3.4B tokens, ~30 GB JSONL (use --total_chars 30e9 for Chinchilla 8.5B)
PYTHONPATH=. python scripts/prepare_data.py --out_dir data/corpus_v5

# 1. tokenizer: prints chars/token vs Qwen; commit tokenizer/akshara-bpe/
PYTHONPATH=. python scripts/train_tokenizer.py --input data/corpus_v5/train.jsonl \
    --vocab_size 49152 --out_dir tokenizer/akshara-bpe --compare Qwen/Qwen3.5-0.8B

# 2. pack into uint16 token files
PYTHONPATH=. python scripts/tokenize_corpus.py --input data/corpus_v5/train.jsonl
PYTHONPATH=. python scripts/tokenize_corpus.py --input data/corpus_v5/val.jsonl

# 3. pretrain (multi-session; every checkpoint backed up to Saurab0/akshara-pretrain-v2)
PYTHONPATH=. python scripts/pretrain.py --config configs/a100_pretrain_v2.json
#    resume: --resume checkpoints/pretrain_v2.pt
```

**Compute warning:** 3.2B tokens at 443M params is ~8.5e18 FLOPs, so **~19 h on an A100 at 40% MFU, ~66 h at v1's measured efficiency**. Measure tok/s in the first 100 steps and set `train_steps` from that. See [ROADMAP_V2.md §3](docs/ROADMAP_V2.md).

**Language eval** (works on v1 and v2 checkpoints):

```bash
PYTHONPATH=. python scripts/lm_eval.py --ckpt checkpoints/pretrain_v2.pt --device cuda
```

**Smoke test** (CPU, about a minute; needs a tokenizer at `tokenizer/akshara-bpe` or `--tokenizer <dir>`):

```bash
PYTHONPATH=. python scripts/pretrain.py --config configs/smoke/pretrain.json
```

**Self-checks:** `PYTHONPATH=. python src/models/attention.py`, `src/optim.py`, `src/data/text_dataset.py`.

### OCR data (stage 5)

Reading Devanagari is one of the stage-5 tasks. Data plan: [docs/OCR_FINETUNE_PLAN.md](docs/OCR_FINETUNE_PLAN.md).

| Dataset | Use |
|---|---|
| [interfaze-ai/ocr-mlt-50m](https://huggingface.co/datasets/interfaze-ai/ocr-mlt-50m) | filter `ne`/`hi`/`en` → bulk real text lines (Apache 2.0) |
| [Mozhi-LR (CVIT/IIIT)](https://cvit.iiit.ac.in/images/ConferencePapers/2024/Printed-OCR-for-Extremely-Low-resource-Indic-Languages.pdf) | real printed Nepali words |
| [oscar-corpus/mOSCAR](https://huggingface.co/datasets/oscar-corpus/mOSCAR) | image source to pseudo-label with a teacher |
| [heiDATA printed Devanagari](https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/EGOKEI) | 5,139 real lines, held-out **eval only** |

`scripts/train_ocr.py` (crop → text, CER eval) still works for OCR-only experiments on the v2 decoder:

```bash
PYTHONPATH=. python scripts/train_ocr.py --config configs/ocr_finetune.json \
    --train_path data/crops/lines/data.jsonl --device cuda
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

v1 was trained on a Lightning.ai A100, and v2 targets the same. Cloud boxes are **ephemeral** — the
disk and installed packages do **not** survive a session ending. The playbook
that keeps a run safe (learned after losing a 9h run):

- **Back up every checkpoint to HF** (`hf_repo` in the run config, e.g. `Saurab0/akshara-pretrain-v2`). The
  namespace is **case-sensitive** and the token must be **write**-scoped.
- **Reinstall `flash-linear-attention==0.5.1`** on each new machine (the GDN
  weights won't load without it).
- **Corpus isn't persisted on the box.** Regenerate it, or better, push `data/corpus_v5/*.bin` to an HF dataset repo once and download it each session.
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
│   │   ├── text_dataset.py     # PackedTokenDataset (.bin) + JSONL loader
│   │   ├── synth_data.py       # Synthetic crop generator (being rebuilt)
│   │   └── hf_dataset.py       # HuggingFace dataset loaders
│   ├── optim.py                # Muon + AdamW optimizer split
│   └── pipeline.py             # Surya layout → recognizer → HTML assembly
├── scripts/
│   ├── prepare_data.py         # Streaming, char-budgeted corpus download
│   ├── train_tokenizer.py      # Own byte-level BPE tokenizer
│   ├── tokenize_corpus.py      # JSONL → packed uint16 .bin
│   ├── pretrain.py             # Language pretraining (decoder)
│   ├── train_ocr.py            # Stage 2/3: OCR fine-tuning
│   └── generate.py             # Inference script
├── configs/
│   ├── base.json               # Shared v2 architecture
│   ├── a100_pretrain_v2.json   # v2 pretrain (443M, Muon)
│   ├── a100_pretrain.json      # v1 pretrain (historical, pinned)
│   ├── ocr_finetune.json       # Stage 2/3 training config
│   └── smoke/                  # Fast smoke-test configs
├── config/
│   ├── config.py               # Dataclass configs
│   └── loader.py               # JSON → dataclass loader
├── tokenizer/                  # akshara-bpe/ (trained tokenizer), verify.py
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
