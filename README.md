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
| 0 — Corpus | `prepare_data.py` | — | `corpus_v6`, 30B chars ≈ 7.6B tokens: FineWeb-2 ne/hi/mr/sa 35% · Nepali extras (HPLT, IndicCorp, FinePDFs, GlotCC) 12% · FineWeb-Edu 24% · code 9% + JSON/YAML/Markdown 1% · FineMath 6% · Cosmopedia 4% · Wikipedia 7% · parallel (BPCC, OPUS) 2% · plus Nepal's minority languages | building |
| 1 — Tokenizer | `train_tokenizer.py` | 48k byte-level BPE | corpus sample | code ✅ |
| 2 — Pack | `tokenize_corpus.py` | — | corpus → uint16 `.bin` | code ✅ |
| 3 — Pretrain | `pretrain.py` | decoder | ~8.7B tokens at 2k context, WSD + anneal mix (Chinchilla ≈ 8.9B) | code ✅, **run next** |
| 3b — Long context | `pretrain.py` | decoder | 2k → 16k, ~0.4B tokens of long docs, RoPE base 10k → 1M | config ✅ |
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
pip install flash-linear-attention         # GDN Triton kernel (CUDA only); code targets the 0.6 API
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
# 0a. Nepali sources beyond FineWeb-2: ~18B chars available, all permissive.
#     HPLT 2.0 / IndicCorpV2 (CC0), FinePDFs (ODC-BY), GlotCC (CC0), DCAD (train-only).
python scripts/collect_nepali.py --list        # sources, licenses, configs
python scripts/collect_nepali.py --sources hplt2,indiccorp,finepdfs
python scripts/collect_nepali.py               # all five -> data/nepali/*.jsonl

# 0b–2. one command: corpus → (tokenizer, only if missing) → packed .bin → private HF dataset.
#     Resumable per source; a watchdog restarts the data step if an HF stream hangs.
TOTAL_CHARS=30e9 NEPALI_SHARE=0.12 OUT=data/corpus_v6 HF_REPO=Saurab0/akshara-corpus-v6 \
    nohup caffeinate -is scripts/build_corpus.sh > logs/build_v6.log 2>&1 &
#     The 48k tokenizer was trained once on corpus_v5 and is committed (tokenizer/akshara-bpe/);
#     RETRAIN_TOKENIZER=1 retrains it (new ids: old checkpoints become incompatible).
#     Packing drops documents with a 1–13 token span repeated ≥32× in a row (loss-spike junk).

# 2b. anneal mix for the LR-decay phase (math/code/textbooks/parallel-heavy)
PYTHONPATH=. python scripts/prepare_data.py --mix anneal --no_minority --total_chars 4.5e9 \
    --nepali_share 0.10 --out_dir data/corpus_anneal
PYTHONPATH=. python scripts/tokenize_corpus.py --input data/corpus_anneal/train.jsonl

# 3. pretrain (multi-session; every checkpoint backed up to Saurab0/akshara-pretrain-v2)
PYTHONPATH=. python scripts/pretrain.py --config configs/a100_pretrain_v2.json      # A100, bf16
#    or configs/kaggle_t4_pretrain_v2.json (T4/P100, fp16 + GradScaler). Don't mix fp16/bf16 in one run.
#    resume: --resume checkpoints/pretrain_v2.pt
#    first: the comparison runs in configs/ablations/ (148M proxy, LR sweep, attention-only, AdamW)

# 3b. long context 2k → 16k from the finished weights
PYTHONPATH=. python scripts/tokenize_corpus.py --input data/corpus_v6/train.jsonl \
    --min_tokens 4096 --out data/corpus_long/train.bin
PYTHONPATH=. python scripts/pretrain.py --config configs/a100_longctx_v2.json

# tests (CPU, ~1 min; also run in GitHub Actions)
PYTHONPATH=. pytest -q tests/
```

**Nepali collection** is append-safe: rerunning a source skips documents already on disk (`--restart` truncates). Each source has a built-in cap (~4.3B chars total, ~13 GB JSONL); `--max_chars N` overrides it and `--max_chars -1` removes all caps (~19B chars, ~57 GB). Parquet shards are downloaded one at a time to `~/.cache/akshara_nepali` and deleted after reading; `data/nepali/MANIFEST.json` records license, volume and filter counts per source.

**Compute warning:** 8.7B tokens at 443M params is ~2.3e19 FLOPs: **~52 h on an A100 at 40% MFU, ~180 h at v1's measured efficiency (~12% MFU)**, roughly 300–700 h on Kaggle 2× T4. Measure tok/s in the first 100 steps and set `train_steps` from that. WSD makes a shorter run safe: the pre-decay checkpoint (`pretrain_v2_predecay_<step>.pt`) can be extended later. See [ROADMAP_V2.md §3](docs/ROADMAP_V2.md).

**Inference:** `HybridDecoder.generate()` uses a KV cache for the attention layers and the fixed-size memory for the GDN layers (temperature / top-k / top-p). `src/speculative.py` does speculative decoding with a small draft model that shares the tokenizer (e.g. the 148M proxy).

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
- **Install `flash-linear-attention`** on each GPU machine for speed. v2 GDN
  weights (`gdn_impl: native`) load and run anywhere, CPU and Mac included;
  only v1 checkpoints (`gdn_impl: fla_layer`) need FLA to load.
- **Corpus isn't persisted on the box.** Download `train.bin` / `val.bin` from the
  private `Saurab0/akshara-corpus-v6` dataset each session.
- On a free-tier allowance, **measure tok/s early and set `train_steps`** so the
  WSD decay lands inside the budget, and keep the pre-decay checkpoint to extend later.

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
