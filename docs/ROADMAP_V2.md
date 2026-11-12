# Akshara v2 — Roadmap: our own multimodal LLM, from scratch

**Goal:** a small multimodal LLM — image + text in, text out — that reads
Devanagari and English documents, captions images, and answers simple
questions about them. Every learned component except the vision backbone is
built and trained by us, so every stage is something we understand end to end.

**Honest expectation:** at this compute it will caption, OCR, and do simple
VQA well; multi-step reasoning will stay weak. That is the trade we accept for
building the language model ourselves instead of borrowing one.

---

## 1. Why v2 (what v1 taught us)

v1 (Stage 1, finished, HF `Saurab0/akshara-pretrain`) worked — dev ppl ≈ 6.7,
fluent Nepali. Measuring it showed where the compute went:

| Finding | Number | Consequence |
|---|---|---|
| Qwen's 248k vocab on a 110M-body model | embedding 190M params; output head ≈ 380M FLOPs/token, more than all 16 layers | most of the compute went into the vocabulary, not the layers that learn |
| Qwen tokenizer on Devanagari | **1.46 chars/token** (ours: **3.52**) | the same Nepali text costs 2.4× more tokens |
| Corpus size vs tokens trained | local `data/corpus` ≈ 62M Qwen tokens; run trained 540M | the model saw the same text ~9× (if `corpus_v4` matched it) |
| Chinchilla-optimal for the body | ~20 tokens/param → ~2.2B | v1 was ~4× under-trained, on repeated data |

v2 fixes all four, and — since pretraining restarts anyway — takes the
architecture upgrades that would otherwise have been wasted on a finished run.

---

## 2. What changed (v1 → v2)

| | v1 | v2 |
|---|---|---|
| Tokenizer | Qwen3.5-0.8B, 248k | **own byte-level BPE, 32k** (`tokenizer/akshara-bpe`) |
| Decoder | 768 × 16 layers, 12 GDN + 4 attn | **1024 × 32 layers, 24 GDN + 8 attn** |
| Params | 308M (190M embedding) | **427M (34M embedding, 393M body)** |
| Context | 512 | **1024** |
| Attention | hand-written softmax + stored mask | **QK-norm + fused SDPA (FlashAttention)** |
| Optimizer | AdamW | **Muon (hidden matrices) + AdamW (rest)** |
| Data | ~1M docs in RAM, padded per doc | **char-budgeted streaming → packed uint16 `.bin`** |
| English share | 20% | 30%+ (instruction data is mostly English) |
| Vision | DINOv2-S/14, 22M | unchanged (see §5) |

Code: `scripts/train_tokenizer.py`, `scripts/tokenize_corpus.py`,
`scripts/prepare_data.py` (rewritten), `src/optim.py`,
`src/models/attention.py`, `src/data/text_dataset.py::PackedTokenDataset`,
`configs/base.json`, `configs/a100_pretrain_v2.json`. v1 is pinned in
`configs/a100_pretrain.json` and v1 checkpoints still load.

---

## 3. The stages

```
 0. Corpus        prepare_data.py        → data/corpus_v5/{train,val}.jsonl
 1. Tokenizer     train_tokenizer.py     → tokenizer/akshara-bpe/
 2. Pack          tokenize_corpus.py     → data/corpus_v5/{train,val}.bin
 3. Pretrain      pretrain.py            → Saurab0/akshara-pretrain-v2    (decoder only)
 4. Align         [to build]             → connector learns vision → token space
 5. Instruct      [to build]             → chat, VQA, doc QA, OCR
```

Stages 0–3 are implemented and smoke-tested. 4–5 are built once the v2
decoder exists (their design is fixed below, and the tokenizer already
reserves their special tokens, so nothing upstream changes).

### Stage 0 — Corpus (`scripts/prepare_data.py`)

Streams each source to disk until its character budget is met. Budgets are
characters, not documents (one Wikipedia article ≠ one forum post).

| Source | Share | Note |
|---|---|---|
| FineWeb-2 `npi_Deva` (Nepali) | 30% | may run dry first — the script warns |
| FineWeb-2 `hin_Deva` (Hindi) | 25% | |
| FineWeb-2 `mar_Deva` / `san_Deva` | 4% / 1% | |
| FineWeb-Edu (English, EduScore ≥ 3) | 30% | |
| FineMath `finemath-4plus` | 3% | digits, equations, step-by-step text |
| Wikipedia ne / hi / en | 3% / 2% / 2% | curated |

Validation = 1 doc in 200, chosen by a CRC32 hash of the text (deterministic,
spans every source).

```bash
PYTHONPATH=. python scripts/prepare_data.py --total_chars 12e9 --out_dir data/corpus_v5
```
12e9 chars ≈ 3.4B tokens ≈ 30 GB of JSONL. For Chinchilla-optimal 427M
(~8.5B tokens) use `--total_chars 30e9`.

### Stage 1 — Tokenizer (`scripts/train_tokenizer.py`)

Byte-level BPE, NFC-normalised, with a pre-tokenizer that keeps Devanagari
vowel signs / virama / ZWJ attached to their consonants (the GPT-2 regex tears
them off). Special tokens reserved now: `<|bos|> <|eos|> <|pad|> <|image|>
<|im_start|> <|im_end|>` + 10 spare. Details: [02_tokenizer.md](02_tokenizer.md).

```bash
PYTHONPATH=. python scripts/train_tokenizer.py --input data/corpus_v5/train.jsonl \
    --vocab_size 32768 --out_dir tokenizer/akshara-bpe --compare Qwen/Qwen3.5-0.8B
```
Measured on a 100 MB sample of the v1 corpus: **3.52 chars/token Devanagari,
4.05 Latin** (Qwen: 1.46 / 4.22). Re-check on the full corpus; try 48k or 64k
if Latin compression lags (anything ≤ 65,535 fits the uint16 shards). Commit
the resulting `tokenizer/akshara-bpe/` — every later stage depends on it.

### Stage 2 — Pack (`scripts/tokenize_corpus.py`)

```bash
PYTHONPATH=. python scripts/tokenize_corpus.py --input data/corpus_v5/train.jsonl
PYTHONPATH=. python scripts/tokenize_corpus.py --input data/corpus_v5/val.jsonl
```
Each doc → `<|bos|> … <|eos|>`, concatenated into one uint16 file.
`PackedTokenDataset` reads fixed 1024-token windows from a memmap: no padding,
no RAM, no tokenization during training.

### Stage 3 — Pretrain v2 (`scripts/pretrain.py`)

```bash
pip install flash-linear-attention==0.5.1 && huggingface-cli login
PYTHONPATH=. python scripts/pretrain.py --config configs/a100_pretrain_v2.json
# new session: pull pretrain_v2.pt from Saurab0/akshara-pretrain-v2, then
PYTHONPATH=. python scripts/pretrain.py --config configs/a100_pretrain_v2.json \
    --resume checkpoints/pretrain_v2.pt
```

Recipe: Muon (`adjust_lr_fn="match_rms_adamw"`, so one lr drives both
optimizers) lr 4e-4 → 4e-5 cosine, 1k warmup, wd 0.1, 98k tokens/step
(8 × 1024 × 12), bf16 + FLA, 32.5k steps ≈ 3.2B tokens.

**Compute — read before starting.** 427M params ≈ 6 × 427M = 2.6 GFLOP/token.

| Tokens | FLOPs | at v1's measured efficiency (~36 TFLOP/s) | at 40% MFU (~125 TFLOP/s) |
|---|---|---|---|
| 3.2B (this config) | 8.2e18 | **~63 h** | ~18 h |
| 8.5B (Chinchilla) | 2.2e19 | ~170 h | ~48 h |

v1 ran at roughly 12% MFU; the smaller vocab, fused attention, and
`compile: true` should raise it, but **measure tok/s in the first 100 steps**
and set `train_steps` from that. This is a multi-session run — the HF backup
on every save is what makes it survivable. If the budget can't cover it, the
fallback is a 768 × 20 config (163M, ~3× cheaper).

Gate to move on: `lm_eval.py` on the v2 checkpoint should match or beat v1 on
Nepali and clearly beat it on English.

### Stage 4 — Alignment (to build)

Freeze DINOv2 and the decoder; train only the connector on image–caption
pairs (LLaVA-Pretrain 558k; add Devanagari-text images from ocr-mlt-50m so
the connector learns text-in-image early). Prompt format:

```
<|im_start|>user\n<|image|>\nDescribe the image.<|im_end|>\n<|im_start|>assistant\n{caption}<|im_end|>
```
`<|image|>` is replaced by the connector's visual tokens; loss only on the
assistant span. One epoch, lr ~1e-3 (connector only).

Needed code: chat-format dataset with image-token splicing (generalises
`CropOCRDataset`), loss masking by span, HF backup in the training script
(`train_ocr.py` does not have it yet).

### Stage 5 — Instruction tuning (to build)

Unfreeze connector + decoder (vision optionally, low lr). Mix:

| Data | Teaches |
|---|---|
| LLaVA-Instruct 665k / The Cauldron | general VQA, chat |
| Docmatix | document QA |
| ocr-mlt-50m (`ne`/`hi`/`en`), Mozhi-LR, synthetic crops | reading Devanagari — the OCR work from [OCR_FINETUNE_PLAN.md](OCR_FINETUNE_PLAN.md) lives here as one task |
| translated / Nepali-written QA (small) | answering in Nepali |

Plus **line-shaped / variable-resolution input**: scale each image to a fixed
patch budget at its own aspect ratio (a text line becomes e.g. 64×8 patches
instead of a mostly-blank 32×32 square). DINOv2's position embeddings
interpolate, so only batching/masking changes.

Eval: CER on heiDATA (held out, never trained), a small VQA set, and
`lm_eval.py` to confirm the text ability survived.

---

## 4. Order of work

1. `prepare_data.py` on the training box (or a big-disk CPU box → upload to HF datasets)
2. Train + commit the tokenizer; check compression on the full corpus
3. Pack to `.bin`
4. 100-step pretrain probe → measure tok/s → set `train_steps`
5. Full pretrain (multi-session), `lm_eval.py` gate
6. Build + run Stage 4, then Stage 5

---

## 5. Vision encoder — why it stays pretrained (22M)

DINOv2-S/14 is 22M params: tiny next to the 427M decoder (LLaVA-style models
are typically 5–20% vision). Training a ViT from scratch needs hundreds of
millions of images; we still build everything around it (patching, position
interpolation, variable resolution, the connector). Upgrade path if the
vision side becomes the bottleneck: DINOv2-B/14 (86M) or SigLIP-B/16 (93M,
text-aligned) — a config + connector-width change. A from-scratch tiny ViT is
a good side experiment, not the main path.

---

## 6. Audio (decided 2026-11-12)

**Speech output: separate TTS model.** Akshara always outputs text, and a
separate pretrained TTS model reads that text aloud. The decoder never
generates audio, so there are no codec tokens, no vocabulary growth and no
audio-generation training. Any Nepali/Hindi/English TTS can be plugged in
behind `generate.py` / `src/pipeline.py` without touching the model.

**Speech input: later, same pattern as vision (not scheduled).**
```
audio ──► Whisper-small encoder (pretrained, ~88M) ──► connector (4× downsample)
      ──► replaces <|audio|> ──► decoder ──► text
```
- `<|audio|>` takes one of the 10 reserved special tokens, so there's no embedding resize.
- 30 s of audio → 1500 Whisper frames → ~375 tokens after downsampling, which fits the 4096-position RoPE tables.
- Stages: align on ASR pairs (audio → transcript), then instruction-tune
  (spoken questions, speech translation).
- Nepali data: OpenSLR 54, Common Voice `ne`, FLEURS `ne_np`.

**Why not native speech output:** it needs thousands of codec tokens (EnCodec/Mimi/SNAC)
in the vocabulary, much heavier training, and Nepali TTS data is scarce. If
it's ever wanted, grow the embedding after pretraining and train the new
rows in a dedicated audio stage.
