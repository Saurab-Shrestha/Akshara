# 01 · Project Setup

Getting the development environment ready before writing any model code.

---

## Folder structure

```
nepali-ocr-vlm/
├── src/
│   ├── models/          ← all model components
│   └── data/            ← dataset classes (coming)
├── tokenizer/
│   └── verify.py        ← Devanagari coverage check
├── scripts/             ← training + inference (coming)
├── data/
│   ├── corpus/          ← Nepali text for pretraining
│   └── synthetic/       ← (image, text) pairs for OCR
├── docs/                ← you are here
├── venv/                ← Python virtual environment (not committed)
└── requirements.txt
```

Why this layout? `src/models/` keeps each component isolated — you can test
`rms_norm.py` without importing the full model. `PYTHONPATH=.` lets any file
import as `from src.models.rms_norm import RMSNorm`.

---

## Setup commands

```bash
# Clone and enter
cd ~/Documents
mkdir -p ocr/nepali-ocr-vlm && cd ocr/nepali-ocr-vlm

# Virtual environment (keeps dependencies isolated)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Dependencies (`requirements.txt`)

```
torch
torchvision
transformers        ← Hugging Face, for loading Qwen tokenizer
huggingface-hub     ← downloading model files
sentencepiece       ← tokenizer backend
tiktoken            ← alternative tokenizer (OpenAI format)
datasets            ← loading Nepali text corpus
Pillow              ← image loading for OCR
numpy
Faker               ← generating synthetic text for data pipeline
tqdm                ← progress bars during training
tensorboard         ← loss curves
ipykernel           ← running in Jupyter on Kaggle
```

**Note on `fla` (flash-linear-attention):** Not in requirements.txt because it
requires CUDA + Triton. On Kaggle, install separately:
```bash
pip install git+https://github.com/fla-org/flash-linear-attention
```

---

## Running any file

Every file in `src/models/` has a `__main__` self-check at the bottom.
Always run with `PYTHONPATH=.` so `from src.models.X import Y` resolves:

```bash
PYTHONPATH=. venv/bin/python src/models/rms_norm.py
PYTHONPATH=. venv/bin/python src/models/hybrid_decoder.py
```

---

## What we did NOT do

- No `__init__.py` files yet — not needed until we build a training script that
  imports from multiple modules. Add them then.
- No Docker — overkill for a learning project on one machine.
- No pre-commit hooks — add when the team grows.

---

**Next:** [02 · Tokenizer](02_tokenizer.md)
