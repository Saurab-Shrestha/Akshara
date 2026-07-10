"""
Kaggle entry point — runs the full training pipeline on a T4 GPU.

/kaggle/src/     = read-only, our pushed code lives here
/kaggle/working/ = writable, all outputs (fonts, data, checkpoints) go here

Download checkpoints after run:
    kaggle kernels output saurabstha5/akshara -p ./checkpoints
"""

import os
import subprocess
import sys

# /kaggle/src/ is read-only. Stay in /kaggle/working/ for all writes.
SRC  = os.path.dirname(os.path.abspath(__file__))  # /kaggle/src
WORK = "/kaggle/working"

os.chdir(WORK)
sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = SRC
print(f"SRC : {SRC}")
print(f"WORK: {WORK}")


def run(cmd, **kwargs):
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kwargs)


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── 1. dependencies ───────────────────────────────────────────────────────────

section("Installing dependencies")
# Do NOT reinstall torch — Kaggle's preinstalled build matches the T4 (sm_75).
run("pip install -q transformers datasets opencv-python-headless")

# ── 2. verify GPU ─────────────────────────────────────────────────────────────

section("GPU check")
import torch
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: no GPU — training will be very slow")

# ── 3. Noto font → /kaggle/working/fonts/ ────────────────────────────────────

section("Noto Sans Devanagari font")
font_dir  = os.path.join(WORK, "fonts")
font_path = os.path.join(font_dir, "NotoSansDevanagari-Regular.ttf")
os.makedirs(font_dir, exist_ok=True)
if not os.path.exists(font_path):
    run(
        "wget -q 'https://github.com/googlefonts/noto-fonts/raw/main/"
        "hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf'"
        f" -O {font_path}"
    )
print(f"font: {font_path}  ({os.path.getsize(font_path)//1024} KB)")

# ── 4. data → /kaggle/working/data/ ──────────────────────────────────────────

section("Data preparation")
py = f"python {SRC}/scripts/prepare_data.py"

run(f"PYTHONPATH={SRC} {py} --stage corpus   --max_samples 200000")
run(f"PYTHONPATH={SRC} {py} --stage rendered --max_samples 100000")
run(f"PYTHONPATH={SRC} {py} --stage cord")
run(f"PYTHONPATH={SRC} {py} --stage iam")
run(f"PYTHONPATH={SRC} {py} --stage synth --font_path {font_path} --n_synth 50000")
run(f"PYTHONPATH={SRC} {py} --stage merge")

# ── 5. checkpoints dir ────────────────────────────────────────────────────────

ckpt_dir = os.path.join(WORK, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

# ── 6. stage 1: language pretrain ────────────────────────────────────────────

section("Stage 1 — Language pretraining")
run(
    f"PYTHONPATH={SRC} python {SRC}/scripts/pretrain.py "
    f"--config {SRC}/configs/pretrain.json "
    f"--train_path {WORK}/data/corpus/train.jsonl "
    f"--dev_path   {WORK}/data/corpus/val.jsonl "
    f"--out_ckpt   {ckpt_dir}/pretrain.pt "
    f"--device cuda"
)

# ── 7. stage 2: OCR fine-tune on English docs ─────────────────────────────────

section("Stage 2 — OCR fine-tune (English documents)")
run(
    f"PYTHONPATH={SRC} python {SRC}/scripts/train_ocr.py "
    f"--config        {SRC}/configs/ocr_finetune.json "
    f"--pretrain_ckpt {ckpt_dir}/pretrain.pt "
    f"--train_path    {WORK}/data/documents/train.jsonl "
    f"--dev_path      {WORK}/data/documents/val.jsonl "
    f"--out_ckpt      {ckpt_dir}/ocr.pt "
    f"--device cuda"
)

# ── 8. stage 3: Nepali fine-tune ──────────────────────────────────────────────

section("Stage 3 — Nepali fine-tune")
run(
    f"PYTHONPATH={SRC} python {SRC}/scripts/train_ocr.py "
    f"--config        {SRC}/configs/ocr_finetune.json "
    f"--pretrain_ckpt {ckpt_dir}/ocr.pt "
    f"--train_path    {WORK}/data/documents/nepali_synth/data_train.jsonl "
    f"--dev_path      {WORK}/data/documents/nepali_synth/data_val.jsonl "
    f"--lr 3e-5 --train_steps 10000 "
    f"--out_ckpt {ckpt_dir}/ocr_nepali.pt "
    f"--device cuda"
)

# ── 9. summary ────────────────────────────────────────────────────────────────

section("Done")
for name in ["pretrain.pt", "ocr.pt", "ocr_nepali.pt"]:
    p = os.path.join(ckpt_dir, name)
    if os.path.exists(p):
        print(f"  {p}  ({os.path.getsize(p)/1e6:.0f} MB)")

print("\nDownload checkpoints:")
print("  kaggle kernels output saurabstha5/akshara -p ./checkpoints")
