"""
Kaggle entry point — Stage 1: language pretraining only.

Kaggle script kernels only upload this single file; the rest of the codebase
is cloned from GitHub at runtime.

/kaggle/working/Akshara/  = cloned repo (SRC)
/kaggle/working/data/     = corpus + cache
/kaggle/working/checkpoints/ = saved weights

Download checkpoint when done:
    kaggle kernels output saurabstha5/akshara -p ./checkpoints
"""

import os
import subprocess
import sys

WORK     = "/kaggle/working"
REPO_URL = "https://github.com/Saurab-Shrestha/Akshara.git"
REPO_DIR = os.path.join(WORK, "Akshara")
CKPT_DIR = os.path.join(WORK, "checkpoints")

os.chdir(WORK)


def run(cmd):
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── 1. download repo ──────────────────────────────────────────────────────────

section("Downloading Akshara repo")
# git clone over HTTPS prompts for credentials in non-interactive envs even on
# public repos. wget the zip instead — simpler and always works.
ZIP = "/tmp/akshara.zip"
run(f"wget -q {REPO_URL.replace('.git', '')}/archive/refs/heads/main.zip -O {ZIP}")
run(f"unzip -q -o {ZIP} -d /tmp/akshara_unzip")
if os.path.exists(REPO_DIR):
    run(f"rm -rf {REPO_DIR}")
run(f"mv /tmp/akshara_unzip/Akshara-main {REPO_DIR}")

SRC = REPO_DIR
sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = SRC
print(f"SRC : {SRC}")

# ── 2. dependencies ───────────────────────────────────────────────────────────

section("Installing dependencies")
# Skip torch/torchvision — Kaggle's pre-installed build is already CUDA-compatible.
# Reinstalling from PyPI would overwrite it with a CPU or incompatible CUDA build.
run(f"grep -vE '^torch|^torchvision|^torchaudio' {SRC}/requirements.txt "
    f"| pip install -q -r /dev/stdin")

# ── 3. verify GPU ─────────────────────────────────────────────────────────────

section("GPU check")
import torch
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: no GPU — training will be very slow")

# ── 4. Nepali corpus ──────────────────────────────────────────────────────────

section("Nepali corpus (Wikipedia)")
run(f"PYTHONPATH={SRC} python {SRC}/scripts/prepare_data.py "
    f"--stage corpus --max_samples 200000")

# ── 5. pretraining ────────────────────────────────────────────────────────────

# 5 000 optimizer steps fits comfortably within Kaggle's 9-hour GPU limit.
# Each step = batch_size(16) × grad_accum(4) × seq_len(2048) = 131k tokens.
# Resume next session by pushing again with --resume pointing at pretrain.pt.

section("Stage 1 — Language pretraining  (5 000 steps)")
os.makedirs(CKPT_DIR, exist_ok=True)

resume = os.path.join(CKPT_DIR, "pretrain.pt")
resume_flag = f"--resume {resume}" if os.path.exists(resume) else ""

run(
    f"PYTHONPATH={SRC} python {SRC}/scripts/pretrain.py "
    f"--config      {SRC}/configs/pretrain.json "
    f"--train_path  {WORK}/data/corpus/train.jsonl "
    f"--dev_path    {WORK}/data/corpus/val.jsonl "
    f"--train_steps 5000 "
    f"--out_ckpt    {CKPT_DIR}/pretrain.pt "
    f"--device cuda "
    f"{resume_flag}"
)

# ── 6. done ───────────────────────────────────────────────────────────────────

section("Done")
ckpt = os.path.join(CKPT_DIR, "pretrain.pt")
if os.path.exists(ckpt):
    print(f"  {ckpt}  ({os.path.getsize(ckpt)/1e6:.0f} MB)")

print("\nDownload checkpoint:")
print("  kaggle kernels output saurabstha5/akshara -p ./checkpoints")
