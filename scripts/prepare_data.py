"""
Data preparation script — downloads and formats all training data.

STAGES
------
  corpus     Download Nepali Wikipedia → data/corpus/{train,val}.jsonl
             Used by scripts/pretrain.py for language pretraining.

  rendered   Download HuggingFaceM4/RenderedText → data/documents/
             Clean rendered-text images, good first OCR dataset.

  cord       Download CORD-v2 receipt dataset → data/documents/
             Teaches the model HTML table output for forms/invoices.

  iam        Download IAM handwriting dataset → data/documents/
             Teaches robustness to non-printed / degraded text.

  synth      Generate synthetic Nepali document images → data/documents/nepali/
             Requires a Devanagari font (see --font_path).

  all        Run corpus + rendered + cord + iam (no synth — font needed separately)

USAGE
------
  # On Kaggle / first time setup:
  python scripts/prepare_data.py --stage all

  # Nepali synthetic data (after downloading font):
  python scripts/prepare_data.py --stage synth --font_path fonts/NotoSansDevanagari-Regular.ttf

  # Single stage:
  python scripts/prepare_data.py --stage corpus --max_samples 100000
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


# ── corpus ──────────────────────────────────────────────────────────────────────

def prepare_corpus(max_samples: int = 200_000, val_ratio: float = 0.02):
    """Download Nepali Wikipedia and split into train/val JSONL."""
    print("\n=== Nepali corpus (Wikipedia) ===")
    from datasets import load_dataset

    os.makedirs("data/corpus", exist_ok=True)
    train_path = "data/corpus/train.jsonl"
    val_path   = "data/corpus/val.jsonl"

    if os.path.exists(train_path):
        print(f"  already exists: {train_path} — skipping")
        return

    print(f"  downloading up to {max_samples:,} articles…")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.ne",
        split="train", streaming=True, trust_remote_code=True,
    )

    lines = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        text = row["text"].strip()
        if len(text) < 50:
            continue
        lines.append(json.dumps({"text": text}, ensure_ascii=False))
        if (i + 1) % 10_000 == 0:
            print(f"  {i + 1:,} articles loaded")

    random.shuffle(lines)
    n_val = max(1, int(len(lines) * val_ratio))

    with open(val_path,   "w", encoding="utf-8") as f:
        f.write("\n".join(lines[:n_val]))
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines[n_val:]))

    print(f"  {len(lines) - n_val:,} train + {n_val:,} val → data/corpus/")


# ── rendered text ────────────────────────────────────────────────────────────────

def prepare_rendered(max_samples: int = 100_000):
    """Download HuggingFaceM4/RenderedText → JSONL."""
    print("\n=== RenderedText ===")
    from src.data.hf_dataset import load_rendered_text, write_jsonl
    from src.data.ocr_dataset import generate_split

    out_dir  = "data/documents/rendered_text"
    jsonl    = f"{out_dir}.jsonl"
    if os.path.exists(jsonl):
        print(f"  already exists: {jsonl} — skipping")
        return

    records = load_rendered_text(max_samples=max_samples, cache_dir="data/hf_cache")
    write_jsonl(records, jsonl)
    generate_split(jsonl, val_ratio=0.05)


def prepare_cord():
    """Download CORD-v2 receipt dataset → JSONL."""
    print("\n=== CORD-v2 (receipts / tables) ===")
    from src.data.hf_dataset import load_cord, write_jsonl
    from src.data.ocr_dataset import generate_split

    for split in ("train", "validation"):
        tag   = "val" if split == "validation" else "train"
        jsonl = f"data/documents/cord_{tag}.jsonl"
        if os.path.exists(jsonl):
            print(f"  already exists: {jsonl} — skipping")
            continue
        records = load_cord(cache_dir="data/hf_cache", split=split)
        write_jsonl(records, jsonl)

    # Merge train+val into one file, then re-split deterministically
    _merge_and_split("data/documents/cord_train.jsonl",
                     "data/documents/cord_val.jsonl",
                     "data/documents/cord.jsonl")


def prepare_iam():
    """Download IAM handwriting dataset → JSONL."""
    print("\n=== IAM handwriting ===")
    from src.data.hf_dataset import load_iam, write_jsonl
    from src.data.ocr_dataset import generate_split

    jsonl = "data/documents/iam.jsonl"
    if os.path.exists(jsonl):
        print(f"  already exists: {jsonl} — skipping")
        return

    records = load_iam(cache_dir="data/hf_cache", split="train")
    write_jsonl(records, jsonl)
    generate_split(jsonl, val_ratio=0.05)


# ── synthetic Nepali ─────────────────────────────────────────────────────────────

def prepare_synth(
    font_path: str,
    n_samples: int = 100_000,
    img_size: int = 448,
):
    """Generate synthetic Nepali document images from the corpus."""
    print("\n=== Synthetic Nepali documents ===")
    from src.data.synth_data import generate_dataset
    from src.data.ocr_dataset import generate_split

    out_dir = "data/documents/nepali_synth"
    jsonl   = f"{out_dir}/data.jsonl"
    if os.path.exists(jsonl):
        print(f"  already exists: {jsonl} — skipping")
        return

    corpus = "data/corpus/train.jsonl"
    if not os.path.exists(corpus):
        print("  ERROR: run --stage corpus first to get Nepali text")
        return

    # Extract short text snippets from the corpus
    texts = []
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)["text"]
            # Split into sentences / short phrases
            for sent in article.replace("।", "।\n").splitlines():
                sent = sent.strip()
                if 10 < len(sent) < 120:
                    texts.append(sent)
            if len(texts) >= 50_000:
                break

    print(f"  {len(texts):,} text snippets extracted from corpus")

    jsonl_out = generate_dataset(
        out_dir,
        n_samples=n_samples,
        texts=texts,
        font_path=font_path,
        img_size=img_size,
        seed=42,
    )
    generate_split(jsonl_out, val_ratio=0.05)


# ── merge JSONL files for combined training ──────────────────────────────────────

def merge_ocr_datasets(output: str = "data/documents/train.jsonl"):
    """
    Merge all downloaded OCR JSONL files into one train and one val file.

    Call this after running all individual prepare_* stages.
    """
    print("\n=== Merging OCR datasets ===")

    sources = [
        ("data/documents/rendered_text_train.jsonl", "data/documents/rendered_text_val.jsonl"),
        ("data/documents/cord.jsonl",                None),   # already split
        ("data/documents/iam_train.jsonl",           "data/documents/iam_val.jsonl"),
        ("data/documents/nepali_synth/data_train.jsonl", "data/documents/nepali_synth/data_val.jsonl"),
    ]

    train_lines, val_lines = [], []
    for train_src, val_src in sources:
        if os.path.exists(train_src):
            with open(train_src, encoding="utf-8") as f:
                train_lines += [l for l in f if l.strip()]
            print(f"  train ← {train_src}")
        if val_src and os.path.exists(val_src):
            with open(val_src, encoding="utf-8") as f:
                val_lines += [l for l in f if l.strip()]
            print(f"  val   ← {val_src}")

    rng = random.Random(42)
    rng.shuffle(train_lines)
    rng.shuffle(val_lines)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    val_output = output.replace("train", "val")

    with open(output,     "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with open(val_output, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    print(f"\n  {len(train_lines):,} train → {output}")
    print(f"  {len(val_lines):,}  val  → {val_output}")


# ── helpers ──────────────────────────────────────────────────────────────────────

def _merge_and_split(train_path, val_path, merged_path):
    lines = []
    for p in (train_path, val_path):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                lines += [l for l in f if l.strip()]
    random.Random(42).shuffle(lines)
    from src.data.ocr_dataset import generate_split
    with open(merged_path, "w") as f:
        f.writelines(lines)
    generate_split(merged_path, val_ratio=0.05)


# ── CLI ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for Akshara")
    parser.add_argument("--stage", choices=["corpus", "rendered", "cord", "iam", "synth", "merge", "all"],
                        default="all")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of samples per dataset (default: full dataset)")
    parser.add_argument("--font_path", type=str, default=None,
                        help="Path to NotoSansDevanagari-Regular.ttf (required for --stage synth)")
    parser.add_argument("--n_synth", type=int, default=100_000,
                        help="Number of synthetic Nepali samples to generate")
    parser.add_argument("--img_size", type=int, default=448)
    args = parser.parse_args()

    stages = (
        ["corpus", "rendered", "cord", "iam"]
        if args.stage == "all"
        else [args.stage]
    )

    for stage in stages:
        if stage == "corpus":
            prepare_corpus(max_samples=args.max_samples or 200_000)
        elif stage == "rendered":
            prepare_rendered(max_samples=args.max_samples or 100_000)
        elif stage == "cord":
            prepare_cord()
        elif stage == "iam":
            prepare_iam()
        elif stage == "synth":
            if not args.font_path:
                print("ERROR: --font_path required for synth stage")
                print("  brew install font-noto-sans-devanagari")
                print("  then: python scripts/prepare_data.py --stage synth --font_path /path/to/NotoSansDevanagari-Regular.ttf")
                return
            prepare_synth(args.font_path, n_samples=args.n_synth, img_size=args.img_size)
        elif stage == "merge":
            merge_ocr_datasets()

    if args.stage in ("all", "merge"):
        merge_ocr_datasets()

    print("\nDone. Next steps:")
    print("  1. python scripts/pretrain.py --config configs/pretrain.json")
    print("  2. python scripts/train_ocr.py --config configs/ocr_finetune.json \\")
    print("         --pretrain_ckpt checkpoints/pretrain.pt")


if __name__ == "__main__":
    main()
