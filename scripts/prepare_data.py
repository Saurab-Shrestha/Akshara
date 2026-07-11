"""
Data preparation for Akshara — SmolLM-style quality-centric approach.

Downloads and mixes multilingual text corpora with education-quality filtering.

SOURCES:
    FineWeb-2:    Multilingual web with quality scores.
                  Primary source for Devanagari script languages.
    FineWeb-Edu:  Education-filtered English web (EduScore >= 3).
    Wikipedia:    Clean articles for all target languages (supplement).

QUALITY FILTER:
    FineWeb-2: length-filtered only. FineWeb-2 exposes no single EduScore
               field, so QUALITY_THRESHOLD does not apply here.
    FineWeb-Edu: score >= QUALITY_THRESHOLD (EduScore 0-5).
    Wikipedia: no filtering (already curated).

DATA MIX (1M docs default):
    70%  FineWeb-2 Devanagari (ne, hi, mr, sa)
    20%  FineWeb-Edu          (English, high-quality)
    10%  Wikipedia            (mixed ne, hi, en, mr, sa)

USAGE:
    # Default 1M docs
    PYTHONPATH=. python scripts/prepare_data.py

    # Custom size
    PYTHONPATH=. python scripts/prepare_data.py --max_samples 500000 --out_dir data/corpus_v4

    # Custom quality threshold
    PYTHONPATH=. python scripts/prepare_data.py --quality 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

# Minimum EduScore quality threshold (0–5, higher = more educational)
QUALITY_THRESHOLD = 3

# FineWeb-2 configs for Devanagari script languages
_FINEWEB2_DEVANAGARI = {
    "ne": "npi_Deva",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "sa": "san_Deva",
}

# Wikipedia language codes
_WIKI_CONFIGS = {
    "ne": "20231101.ne",
    "hi": "20231101.hi",
    "en": "20231101.en",
    "mr": "20231101.mr",
    "sa": "20231101.sa",
}


def _load_fineweb2(config: str, max_samples: int) -> list[str]:
    """Load documents from FineWeb-2 (length-filtered; no EduScore field)."""
    from datasets import load_dataset

    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-2", config, split="train", streaming=True,
        )
    except Exception as e:
        print(f"    [warn] cannot load FineWeb-2 '{config}': {e}")
        return []

    texts: list[str] = []
    it = iter(ds)
    i = 0
    while len(texts) < max_samples:
        try:
            sample = next(it)
        except StopIteration:
            break
        except Exception as e:
            if "Cast" in type(e).__name__ or "Schema" in type(e).__name__:
                continue
            raise
        i += 1
        text = sample.get("text")
        if text is None:
            continue
        text = text.strip()
        if len(text) >= 50:
            texts.append(text)
        if i % 10000 == 0:
            print(f"    ... {i} scanned, {len(texts)} kept")
    return texts


def _load_fineweb_edu(
    max_samples: int, min_quality: int,
) -> list[str]:
    """Load English documents from FineWeb-Edu filtered by EduScore."""
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-100BT",
        split="train", streaming=True,
    )
    texts: list[str] = []
    for sample in ds:
        if len(texts) >= max_samples:
            break
        score = sample.get("score")
        if score is not None and score < min_quality:
            continue
        text = (sample.get("text") or "").strip()
        if len(text) >= 50:
            texts.append(text)
    return texts


def _load_wiki(lang: str, max_samples: int) -> list[str]:
    """Load Wikipedia articles for a language."""
    from datasets import load_dataset

    config = _WIKI_CONFIGS.get(lang)
    if config is None:
        return []

    ds = load_dataset(
        "wikimedia/wikipedia", config, split="train", streaming=True
    )
    texts: list[str] = []
    for sample in ds:
        if len(texts) >= max_samples:
            break
        text = (sample.get("text") or "").strip()
        if len(text) >= 50:
            texts.append(text)
    return texts


def _write_jsonl(texts: list[str], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def prepare_corpus(
    max_samples: int = 200_000,
    seed: int = 42,
    val_ratio: float = 0.05,
    out_dir: str = "data/corpus",
    quality: int = QUALITY_THRESHOLD,
):
    """
    Download a multilingual corpus with SmolLM-style quality filtering.

    Data mix:
        - 70% FineWeb-2 Devanagari (ne, hi, mr, sa)
        - 20% FineWeb-Edu English
        - 10% Wikipedia (multilingual)
    """
    rng = random.Random(seed)
    out = Path(out_dir)

    n_fw2 = int(max_samples * 0.70)
    n_edu = int(max_samples * 0.20)
    n_wiki = int(max_samples * 0.10)

    all_texts: list[str] = []

    # --- FineWeb-2 Devanagari ---
    fw2_targets = {
        "ne (Nepali)": int(n_fw2 * 0.38),
        "hi (Hindi)": int(n_fw2 * 0.38),
        "mr (Marathi)": int(n_fw2 * 0.16),
        "sa (Sanskrit)": int(n_fw2 * 0.08),
    }
    for label, target in fw2_targets.items():
        lang_code = label.split(" ")[0]
        config = _FINEWEB2_DEVANAGARI.get(lang_code)
        if config is None or target == 0:
            continue
        print(f"  [FineWeb-2 {label}] loading {target:,} docs...")
        try:
            texts = _load_fineweb2(config, target)
            print(f"    → got {len(texts):,} docs")
            all_texts.extend(texts)
        except Exception as e:
            print(f"    [warn] failed to load FineWeb-2 {config}: {e}")
            continue

    # --- FineWeb-Edu English ---
    print(f"  [FineWeb-Edu] loading {n_edu:,} docs (quality>={quality})...")
    edu_texts = _load_fineweb_edu(n_edu, min_quality=quality)
    print(f"    → got {len(edu_texts):,} docs")
    all_texts.extend(edu_texts)

    # --- Wikipedia multilingual ---
    wiki_langs = {"ne": 0.3, "hi": 0.25, "en": 0.25, "mr": 0.1, "sa": 0.1}
    for lang_code, ratio in wiki_langs.items():
        target = int(n_wiki * ratio)
        if target == 0:
            continue
        print(f"  [Wikipedia {lang_code}] loading {target:,} docs...")
        texts = _load_wiki(lang_code, target)
        print(f"    → got {len(texts):,} docs")
        all_texts.extend(texts)

    print(f"\n[data] total collected: {len(all_texts):,} docs")

    # Trim to exact max_samples if oversampled
    if len(all_texts) > max_samples:
        rng.shuffle(all_texts)
        all_texts = all_texts[:max_samples]

    rng.shuffle(all_texts)

    n_val = max(1, int(len(all_texts) * val_ratio))
    val_docs = all_texts[:n_val]
    train_docs = all_texts[n_val:]

    _write_jsonl(train_docs, str(out / "train.jsonl"))
    _write_jsonl(val_docs, str(out / "val.jsonl"))
    print(f"[data] done — {len(train_docs)} train / {len(val_docs)} val → {out}/")


def main():
    parser = argparse.ArgumentParser(
        description="Akshara data preparation (SmolLM-style)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=1_000_000,
        help="Total target document count (default: 1M)",
    )
    parser.add_argument(
        "--out_dir", type=str, default="data/corpus",
        help="Output directory for the corpus JSONL files",
    )
    parser.add_argument(
        "--quality", type=int, default=QUALITY_THRESHOLD,
        help="Minimum EduScore quality threshold 0-5 (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split",
    )
    args = parser.parse_args()

    prepare_corpus(
        max_samples=args.max_samples,
        seed=args.seed,
        out_dir=args.out_dir,
        quality=args.quality,
    )


if __name__ == "__main__":
    main()
