"""
HuggingFace Dataset Loaders — pre-training on real document data
=================================================================

WHY PRE-TRAIN ON THESE BEFORE NEPALI SYNTHETIC DATA
-----------------------------------------------------
Our model needs to learn two things:
  1. What documents look like visually (layout, fonts, noise)
  2. How to produce structured HTML output

Real English document datasets teach (1) and (2) simultaneously with high
quality labels.  Synthetic Nepali data then fine-tunes the model on the
specific script.  Training order: HF datasets → Nepali synthetic.

DATASETS
---------
  Docmatix (HuggingFaceM4/Docmatix)
    2.4M document page images + Q&A pairs derived from PDFs.
    We use the plain text answers as our HTML target (wrapped in <p> tags).
    Size: ~70GB.  Use max_samples to cap during development.

  RenderedText (HuggingFaceM4/RenderedText)
    Rendered text images with ground truth strings.  Clean, fast to load.
    Good for bootstrapping visual-text alignment before complex documents.

  CORD-v2 (naver-clova-ix/cord-v2)
    ~11k Indonesian receipt images with structured JSON annotations.
    We convert the JSON to simple HTML tables — good for table training.

  IAM Handwriting (IAM-community/iam-handwriting)
    Handwritten English text lines + transcriptions.
    Wraps each line as <p> — teaches the model to read non-printed text.

USAGE
------
    from src.data.hf_dataset import load_docmatix, write_jsonl

    samples = load_docmatix(max_samples=10_000)
    write_jsonl(samples, "data/documents/docmatix_train.jsonl")

Each loader returns a list of dicts:
    [{"image_path": "...", "html": "<p>...</p>"}, ...]

where image_path is an absolute path to a saved PNG.

STORAGE NOTE
-------------
Images are saved to a local cache directory (data/hf_cache/ by default).
Docmatix at full scale is ~70GB — use max_samples=50_000 for a
manageable pre-training set (~3.5GB).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Optional

_DEFAULT_CACHE = "data/hf_cache"


def write_jsonl(records: List[Dict], path: str) -> str:
    """Write a list of {image_path, html} dicts to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records → {path}")
    return path


def load_rendered_text(
    max_samples: Optional[int] = None,
    cache_dir: str = _DEFAULT_CACHE,
    split: str = "train",
) -> List[Dict]:
    """
    Load HuggingFaceM4/RenderedText — rendered text images with ground truth.

    WHY START HERE: Small, fast, clean.  Good first dataset to verify the
    vision encoder is learning visual-text alignment before scaling up.

    Returns list of {"image_path": str, "html": str}.
    """
    from datasets import load_dataset
    from PIL import Image as PILImage

    img_dir = Path(cache_dir) / "rendered_text"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceM4/RenderedText", split=split, streaming=True)

    records = []
    for i, sample in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        img_path = str(img_dir / f"{i:07d}.png")
        if not os.path.exists(img_path):
            sample["image"].convert("RGB").save(img_path)

        text = sample.get("text", sample.get("caption", "")).strip()
        html = f"<p>{text}</p>" if text else "<p></p>"
        records.append({"image_path": img_path, "html": html})

        if (i + 1) % 1000 == 0:
            print(f"  rendered_text: {i + 1} loaded")

    print(f"Loaded {len(records)} RenderedText samples")
    return records


def load_docmatix(
    max_samples: Optional[int] = None,
    cache_dir: str = _DEFAULT_CACHE,
    split: str = "train",
) -> List[Dict]:
    """
    Load HuggingFaceM4/Docmatix — PDF page images with text content.

    Each sample has multiple Q&A pairs per page.  We take the concatenated
    answers as the text content and wrap in <p> tags.

    WHY: 2.4M real document images — the largest freely available document
    OCR pre-training set.  Teaches the model real-world document appearance
    (scanned noise, mixed fonts, tables, letterheads).

    Returns list of {"image_path": str, "html": str}.
    """
    from datasets import load_dataset

    img_dir = Path(cache_dir) / "docmatix"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceM4/Docmatix", split=split, streaming=True)

    records = []
    for i, sample in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        img_path = str(img_dir / f"{i:07d}.png")
        if not os.path.exists(img_path):
            sample["images"][0].convert("RGB").save(img_path)

        # Concatenate all answers into paragraphs
        texts = []
        for qa in sample.get("texts", []):
            ans = qa.get("answer", "").strip()
            if ans:
                texts.append(ans)
        html = "".join(f"<p>{t}</p>" for t in texts) if texts else "<p></p>"

        records.append({"image_path": img_path, "html": html})

        if (i + 1) % 1000 == 0:
            print(f"  docmatix: {i + 1} loaded")

    print(f"Loaded {len(records)} Docmatix samples")
    return records


def load_cord(
    cache_dir: str = _DEFAULT_CACHE,
    split: str = "train",
) -> List[Dict]:
    """
    Load naver-clova-ix/cord-v2 — receipt images with structured JSON.

    CORD has ~11k Korean/English receipt images with ground truth JSON that
    includes line items, prices, and totals.  We convert to HTML tables.

    WHY: The only freely available labeled receipt/table dataset.  Training
    on CORD teaches the model to produce <table> HTML for structured documents
    like invoices and forms — directly applicable to Nepali government forms.

    Returns list of {"image_path": str, "html": str}.
    """
    from datasets import load_dataset

    img_dir = Path(cache_dir) / "cord"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("naver-clova-ix/cord-v2", split=split)

    records = []
    for i, sample in enumerate(ds):
        img_path = str(img_dir / f"{i:06d}.png")
        if not os.path.exists(img_path):
            sample["image"].convert("RGB").save(img_path)

        html = _cord_to_html(sample.get("ground_truth", "{}"))
        records.append({"image_path": img_path, "html": html})

    print(f"Loaded {len(records)} CORD samples")
    return records


def _cord_to_html(gt_str: str) -> str:
    """Convert CORD JSON ground truth to simple HTML table."""
    try:
        gt = json.loads(gt_str)
    except (json.JSONDecodeError, TypeError):
        return "<p></p>"

    valid_line = gt.get("valid_line", [])
    if not valid_line:
        # Fallback: just grab all text
        words = gt.get("gt_parse", {}).get("menu", [])
        texts = [str(w) for w in words if w]
        return "".join(f"<p>{t}</p>" for t in texts) if texts else "<p></p>"

    # Build a table from line items
    rows = []
    for line in valid_line:
        words = [w.get("text", "") for w in line.get("words", [])]
        if words:
            rows.append(words)

    if not rows:
        return "<p></p>"

    # First row as header if it looks like a header
    header = rows[0]
    body   = rows[1:]
    th = "".join(f"<th>{h}</th>" for h in header)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in body)
    return f"<table><tr>{th}</tr>{tr}</table>"


def load_iam(
    cache_dir: str = _DEFAULT_CACHE,
    split: str = "train",
) -> List[Dict]:
    """
    Load IAM handwriting dataset — handwritten English text lines.

    WHY: Teaches the model to handle non-printed, variable-quality text.
    Handwriting robustness transfers to degraded printed text (old newspapers,
    faded government stamps common in Nepali documents).

    Returns list of {"image_path": str, "html": str}.
    """
    from datasets import load_dataset

    img_dir = Path(cache_dir) / "iam"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("Teklia/IAM-line", split=split)

    records = []
    for i, sample in enumerate(ds):
        img_path = str(img_dir / f"{i:06d}.png")
        if not os.path.exists(img_path):
            sample["image"].convert("RGB").save(img_path)

        text = sample.get("text", "").strip()
        html = f"<p>{text}</p>" if text else "<p></p>"
        records.append({"image_path": img_path, "html": html})

    print(f"Loaded {len(records)} IAM samples")
    return records


# ── self-check ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("hf_dataset self-check — CORD JSON → HTML conversion")
    print("=" * 50)

    # Test _cord_to_html without downloading anything
    sample_gt = json.dumps({
        "valid_line": [
            {"words": [{"text": "Item"}, {"text": "Price"}]},
            {"words": [{"text": "Rice"}, {"text": "200"}]},
            {"words": [{"text": "Total"}, {"text": "200"}]},
        ]
    })
    html = _cord_to_html(sample_gt)
    print(f"  input:  {sample_gt}")
    print(f"  output: {html}")
    assert "<table>" in html
    assert "<th>Item</th>" in html
    assert "<td>Rice</td>" in html
    print("  ✅ CORD → HTML conversion correct")

    # Test write_jsonl
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        records = [{"image_path": "/tmp/a.png", "html": "<p>test</p>"}]
        path = write_jsonl(records, f"{tmp}/out.jsonl")
        with open(path) as f:
            loaded = [json.loads(l) for l in f]
        assert loaded == records
        print("  ✅ write_jsonl round-trips correctly")

    print("\nSelf-check PASSED ✅")
    print("\nTo download datasets:")
    print("  from src.data.hf_dataset import load_rendered_text, write_jsonl")
    print("  records = load_rendered_text(max_samples=10_000)")
    print("  write_jsonl(records, 'data/documents/rendered_text.jsonl')")
