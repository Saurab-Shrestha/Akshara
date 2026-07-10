"""
Document OCR Dataset — full-page image → HTML
===============================================

WHY FULL PAGE INSTEAD OF LINE CROPS
-------------------------------------
Line-crop training forces you to run a separate text detector at inference
time and throws away all document structure (tables, headers, multi-column
layouts become unrecoverable).

Full-page training lets the model learn layout natively:
  - A table is a sequence of <table><tr><td>…</td></tr></table> tokens
  - A heading is a <h1>…</h1> token sequence
  - Columns are just <p> blocks ordered by reading sequence

The model sees the full document image once and generates structured HTML
that captures both content and layout.  No detector needed at inference.

OUTPUT FORMAT (HTML, no bboxes)
---------------------------------
We deliberately omit bounding box coordinates.  Surya includes them because
it needs to overlay text on PDFs.  We don't need that — we need readable
structured output.  Keeping the HTML simple means:
  - Easier to generate synthetically (no annotation required)
  - Shorter token sequences (fewer tokens per page)
  - Easier to post-process into plain text, markdown, or JSON

Example output for a Nepali invoice:
    <h1>बिल</h1>
    <p>मिति: २०८१-०३-१५</p>
    <table>
    <tr><th>वस्तु</th><th>मात्रा</th><th>मूल्य</th></tr>
    <tr><td>चामल</td><td>५ के.जी.</td><td>५०० रु</td></tr>
    </table>
    <p>जम्मा: ५०० रु</p>

DATA FORMAT (JSONL)
--------------------
Each line in the JSONL file is a JSON object:
    {"image": "path/to/image.png", "html": "<p>text</p>"}

The image path can be absolute or relative to the JSONL file's directory.

Alternatively, a directory of image+html pairs is also supported:
    root/
      001.png  +  001.html
      002.png  +  002.html

TRAIN / VAL SPLIT
------------------
Use the generate_split() helper to split a JSONL into train/val files.
Default 95/5 split (val is small because OCR eval is slow — CER decode
is sequential, not batched).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class DocumentOCRDataset(Dataset):
    """
    Paired (document image, HTML string) dataset.

    Supports two source formats:
      1. JSONL file: each line is {"image": "path.png", "html": "<p>…</p>"}
      2. Directory:  pairs of image.png + image.html files

    Args:
        source:      Path to a JSONL file or a directory of image+html pairs.
        tokenizer:   HuggingFace tokenizer (must have bos/eos token ids).
        img_size:    Resize all images to (img_size, img_size).
        max_seq_len: Maximum HTML token sequence length (truncated if longer).
    """

    def __init__(
        self,
        source: str,
        tokenizer,
        img_size: int = 448,
        max_seq_len: int = 2048,
    ):
        self.tokenizer   = tokenizer
        self.img_size    = img_size
        self.max_seq_len = max_seq_len

        self.samples: List[Tuple[str, str]] = []  # (image_path, html_string)
        self._load_source(source)

        self._transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def _load_source(self, source: str):
        p = Path(source)
        if p.is_file() and p.suffix in (".jsonl", ".json"):
            self._load_jsonl(p)
        elif p.is_dir():
            self._load_directory(p)
        else:
            raise ValueError(f"source must be a .jsonl file or directory, got: {source}")

    def _load_jsonl(self, path: Path):
        base = path.parent
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                img_path = obj["image"]
                if not os.path.isabs(img_path):
                    img_path = str(base / img_path)
                self.samples.append((img_path, obj["html"]))

    def _load_directory(self, root: Path):
        for img_path in sorted(root.glob("*.png")):
            html_path = img_path.with_suffix(".html")
            if html_path.exists():
                self.samples.append((str(img_path), html_path.read_text(encoding="utf-8")))
            else:
                print(f"[ocr_dataset] warning: no .html for {img_path.name}, skipping")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, html = self.samples[idx]

        # Image → tensor
        img    = Image.open(img_path).convert("RGB")
        tensor = self._transform(img)   # (3, img_size, img_size)

        # HTML → token ids
        # Sequence: [BOS] <html tokens> [EOS], padded to max_seq_len+1
        eos = self.tokenizer.eos_token_id
        bos = self.tokenizer.bos_token_id or eos  # Qwen has no BOS; use EOS as sentinel

        html_ids = self.tokenizer.encode(html, add_special_tokens=False)
        # Truncate leaving room for BOS + EOS
        html_ids = html_ids[: self.max_seq_len - 1]
        full = [bos] + html_ids + [eos]

        # Pad to max_seq_len + 1
        pad_len = (self.max_seq_len + 1) - len(full)
        full    = full + [eos] * pad_len

        full       = torch.tensor(full, dtype=torch.long)
        input_ids  = full[:-1]   # [BOS, t1, t2, ..., tN]
        target_ids = full[1:]    # [t1, t2, ..., tN, EOS/pad]

        return tensor, input_ids, target_ids


def get_ocr_dataloader(
    source: str,
    tokenizer,
    batch_size: int,
    img_size: int = 448,
    max_seq_len: int = 2048,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    ds = DocumentOCRDataset(source, tokenizer, img_size, max_seq_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def generate_split(
    jsonl_path: str,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Split a JSONL into train/val files.

    Returns:
        (train_path, val_path) — paths to the written split files.
    """
    import random
    rng = random.Random(seed)

    p = Path(jsonl_path)
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    rng.shuffle(lines)

    n_val  = max(1, int(len(lines) * val_ratio))
    val    = lines[:n_val]
    train  = lines[n_val:]

    train_path = p.parent / (p.stem + "_train.jsonl")
    val_path   = p.parent / (p.stem + "_val.jsonl")

    train_path.write_text("\n".join(train))
    val_path.write_text("\n".join(val))

    print(f"Split: {len(train)} train / {len(val)} val → {train_path}, {val_path}")
    return str(train_path), str(val_path)


# ── self-check ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    from PIL import ImageDraw

    class _Tok:
        bos_token_id = 1
        eos_token_id = 2
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 100 + 3 for c in text[:50]]
        def decode(self, ids, skip_special_tokens=True):
            return "decoded"

    print("DocumentOCRDataset self-check")
    print("=" * 40)

    tok = _Tok()

    with tempfile.TemporaryDirectory() as tmp:
        # Write 3 image+html pairs
        for i in range(3):
            img = Image.new("RGB", (300, 400), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Sample {i}", fill=(0, 0, 0))
            img.save(f"{tmp}/{i:03d}.png")
            with open(f"{tmp}/{i:03d}.html", "w") as f:
                f.write(f"<p>Sample {i} text here.</p>")

        # Directory mode
        ds = DocumentOCRDataset(tmp, tok, img_size=64, max_seq_len=32)
        assert len(ds) == 3, f"expected 3 samples, got {len(ds)}"
        img_t, inp, tgt = ds[0]
        assert img_t.shape == (3, 64, 64), f"image shape wrong: {img_t.shape}"
        assert inp.shape == (32,),  f"input_ids shape wrong: {inp.shape}"
        assert tgt.shape == (32,),  f"target_ids shape wrong: {tgt.shape}"
        print(f"  directory mode: {len(ds)} samples  image={tuple(img_t.shape)}  ids={tuple(inp.shape)}  ✅")

        # JSONL mode
        jsonl = f"{tmp}/data.jsonl"
        with open(jsonl, "w") as f:
            for i in range(3):
                f.write(json.dumps({"image": f"{tmp}/{i:03d}.png", "html": f"<h1>Page {i}</h1>"}) + "\n")

        ds2 = DocumentOCRDataset(jsonl, tok, img_size=64, max_seq_len=32)
        assert len(ds2) == 3
        print(f"  JSONL mode:     {len(ds2)} samples  ✅")

        # DataLoader
        dl = get_ocr_dataloader(jsonl, tok, batch_size=2, img_size=64, max_seq_len=32, num_workers=0)
        imgs, inp_b, tgt_b = next(iter(dl))
        assert imgs.shape   == (2, 3, 64, 64)
        assert inp_b.shape  == (2, 32)
        print(f"  DataLoader:     batch images={tuple(imgs.shape)}  ids={tuple(inp_b.shape)}  ✅")

    print("\nSelf-check PASSED ✅")
