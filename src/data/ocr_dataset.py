"""
Crop OCR Dataset — text-region image → plain text
===================================================

DESIGN (post full-page pivot)
------------------------------
The recognizer reads CROPS (a line, a paragraph, a heading, a table cell)
and outputs the plain text in them.  Structure (<p>, <h1>, <table>) comes
from the layout/table-structure models in the inference pipeline, NOT from
this model — the recognizer just reads.

PREPROCESSING — aspect-preserving pad, never squash
----------------------------------------------------
Crops have wild aspect ratios (a line is 20:1, a cell may be 1:2).  We
resize preserving aspect ratio so the crop fits inside a square canvas,
then pad with white ("empty paper").  A squashed line is unreadable;
a padded one is not.
    # ponytail: fixed 448 canvas — Pix2Struct-style variable patch budget
    # is the upgrade path when padding waste hurts throughput.

LOSS MASKING — no EOS spam
---------------------------
Targets are padded with -100 (cross_entropy ignore_index) after the first
EOS, so the model is supervised on real text + ONE stop signal, never on
"predict EOS given EOS" filler.

DATA FORMAT (JSONL)
--------------------
Each line: {"image": "path/to/crop.png", "text": "नेपाली पाठ"}
"image_path" is accepted as an alias for "image"; "html" as alias for
"text" (legacy files).  Paths may be absolute or relative to the JSONL.

A directory of NNN.png + NNN.txt (or .html) pairs is also supported.
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


def pad_to_square(img: Image.Image, size: int) -> Image.Image:
    """Aspect-preserving resize onto a white square canvas (never squash)."""
    w, h = img.size
    scale = size / max(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, (0, 0))  # top-left anchor: reading starts there
    return canvas


class CropOCRDataset(Dataset):
    """
    Paired (crop image, text string) dataset.

    Supports two source formats:
      1. JSONL file: each line is {"image": "path.png", "text": "…"}
         ("image_path"/"html" accepted as aliases)
      2. Directory:  pairs of image.png + image.txt (or .html) files

    Args:
        source:      Path to a JSONL file or a directory of pairs.
        tokenizer:   HuggingFace tokenizer (must have eos token id).
        img_size:    Square canvas size (aspect-preserving pad, no squash).
        max_seq_len: Maximum text token sequence length (truncated if longer).
    """

    def __init__(
        self,
        source: str,
        tokenizer,
        img_size: int = 448,
        max_seq_len: int = 512,
    ):
        self.tokenizer   = tokenizer
        self.img_size    = img_size
        self.max_seq_len = max_seq_len

        self.samples: List[Tuple[str, str]] = []  # (image_path, text)
        self._load_source(source)

        self._to_tensor = transforms.Compose([
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
                img_path = obj.get("image") or obj.get("image_path")
                text     = obj.get("text")  or obj.get("html")
                if not img_path or text is None:
                    raise KeyError(f"line missing image/text keys: {line[:100]}")
                if not os.path.isabs(img_path):
                    img_path = str(base / img_path)
                self.samples.append((img_path, text))

    def _load_directory(self, root: Path):
        for img_path in sorted(root.glob("*.png")):
            for suffix in (".txt", ".html"):
                label = img_path.with_suffix(suffix)
                if label.exists():
                    self.samples.append((str(img_path), label.read_text(encoding="utf-8")))
                    break
            else:
                print(f"[ocr_dataset] warning: no label for {img_path.name}, skipping")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text = self.samples[idx]

        img    = Image.open(img_path).convert("RGB")
        tensor = self._to_tensor(pad_to_square(img, self.img_size))

        # text → token ids: [BOS] t1 … tN [EOS], then pad
        eos = self.tokenizer.eos_token_id
        bos = self.tokenizer.bos_token_id or eos  # Qwen has no BOS; use EOS as sentinel

        ids  = self.tokenizer.encode(text, add_special_tokens=False)
        ids  = ids[: self.max_seq_len - 1]  # leave room for EOS
        full = [bos] + ids + [eos]

        # Inputs pad with EOS (a valid token the model can embed);
        # targets pad with -100 so the loss ignores every position after
        # the first EOS.
        pad_len    = (self.max_seq_len + 1) - len(full)
        input_ids  = torch.tensor((full + [eos]  * pad_len)[:-1], dtype=torch.long)
        target_ids = torch.tensor((full + [-100] * pad_len)[1:],  dtype=torch.long)

        return tensor, input_ids, target_ids


# Backwards-compatible alias (old scripts import DocumentOCRDataset)
DocumentOCRDataset = CropOCRDataset


def get_ocr_dataloader(
    source: str,
    tokenizer,
    batch_size: int,
    img_size: int = 448,
    max_seq_len: int = 512,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    ds = CropOCRDataset(source, tokenizer, img_size, max_seq_len)
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

    # Trailing newline matters: merge scripts concatenate these files
    train_path.write_text("\n".join(train) + "\n")
    val_path.write_text("\n".join(val) + "\n")

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
