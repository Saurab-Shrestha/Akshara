"""
Akshara inference pipeline — page image → structured HTML
==========================================================

DESIGN
------
Structure is language-agnostic; reading is not. So:

  page image
    → Surya layout detection      (pretrained — finds paragraphs/headings/tables)
    → Surya table recognition     (pretrained — cell grid for table regions)
    → crop each region/cell
    → Akshara recognizer          (OUR model — reads Nepali text in each crop)
    → HTML assembly               (plain Python — region class → tag)

The recognizer never sees structure; the structure models never read text.

USAGE
-----
    from src.pipeline import AksharaPipeline
    pipe = AksharaPipeline("checkpoints/ocr.pt")
    html = pipe.process(Image.open("page.png"))

Surya is imported lazily — everything except .process() works without it.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.data.ocr_dataset import pad_to_square, _IMAGENET_MEAN, _IMAGENET_STD
from src.models.vlm import Akshara

# Surya layout labels → HTML tags. Unlisted labels (PageFooter, Picture…) are skipped.
_TAG = {
    "Title":         "h1",
    "SectionHeader": "h2",
    "Text":          "p",
    "ListItem":      "li",
    "Caption":       "p",
    "Footnote":      "p",
}


class AksharaPipeline:

    def __init__(self, ckpt_path: str, device: str = "cpu", batch_size: int = 8):
        self.device     = device
        self.batch_size = batch_size

        ck  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck.get("config")
        if cfg is None:
            raise ValueError(f"{ckpt_path} has no embedded config — cannot build model")

        model_keys = ("img_size", "patch_size", "vision_dim", "vit_layers", "vit_heads",
                      "vocab_size", "n_embed", "n_heads", "n_kv_heads", "n_layers",
                      "max_seq_len", "attn_every")
        # vit_pretrained=False: weights come from the checkpoint — no need to
        # download DINOv2 just to overwrite it
        self.model = Akshara(**{k: cfg[k] for k in model_keys if k in cfg},
                             vit_pretrained=False)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.to(device).eval()
        self.img_size = cfg.get("img_size", 448)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

        self._layout    = None  # lazy — loaded on first .process()
        self._table_rec = None

        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
        self._norm = lambda t: (t - mean) / std

    # ── recognition ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def read_crops(self, crops: list[Image.Image], max_new: int = 256) -> list[str]:
        """OCR a list of PIL crops → list of text strings (batched)."""
        eos = self.tokenizer.eos_token_id
        bos = self.tokenizer.bos_token_id or eos
        texts = []
        for i in range(0, len(crops), self.batch_size):
            batch = crops[i : i + self.batch_size]
            tensors = torch.stack([
                self._norm(torch.from_numpy(
                    np.asarray(pad_to_square(c.convert("RGB"), self.img_size)).copy()
                ).permute(2, 0, 1).float() / 255.0)
                for c in batch
            ]).to(self.device)
            out = self.model.generate(tensors, bos_token_id=bos, eos_token_id=eos,
                                      max_new_tokens=max_new, temperature=0.0)
            for row in out:
                ids = [t for t in row.tolist() if t not in (bos, eos)]
                texts.append(self.tokenizer.decode(ids, skip_special_tokens=True).strip())
        return texts

    # ── layout + assembly ─────────────────────────────────────────────────────

    def _load_surya(self):
        if self._layout is None:
            from surya.layout import LayoutPredictor
            self._layout = LayoutPredictor()
        if self._table_rec is None:
            try:
                from surya.table_rec import TableRecPredictor
                self._table_rec = TableRecPredictor()
            except Exception:
                self._table_rec = False  # tables degrade to a single <p>

    def process(self, page: Image.Image) -> str:
        """Full page → HTML string."""
        self._load_surya()
        page = page.convert("RGB")
        layout = self._layout([page])[0]

        # Reading order: Surya layout boxes carry .position when the reading-order
        # model runs; fall back to top-to-bottom, left-to-right.
        boxes = sorted(
            layout.bboxes,
            key=lambda b: getattr(b, "position", None) or (b.bbox[1], b.bbox[0]),
        )

        # First pass: gather all non-table crops for one batched read
        pending = []   # (index in parts, crop)
        parts   = []   # HTML fragments in reading order
        for b in boxes:
            label = b.label
            crop  = page.crop([int(v) for v in b.bbox])
            if label == "Table":
                parts.append(self._read_table(crop))
            elif label in _TAG:
                parts.append(None)
                pending.append((len(parts) - 1, crop, _TAG[label]))
            # else: Picture/PageFooter/etc — skipped

        texts = self.read_crops([c for _, c, _ in pending])
        for (idx, _, tag), text in zip(pending, texts):
            parts[idx] = f"<{tag}>{text}</{tag}>" if text else ""

        return "\n".join(p for p in parts if p)

    def _read_table(self, table_crop: Image.Image) -> str:
        if not self._table_rec:
            # No table model available — read the whole table as one block
            text = self.read_crops([table_crop])[0]
            return f"<p>{text}</p>" if text else ""

        result = self._table_rec([table_crop])[0]
        cells  = result.cells  # each: .bbox, .row_id, .col_id
        if not cells:
            return ""

        crops = [table_crop.crop([int(v) for v in c.bbox]) for c in cells]
        texts = self.read_crops(crops, max_new=64)

        rows: dict[int, dict[int, str]] = {}
        for cell, text in zip(cells, texts):
            rows.setdefault(cell.row_id, {})[cell.col_id] = text

        html = ["<table>"]
        for r in sorted(rows):
            tds = "".join(f"<td>{rows[r][c]}</td>" for c in sorted(rows[r]))
            html.append(f"<tr>{tds}</tr>")
        html.append("</table>")
        return "\n".join(html)
