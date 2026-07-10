"""
Synthetic Document Generator — full page images with HTML ground truth
=======================================================================

WHY SYNTHETIC DATA FIRST
--------------------------
Real labeled Nepali document datasets barely exist publicly.  Synthetic data
lets us generate unlimited training samples with perfect ground truth by
rendering text programmatically:

  render(html_structure) → (PIL Image, html_string)

The model never sees the rendering code — it only sees the image and must
predict the HTML.  This is exactly the training signal we need.

FONT REQUIREMENT
-----------------
PIL's default font does NOT render Devanagari — characters appear as empty
boxes.  You must provide a TrueType font that supports Devanagari:

  Recommended: Noto Sans Devanagari
  Download:    https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari
  Or install:  brew install font-noto-sans-devanagari  (macOS)
               apt install fonts-noto  (Ubuntu)

If no font_path is given, falls back to PIL default (ASCII only — useful
for smoke tests with English text).

DOCUMENT TYPES GENERATED
--------------------------
  paragraph  — one or more <p> blocks (most common document type)
  heading    — <h1> followed by paragraphs
  table      — <table> with headers and rows (invoices, forms)
"""

from __future__ import annotations

import json
import os
import random
import textwrap
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont


def _load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def _line_height(font, sample: str = "Aम") -> int:
    try:
        bbox = font.getbbox(sample)
        return bbox[3] - bbox[1] + 4
    except AttributeError:
        return font.getsize(sample)[1] + 4


def _draw_wrapped(draw, xy, text, font, color, max_width):
    """Draw text with word-wrap. Returns y after last line."""
    try:
        avg_w = max(1, font.getlength("म") or font.getlength("a"))
    except AttributeError:
        avg_w = 10
    chars = max(1, int(max_width / avg_w))
    lh    = _line_height(font)
    x, y  = xy
    for para in text.split("\n"):
        for line in textwrap.wrap(para, width=chars) or [para]:
            draw.text((x, y), line, font=font, fill=color)
            y += lh
    return y


def _random_bg():
    v = random.randint(230, 255)
    return (v, v, v)


def _random_fg():
    v = random.randint(0, 40)
    return (v, v, v)


def _augment(img: Image.Image, seed: int) -> Image.Image:
    random.seed(seed + 9999)
    angle = random.uniform(-1.5, 1.5)
    if abs(angle) > 0.3:
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=_random_bg())
    r = random.uniform(0.0, 0.8)
    if r > 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=r))
    return img


def generate_paragraph_page(
    paragraphs: List[str],
    font_path: Optional[str] = None,
    img_size: int = 448,
    font_size: int = 18,
    seed: int = 0,
) -> Tuple[Image.Image, str]:
    """Render paragraphs. Returns (image, html)."""
    random.seed(seed)
    img  = Image.new("RGB", (img_size, img_size), _random_bg())
    draw = ImageDraw.Draw(img)
    font = _load_font(font_path, font_size)
    fg   = _random_fg()
    m    = int(img_size * 0.06)
    y    = m
    html_parts = []
    for p in paragraphs:
        y = _draw_wrapped(draw, (m, y), p, font, fg, img_size - 2 * m)
        y += int(font_size * 0.6)
        html_parts.append(f"<p>{p}</p>")
    return _augment(img, seed), "".join(html_parts)


def generate_heading_page(
    title: str,
    paragraphs: List[str],
    font_path: Optional[str] = None,
    img_size: int = 448,
    seed: int = 0,
) -> Tuple[Image.Image, str]:
    """Render heading + paragraphs. Returns (image, html)."""
    random.seed(seed)
    img    = Image.new("RGB", (img_size, img_size), _random_bg())
    draw   = ImageDraw.Draw(img)
    h1f    = _load_font(font_path, 26)
    pf     = _load_font(font_path, 18)
    fg     = _random_fg()
    m      = int(img_size * 0.06)
    y      = m
    y      = _draw_wrapped(draw, (m, y), title, h1f, fg, img_size - 2 * m)
    y     += int(26 * 0.8)
    html_parts = [f"<h1>{title}</h1>"]
    for p in paragraphs:
        y = _draw_wrapped(draw, (m, y), p, pf, fg, img_size - 2 * m)
        y += int(18 * 0.6)
        html_parts.append(f"<p>{p}</p>")
    return _augment(img, seed), "".join(html_parts)


def generate_table_page(
    headers: List[str],
    rows: List[List[str]],
    caption: Optional[str] = None,
    font_path: Optional[str] = None,
    img_size: int = 448,
    seed: int = 0,
) -> Tuple[Image.Image, str]:
    """Render a table with optional caption. Returns (image, html)."""
    random.seed(seed)
    img   = Image.new("RGB", (img_size, img_size), _random_bg())
    draw  = ImageDraw.Draw(img)
    font  = _load_font(font_path, 16)
    hfont = _load_font(font_path, 17)
    fg    = _random_fg()
    grid  = (150, 150, 150)
    m     = int(img_size * 0.06)
    y     = m
    html_parts = []

    if caption:
        y = _draw_wrapped(draw, (m, y), caption, _load_font(font_path, 20), fg, img_size - 2 * m)
        y += 10
        html_parts.append(f"<p>{caption}</p>")

    n_cols = len(headers)
    col_w  = (img_size - 2 * m) // max(1, n_cols)
    lh     = _line_height(font) + 6

    for ci, hdr in enumerate(headers):
        x = m + ci * col_w
        draw.rectangle([x, y, x + col_w, y + lh], outline=grid)
        draw.text((x + 4, y + 3), hdr[:20], font=hfont, fill=fg)
    y += lh

    for row in rows:
        for ci, cell in enumerate(row):
            x = m + ci * col_w
            draw.rectangle([x, y, x + col_w, y + lh], outline=grid)
            draw.text((x + 4, y + 3), str(cell)[:20], font=font, fill=fg)
        y += lh

    th = "".join(f"<th>{h}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    html_parts.append(f"<table><tr>{th}</tr>{tr}</table>")
    return _augment(img, seed), "".join(html_parts)


def generate_dataset(
    output_dir: str,
    n_samples: int,
    texts: List[str],
    font_path: Optional[str] = None,
    img_size: int = 448,
    seed: int = 0,
) -> str:
    """
    Generate a synthetic document dataset.

    Returns path to data.jsonl written in output_dir.
    Document type (paragraph / heading / table) is chosen randomly per sample.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng     = random.Random(seed)
    records = []

    for i in range(n_samples):
        s        = seed + i
        doc_type = rng.choice(["paragraph", "heading", "table"])

        if doc_type == "paragraph":
            paras = rng.sample(texts, min(3, len(texts)))
            img, html = generate_paragraph_page(paras, font_path, img_size, seed=s)

        elif doc_type == "heading":
            title = rng.choice(texts)
            rest  = [t for t in texts if t != title]
            paras = rng.sample(rest, min(2, len(rest)))
            img, html = generate_heading_page(title, paras, font_path, img_size, seed=s)

        else:  # table
            n_cols  = rng.randint(2, 4)
            headers = [rng.choice(texts)[:12] for _ in range(n_cols)]
            rows    = [[rng.choice(texts)[:12] for _ in range(n_cols)] for _ in range(rng.randint(2, 5))]
            caption = rng.choice(texts) if rng.random() > 0.5 else None
            img, html = generate_table_page(headers, rows, caption, font_path, img_size, seed=s)

        fname = f"{i:06d}.png"
        img.save(os.path.join(output_dir, fname))
        records.append({"image": fname, "html": html})

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_samples}")

    jsonl_path = os.path.join(output_dir, "data.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {n_samples} samples → {jsonl_path}")
    return jsonl_path


# ── self-check ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    TEXTS = [
        "नेपाल एक सुन्दर देश हो।",
        "काठमाडौं राजधानी शहर हो।",
        "हिमालय पर्वत श्रृंखला यहाँ छ।",
        "नेपाली भाषा देवनागरी लिपिमा लेखिन्छ।",
        "Total amount: Rs 500",
        "मूल्य", "वस्तु", "मात्रा",
    ]

    print("synth_data self-check (ASCII fallback — no Devanagari font)")
    print("=" * 55)

    with tempfile.TemporaryDirectory() as tmp:
        jsonl = generate_dataset(tmp, n_samples=6, texts=TEXTS, img_size=112, seed=7)

        with open(jsonl) as f:
            records = [json.loads(l) for l in f]

        assert len(records) == 6
        for r in records:
            img = Image.open(os.path.join(tmp, r["image"]))
            assert img.size == (112, 112)
            assert r["html"].strip()

        print(f"\n  {len(records)} samples  ✅")
        for r in records[:3]:
            print(f"    {r['html'][:80]}")

    print("\nSelf-check PASSED ✅")
    print("\nNOTE: pass font_path='NotoSansDevanagari-Regular.ttf' for Devanagari rendering")
