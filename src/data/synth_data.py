"""
Synthetic crop generator — Nepali text → (crop image, text) pairs
==================================================================

THE CURRICULUM KNOB (docs/ARCHITECTURE.md §3)
----------------------------------------------
One generator, one knob:
    --max_lines 1     → Stage 2A line warmup (Pix2Struct-style reading stage)
    --max_lines 12    → Stage 2B paragraph crops

Crops are rendered at NATURAL aspect ratio (a line is wide, a paragraph is
blocky) — `CropOCRDataset.pad_to_square` handles canvasing at train time,
exactly like Surya layout crops will at inference time.

THE INVARIANT: the label is exactly what is drawn. Text that doesn't fit is
never silently clipped — we stop adding words/lines when the crop is full,
and the label stops with them.

AUGMENTATIONS (PIL-only, applied per crop with independent probabilities)
--------------------------------------------------------------------------
random font / size, ink & paper color jitter, Gaussian blur, pixel noise,
small rotation, JPEG re-compression, contrast jitter.
Pix2Struct's warmup used random fonts/colors/sizes; same idea, Nepali fonts.

USAGE
-----
    PYTHONPATH=. python src/data/synth_data.py \
        --corpus data/corpus/train.jsonl \
        --fonts fonts/ \
        --out data/crops/lines \
        --n 100000 --max_lines 1 --seed 42

Output: data/crops/lines/{000000.png, …} + data.jsonl ({"image","text"}).
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


# ── text sampling ──────────────────────────────────────────────────────────────

def load_corpus(path: str) -> list[str]:
    """Load text lines from a Stage-1 corpus JSONL ({"text": ...} per line)."""
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line).get("text", "").strip()
            if len(t) >= 4:
                texts.append(t)
    if not texts:
        raise ValueError(f"no usable text in {path}")
    return texts


def sample_words(texts: list[str], rng: random.Random, n_words: int) -> list[str]:
    """Sample a contiguous run of words from a random corpus document."""
    words: list[str] = []
    while len(words) < n_words:
        doc = rng.choice(texts).split()
        if not doc:
            continue
        start = rng.randrange(len(doc))
        words.extend(doc[start : start + n_words - len(words)])
    return words


# ── rendering ──────────────────────────────────────────────────────────────────

def _rand_colors(rng: random.Random) -> tuple[tuple, tuple]:
    """(paper, ink) — mostly dark-on-light, occasionally inverted."""
    if rng.random() < 0.9:
        paper = tuple(rng.randint(225, 255) for _ in range(3))
        ink   = tuple(rng.randint(0, 70)    for _ in range(3))
    else:
        paper = tuple(rng.randint(0, 60)    for _ in range(3))
        ink   = tuple(rng.randint(200, 255) for _ in range(3))
    return paper, ink


def render_crop(
    texts:     list[str],
    fonts:     list[str],
    rng:       random.Random,
    max_lines: int = 1,
) -> tuple[Image.Image, str]:
    """
    Render 1..max_lines lines of corpus text at natural aspect ratio.

    Returns (image, label) — label is exactly the drawn text, lines joined
    with a single space (the recognizer outputs plain running text; visual
    line breaks are presentation, not content).
    """
    n_lines   = rng.randint(1, max_lines)
    font_size = rng.randint(18, 48)
    font      = ImageFont.truetype(rng.choice(fonts), font_size)
    paper, ink = _rand_colors(rng)

    # Target line width in px — longer for single lines (real lines are wide),
    # tighter for paragraphs (column-ish blocks)
    max_width = rng.randint(300, 900) if n_lines == 1 else rng.randint(250, 600)

    margin       = rng.randint(4, int(font_size * 0.8))
    line_spacing = int(font_size * rng.uniform(1.25, 1.7))

    # Fill lines word-by-word; the label is what actually fits
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    words = sample_words(texts, rng, n_words=n_lines * 14)
    lines: list[str] = []
    cur = ""
    for w in words:
        cand = (cur + " " + w).strip()
        if probe.textlength(cand, font=font) <= max_width or not cur:
            cur = cand
        else:
            lines.append(cur)
            if len(lines) == n_lines:
                cur = ""
                break
            cur = w
    if cur and len(lines) < n_lines:
        lines.append(cur)

    label  = " ".join(lines)
    width  = int(max(probe.textlength(l, font=font) for l in lines)) + 2 * margin
    height = line_spacing * len(lines) + 2 * margin

    img  = Image.new("RGB", (width, height), paper)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((margin, margin + i * line_spacing), line, font=font, fill=ink)

    return img, label


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Light document-noise augmentations. Each applied independently."""
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))
    if rng.random() < 0.3:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.6, 1.3))
    if rng.random() < 0.25:  # small skew; fill with paper color from a corner
        img = img.rotate(rng.uniform(-2.0, 2.0), expand=True,
                         fillcolor=img.getpixel((1, 1)))
    if rng.random() < 0.3:   # JPEG artifacts
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(30, 75))
        img = Image.open(buf).convert("RGB")
    if rng.random() < 0.25:  # sparse salt-and-pepper noise
        px = img.load()
        w, h = img.size
        for _ in range(int(w * h * 0.002)):
            x, y = rng.randrange(w), rng.randrange(h)
            v = rng.randint(0, 255)
            px[x, y] = (v, v, v)
    return img


# ── driver ─────────────────────────────────────────────────────────────────────

def generate(
    corpus:    str,
    fonts_dir: str,
    out_dir:   str,
    n:         int,
    max_lines: int = 1,
    seed:      int = 42,
):
    # Complex-script shaping check: without libraqm PIL draws Devanagari
    # matras/conjuncts in the wrong positions — silently corrupting every
    # training label. Fail loudly instead.
    from PIL import features
    if not features.check("raqm"):
        raise RuntimeError(
            "PIL lacks libraqm — Devanagari will render with broken matra/"
            "conjunct shaping. Install with: pip install pillow --no-binary "
            ":all: (after `apt install libraqm-dev`) or use a Pillow wheel "
            "with raqm support.")

    rng   = random.Random(seed)
    texts = load_corpus(corpus)
    fonts = sorted(str(p) for ext in ("*.ttf", "*.otf") for p in Path(fonts_dir).glob(ext))
    if not fonts:
        raise FileNotFoundError(
            f"no .ttf/.otf fonts in {fonts_dir} — see README for the Noto "
            "Devanagari download command; add more fonts for variety")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[synth] {n:,} crops | max_lines={max_lines} | "
          f"{len(fonts)} fonts | {len(texts):,} corpus docs → {out}")

    with open(out / "data.jsonl", "w", encoding="utf-8") as f:
        for i in range(n):
            img, label = render_crop(texts, fonts, rng, max_lines)
            img = augment(img, rng)
            fname = f"{i:06d}.png"
            img.save(out / fname)
            f.write(json.dumps({"image": fname, "text": label},
                               ensure_ascii=False) + "\n")
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1:,}/{n:,}")
    print(f"[synth] done → {out}/data.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Akshara synthetic crop generator")
    ap.add_argument("--corpus",    required=True, help="Stage-1 corpus JSONL")
    ap.add_argument("--fonts",     default="fonts/", help="dir of .ttf/.otf Devanagari fonts")
    ap.add_argument("--out",       required=True, help="output directory")
    ap.add_argument("--n",         type=int, default=10_000)
    ap.add_argument("--max_lines", type=int, default=1,
                    help="1 = Stage 2A line warmup; ~12 = Stage 2B paragraphs")
    ap.add_argument("--seed",      type=int, default=42)
    a = ap.parse_args()
    generate(a.corpus, a.fonts, a.out, a.n, a.max_lines, a.seed)
