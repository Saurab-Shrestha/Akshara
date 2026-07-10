# Stage 11 — Text Detection & Full Document Pipeline

## Why detection matters

Our VLM is trained to read **one line at a time**.  Give it a full newspaper
page and it will try to read everything as a single sequence — the visual
tokens are 14×14 patches at 224×224, so one patch covers roughly 16 pixels.
A typical A4 page at 200 dpi has 1654×2339 pixels: a single patch covers a
140×167 pixel block, blurring dozens of characters together.

The fix is the classic two-stage OCR pipeline:

```
full document image
        │
        ▼
┌────────────────────┐
│  Text Detector     │  "where is the text?"
│  (SuryaDetector)   │  → list of bounding boxes
└────────────────────┘
        │  crop each box
        ▼
┌────────────────────┐
│  Text Recogniser   │  "what does it say?"
│  (NepaliOCR VLM)   │  → string per crop
└────────────────────┘
        │
        ▼
   OCRResult (lines + bboxes)
```

---

## What Surya's detector does

Surya uses **EfficientViT** — a lightweight ViT variant where standard
self-attention is replaced by **LiteMLA** (Linear Multi-scale Attention).
LiteMLA runs in O(T) instead of O(T²) by projecting keys and values before
computing the attention matrix, making it fast on high-resolution feature maps.

The detector produces two heatmaps:
- **linemap**: probability each pixel belongs to a text line
- **affinity**: probability adjacent pixels belong to the same line

CRAFT-style post-processing thresholds both maps and runs connected-component
analysis to produce bounding boxes.

Why borrow it instead of building it ourselves:
- It is Apache-2.0 licensed (code + weights) — we can use it commercially
- It was trained on multilingual documents including Devanagari layouts
- Building EfficientViT + CRAFT post-processing from scratch is 2-3 weeks of
  work that teaches us nothing new (we already understand attention from
  Stage 3 and vision encoders from Stage 8)

---

## SuryaDetector wrapper (`src/detection/surya_detector.py`)

```python
from src.detection.surya_detector import SuryaDetector

detector = SuryaDetector()          # lazy load — no download until first call

# Returns normalised boxes: (left, top, right, bottom) in [0, 1]
boxes = detector.detect("page.png")

# Returns (PIL crop, norm_box) pairs sorted in reading order
crops = detector.crop_lines("page.png", pad_px=4)
```

**Graceful fallback**: if `surya` is not installed, the detector falls back to
an OpenCV contour approach — threshold, horizontal dilation, connected
components.  This works well on clean synthetic images (white background, dark
text) and breaks gracefully on real-world layouts.  Your training pipeline
keeps working even without a Surya install.

**Reading order sort**: boxes are sorted with a band heuristic — two boxes
whose vertical centres are within 1.5× the median box height are treated as
the same line, then sorted by left edge within each band.  This handles minor
y-position noise without complex layout analysis.

---

## Full pipeline (`src/detection/pipeline.py`)

```python
from src.detection.pipeline import NepaliOCRPipeline

# Load from checkpoint (downloads tokenizer if needed)
pipeline = NepaliOCRPipeline.from_checkpoint("checkpoints/ocr.pt")

# Single image — returns OCRResult
result = pipeline.run("invoice.png")
print(result.full_text)          # all lines joined with newlines

# Access individual lines
for line in result.lines:
    print(f"{line.bbox}  →  {line.text}")

# Batch
results = pipeline.batch_run(["page1.png", "page2.png"])
```

**Two modes**:
- `mode="line"` (default): detect lines → crop → OCR each crop.  Use for
  real documents with multiple lines.
- `mode="full"`: send entire image directly to VLM.  Use only for
  pre-cropped single-line images or quick tests.

---

## Data classes

```
OCRLine
  .text        str          decoded text for this line
  .bbox        NormBox      (left, top, right, bottom) in [0, 1]
  .confidence  float        1.0 now; reserved for logprob-based confidence

OCRResult
  .lines       List[OCRLine]
  .full_text   str (property) all lines joined with '\n'
  .image_path  Optional[str]
```

---

## Self-check results

```
# surya_detector.py
OpenCV fallback: found 2 box(es)   ← two synthetic rectangles
[detector] loading Surya EfficientViT detection model…
[detector] model ready
Surya found 1 box(es)              ← trained on real text, not solid rectangles

# pipeline.py  (tiny untrained model, 3 synthetic rectangles)
Full-page mode: OCRResult(1 lines)
Line mode:      OCRResult(3 lines)
Self-check PASSED ✅
```

---

## Installation note

```bash
# OpenCV fallback (always install — needed for synthetic data too)
pip install opencv-python-headless

# Surya real detector (73MB weights downloaded on first use)
pip install surya-ocr
# or from the cloned repo:
pip install -e /path/to/surya-0.20.0
```

The weights are cached at `~/.cache/datalab/models/text_detection/2025_05_07`
after the first download.

---

## What's next

- **Stage 12** — Synthetic Devanagari data generation with Noto Sans
  Devanagari font (currently PIL's default font can't render Devanagari)
- **Stage 13** — End-to-end training run on Kaggle T4
