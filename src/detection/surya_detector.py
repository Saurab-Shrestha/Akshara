"""
Text Line Detector — Surya EfficientViT wrapper
=================================================

WHY WE BORROW INSTEAD OF BUILD
--------------------------------
Detection (finding where text is on a page) and recognition (reading what the
text says) are two separate hard problems.  Surya's EfficientViT detection
model is:
  - Apache-2.0 licensed (code AND weights — unlike the VLM)
  - Already trained on multilingual documents including Devanagari layouts
  - Handles multi-column, mixed-content pages that OpenCV contours can't

Building our own detector from scratch would take 2-3 weeks and add no
architectural learning — the interesting part of this project is the
recognition VLM.  We use Surya's detector as infrastructure.

HOW SURYA DETECTION WORKS
--------------------------
1. Input: PIL Image (any size)
2. EfficientViT encoder (MBConv + LiteMLA attention) produces two heatmaps:
   - linemap:    probability that each pixel belongs to a text line
   - affinity:   probability that adjacent pixels belong to the same line
3. Post-processing (heatmap.py / CRAFT-style connected components):
   - Threshold linemap → binary mask
   - Connected components → bounding polygons
   - Merge overlapping boxes
4. Output: list of PolygonBox objects with (x1, y1, x2, y2) corners

OUR WRAPPER
-----------
`SuryaDetector` is a thin wrapper that:
  - Handles import errors gracefully (surya may not be installed)
  - Returns normalized bounding boxes (0.0–1.0) instead of pixel coords
    so OCRDataset can crop without knowing the original image size
  - Sorts detections top-left → bottom-right (reading order heuristic)
  - Has a CPU fallback using OpenCV contours when surya is unavailable

INSTALLATION
------------
Surya must be installed separately (it has heavy dependencies):

    # From the cloned surya-0.20.0 directory:
    pip install -e /path/to/surya-0.20.0

Or from PyPI (may be a newer version):
    pip install surya-ocr

The detector checkpoint is downloaded automatically from the Datalab S3 on
first use (~100MB).

USAGE
-----
    from src.detection.surya_detector import SuryaDetector

    detector = SuryaDetector()                # loads model on first call
    boxes = detector.detect("page.png")       # list of (x1, y1, x2, y2) normalised
    crops = detector.crop_lines("page.png")   # list of PIL.Image line crops
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image


# Normalised bounding box: (left, top, right, bottom) all in [0.0, 1.0]
NormBox = Tuple[float, float, float, float]


class SuryaDetector:
    """
    Wraps Surya's EfficientViT text detection model.

    The model is loaded lazily on the first call to ``detect()`` or
    ``crop_lines()`` — importing this class does not trigger any downloads.

    If Surya is not installed, falls back to a simple OpenCV contour detector
    suitable for clean synthetic images (not real-world documents).
    """

    def __init__(self, device: Optional[str] = None):
        """
        Args:
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.device = device
        self._predictor = None    # loaded lazily
        self._surya_available: Optional[bool] = None

    def _check_surya(self) -> bool:
        """Return True if surya is importable."""
        if self._surya_available is None:
            try:
                import surya  # noqa: F401
                self._surya_available = True
            except ImportError:
                self._surya_available = False
                print("[detector] surya not installed — falling back to OpenCV contour detection")
                print("[detector] install: pip install surya-ocr  (or pip install -e /path/to/surya-0.20.0)")
        return self._surya_available

    def _load_predictor(self):
        """Load the Surya DetectionPredictor (downloads weights on first call)."""
        if self._predictor is not None:
            return

        from surya.detection import DetectionPredictor

        if self.device is not None:
            import os
            os.environ.setdefault("TORCH_DEVICE", self.device)

        print("[detector] loading Surya EfficientViT detection model…")
        self._predictor = DetectionPredictor()
        print("[detector] model ready")

    def detect(self, image: Image.Image | str | Path) -> List[NormBox]:
        """
        Detect text line bounding boxes in an image.

        Args:
            image: PIL Image, or a path to an image file.

        Returns:
            List of (left, top, right, bottom) normalised to [0.0, 1.0].
            Sorted in approximate reading order (top-to-bottom, left-to-right).
            Empty list if no text found.
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        W, H = image.size

        if self._check_surya():
            self._load_predictor()
            results = self._predictor([image])
            boxes_raw = results[0].bboxes   # list of PolygonBox

            norm_boxes = []
            for poly in boxes_raw:
                # PolygonBox has .bbox = [x1, y1, x2, y2] in pixels
                x1, y1, x2, y2 = poly.bbox
                norm_boxes.append((
                    max(0.0, x1 / W),
                    max(0.0, y1 / H),
                    min(1.0, x2 / W),
                    min(1.0, y2 / H),
                ))

            return _sort_reading_order(norm_boxes)

        else:
            return self._opencv_fallback(image, W, H)

    def crop_lines(
        self,
        image: Image.Image | str | Path,
        pad_px: int = 4,
    ) -> List[Tuple[Image.Image, NormBox]]:
        """
        Detect text lines and return cropped PIL images.

        Args:
            image:  PIL Image or path.
            pad_px: Pixels of padding around each crop (prevents edge clipping).

        Returns:
            List of (crop, norm_box) pairs in reading order.
            crop: PIL.Image RGB crop of the text line.
            norm_box: normalised (left, top, right, bottom) of the crop in the
                      original image (useful to reassemble layout later).
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        W, H = image.size
        boxes = self.detect(image)

        crops = []
        for (x1n, y1n, x2n, y2n) in boxes:
            # Convert back to pixels and add padding
            x1 = max(0, int(x1n * W) - pad_px)
            y1 = max(0, int(y1n * H) - pad_px)
            x2 = min(W, int(x2n * W) + pad_px)
            y2 = min(H, int(y2n * H) + pad_px)

            crop = image.crop((x1, y1, x2, y2))
            crops.append((crop, (x1n, y1n, x2n, y2n)))

        return crops

    def _opencv_fallback(self, image: Image.Image, W: int, H: int) -> List[NormBox]:
        """
        Minimal OpenCV contour detector for clean synthetic images.

        Works well on white-background synthetic training images.
        Fails on real-world multi-column layouts (use Surya for those).
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("[detector] OpenCV not available — returning full image as single box")
            return [(0.0, 0.0, 1.0, 1.0)]

        img_np = np.array(image.convert("L"))           # grayscale
        _, binary = cv2.threshold(img_np, 200, 255, cv2.THRESH_BINARY_INV)

        # Dilate horizontally to merge characters in a line
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 2))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 4:                         # skip noise
                continue
            boxes.append((x / W, y / H, (x + w) / W, (y + h) / H))

        return _sort_reading_order(boxes) if boxes else [(0.0, 0.0, 1.0, 1.0)]


def _sort_reading_order(boxes: List[NormBox]) -> List[NormBox]:
    """
    Sort bounding boxes in approximate reading order: top-to-bottom, then
    left-to-right within a "band" of similar vertical position.

    Two boxes are considered on the same line if their vertical centres are
    within 1.5× the median box height of each other.
    """
    if not boxes:
        return boxes

    heights = [y2 - y1 for (_, y1, _, y2) in boxes]
    median_h = sorted(heights)[len(heights) // 2]
    band = median_h * 1.5

    def sort_key(box):
        _, y1, _, y2 = box
        cy = (y1 + y2) / 2
        # Snap y to a band so nearby rows sort together
        band_idx = int(cy / band)
        return (band_idx, box[0])   # band first, then left edge

    return sorted(boxes, key=sort_key)


# ── self-check ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("SuryaDetector self-check")
    print("=" * 40)

    detector = SuryaDetector()

    # Create a tiny synthetic test image with text-like dark rectangles
    from PIL import ImageDraw
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Simulate two text lines as black rectangles
    draw.rectangle([20, 30, 380, 55], fill=(30, 30, 30))
    draw.rectangle([20, 80, 280, 105], fill=(30, 30, 30))

    # Test with OpenCV fallback (doesn't need surya installed)
    boxes = detector._opencv_fallback(img, 400, 200)
    print(f"\nOpenCV fallback: found {len(boxes)} box(es)")
    for b in boxes:
        print(f"  {tuple(f'{v:.3f}' for v in b)}")
    assert len(boxes) >= 1, "expected at least one box"

    # Test crop_lines (uses fallback if surya not installed)
    crops = detector.crop_lines(img, pad_px=2)
    print(f"\ncrop_lines: returned {len(crops)} crop(s)")
    for crop_img, norm_box in crops:
        print(f"  crop size={crop_img.size}  box={tuple(f'{v:.3f}' for v in norm_box)}")
        assert crop_img.size[0] > 0 and crop_img.size[1] > 0

    print("\nSelf-check PASSED ✅")

    # If surya is installed, test the real detector too
    if detector._check_surya():
        print("\nSurya is installed — testing real detection on synthetic image...")
        try:
            boxes_surya = detector.detect(img)
            print(f"  Surya found {len(boxes_surya)} box(es)")
        except Exception as e:
            print(f"  Surya detection failed: {e}")
    else:
        print("\nSurya not installed — skipping real detector test.")
        print("Install with: pip install surya-ocr")
