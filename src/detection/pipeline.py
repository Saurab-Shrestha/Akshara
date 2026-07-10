"""
Full OCR Pipeline: Detection → Crop → Recognition
===================================================

WHY A SEPARATE PIPELINE MODULE
--------------------------------
The model (vlm.py) is responsible for "given these tokens and this image, what
comes next?".  The pipeline is responsible for the full document workflow:
  1. Detect text line locations (SuryaDetector)
  2. Crop each line
  3. Run OCR on each crop (Akshara)
  4. Re-assemble into a document with spatial layout

Keeping these separate means:
  - You can swap the detector (Surya → your own EAST/DBNet later)
  - You can swap the recogniser (our VLM → a larger one later)
  - Each component is testable in isolation

PIPELINE MODES
--------------
  FULL_PAGE  — send the entire image to the VLM (works for clean synthetic images)
  LINE       — detect lines, crop each one, OCR each crop (needed for real documents)

For real Nepali documents (newspapers, scanned books), always use LINE mode.
Full-page sends too many visual tokens and the decoder gets confused by layout.

OUTPUT FORMAT
-------------
    [
        {"text": "नेपाल एक सुन्दर देश हो।", "bbox": (0.05, 0.15, 0.95, 0.28)},
        {"text": "काठमाडौं राजधानी हो।",   "bbox": (0.05, 0.30, 0.75, 0.43)},
        ...
    ]

Each entry has the decoded text and the normalised bounding box of the source
line in the original image.  This is enough to render an overlay, build a
searchable PDF, or dump plain text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

from src.detection.surya_detector import SuryaDetector, NormBox


# ImageNet stats — must match vit.py / ocr_dataset.py
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


@dataclass
class OCRLine:
    """One recognised text line with its position in the source image."""
    text: str
    bbox: NormBox           # (left, top, right, bottom) normalised [0, 1]
    confidence: float = 1.0 # reserved for future logprob-based confidence


@dataclass
class OCRResult:
    """All recognised lines from one image."""
    lines: List[OCRLine] = field(default_factory=list)
    image_path: Optional[str] = None

    @property
    def full_text(self) -> str:
        """All lines joined with newlines — plain text output."""
        return "\n".join(line.text for line in self.lines)

    def __repr__(self):
        return f"OCRResult({len(self.lines)} lines)"


class AksharaPipeline:
    """
    Full document OCR pipeline.

    Combines:
      - SuryaDetector    (text line localisation)
      - Akshara VLM    (text recognition per crop)
      - Qwen3.5 tokenizer (decode token ids → string)

    Usage:
        pipeline = AksharaPipeline.from_checkpoint("checkpoints/ocr.pt")
        result   = pipeline.run("document.png")
        print(result.full_text)
    """

    def __init__(
        self,
        model,              # Akshara instance
        tokenizer,          # HuggingFace tokenizer
        img_size: int = 224,
        device: str = "cpu",
        temperature: float = 0.1,
        max_new_tokens: int = 256,
    ):
        self.model        = model
        self.tokenizer    = tokenizer
        self.img_size     = img_size
        self.device       = device
        self.temperature  = temperature
        self.max_new_tokens = max_new_tokens

        self.detector = SuryaDetector(device=device)

        self._transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[str] = None,
        temperature: float = 0.1,
    ) -> "AksharaPipeline":
        """
        Load a trained Akshara checkpoint and build the full pipeline.

        Args:
            checkpoint_path: Path to a checkpoint saved by scripts/train_ocr.py
            device:          'cuda', 'cpu', or None (auto-detect)
            temperature:     Decoding temperature (0.1 = near-greedy, good for OCR)
        """
        import torch
        from src.models.vlm import Akshara
        from transformers import AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})

        model = Akshara(
            vocab_size  = cfg.get("vocab_size",  248044),
            n_embed     = cfg.get("n_embed",     768),
            n_heads     = cfg.get("n_heads",     12),
            n_kv_heads  = cfg.get("n_kv_heads",  3),
            n_layers    = cfg.get("n_layers",    12),
            max_seq_len = cfg.get("max_seq_len", 512),
            attn_every  = cfg.get("attn_every",  4),
            img_size    = cfg.get("img_size",    224),
            patch_size  = cfg.get("patch_size",  16),
            vision_dim  = cfg.get("vision_dim",  384),
            vit_layers  = cfg.get("vit_layers",  12),
            vit_heads   = cfg.get("vit_heads",   6),
        )
        model.load_state_dict(ck["model_state_dict"])
        model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-0.8B", trust_remote_code=True
        )

        return cls(
            model      = model,
            tokenizer  = tokenizer,
            img_size   = cfg.get("img_size", 224),
            device     = device,
            temperature= temperature,
        )

    def run(
        self,
        image: Image.Image | str | Path,
        mode: str = "line",
    ) -> OCRResult:
        """
        Run OCR on a document image.

        Args:
            image: PIL Image or path to an image file.
            mode:  "line" — detect + crop + OCR each line (recommended for real docs)
                   "full" — send whole image (fast, only works for clean single-line crops)

        Returns:
            OCRResult with all recognised lines.
        """
        if not isinstance(image, Image.Image):
            image_path = str(image)
            image = Image.open(image).convert("RGB")
        else:
            image_path = None

        if mode == "full":
            return self._run_full_page(image, image_path)
        else:
            return self._run_line_by_line(image, image_path)

    def _run_full_page(self, image: Image.Image, image_path) -> OCRResult:
        """
        OCR the entire image as one input.
        Use only for single-line images or clean pre-cropped regions.
        """
        tensor = self._transform(image).unsqueeze(0).to(self.device)
        text   = self._decode(tensor)
        return OCRResult(
            lines      = [OCRLine(text=text, bbox=(0.0, 0.0, 1.0, 1.0))],
            image_path = image_path,
        )

    def _run_line_by_line(self, image: Image.Image, image_path) -> OCRResult:
        """
        Detect text lines, crop each one, run OCR on each crop.
        Recommended for real documents with multiple lines.
        """
        crops = self.detector.crop_lines(image, pad_px=4)

        if not crops:
            # Fallback: no lines detected → treat full image as one region
            return self._run_full_page(image, image_path)

        lines = []
        for crop_img, norm_box in crops:
            tensor = self._transform(crop_img).unsqueeze(0).to(self.device)
            text   = self._decode(tensor)
            if text.strip():                    # skip empty detections
                lines.append(OCRLine(text=text.strip(), bbox=norm_box))

        return OCRResult(lines=lines, image_path=image_path)

    @torch.no_grad()
    def _decode(self, image_tensor: torch.Tensor) -> str:
        """
        Autoregressively decode one image tensor to text.

        Uses generate_step so we can implement custom stop conditions.
        The full model.generate() also works but is slightly less flexible.
        """
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id

        tokens = torch.tensor([[bos]], dtype=torch.long, device=self.device)

        for _ in range(self.max_new_tokens):
            logits = self.model.generate_step(image_tensor, tokens)  # (1, vocab)

            if self.temperature <= 1e-6:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                probs    = torch.softmax(logits / self.temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_tok], dim=1)

            if next_tok.item() == eos:
                break

        ids = [t for t in tokens[0].tolist() if t not in (bos, eos)]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def batch_run(
        self,
        images: List[Image.Image | str | Path],
        mode: str = "line",
    ) -> List[OCRResult]:
        """
        Run OCR on a list of images.

        Currently processes sequentially. For production, this would be
        batched across the recognition model. Batching detection separately
        and grouping crops by image would be the right approach — implement
        when throughput matters.
        """
        return [self.run(img, mode=mode) for img in images]


# ── self-check ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    from PIL import ImageDraw

    print("AksharaPipeline self-check (untrained model — output is random)")
    print("=" * 60)

    # Build a tiny untrained model
    from src.models.vlm import Akshara

    class _TinyTok:
        bos_token_id = 1
        eos_token_id = 2
        def decode(self, ids, skip_special_tokens=True):
            return "test"

    tiny_model = Akshara(
        vocab_size=100, n_embed=64, n_heads=4, n_kv_heads=2,
        n_layers=2, max_seq_len=32, attn_every=4,
        img_size=32, patch_size=16, vision_dim=64, vit_layers=2, vit_heads=4,
    )

    pipeline = AksharaPipeline(
        model=tiny_model, tokenizer=_TinyTok(),
        img_size=32, device="cpu", temperature=0.0, max_new_tokens=5,
    )

    # Synthetic page image
    img = Image.new("RGB", (200, 120), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 15, 190, 35], fill="black")
    draw.rectangle([10, 50, 150, 70], fill="black")
    draw.rectangle([10, 85, 180, 105], fill="black")

    # Full-page mode
    result_full = pipeline.run(img, mode="full")
    print(f"\nFull-page mode: {result_full}")
    assert len(result_full.lines) == 1

    # Line mode (uses OpenCV fallback if surya not installed)
    result_line = pipeline.run(img, mode="line")
    print(f"Line mode:      {result_line}")
    assert len(result_line.lines) >= 1

    print(f"\nfull_text (line mode):\n{result_line.full_text}")
    print("\nSelf-check PASSED ✅")
