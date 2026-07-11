"""
Inference script: run OCR on an image and print the decoded Nepali text.

WHY a separate inference script:
    Training scripts carry dataloaders, gradient accumulation, and eval loops that
    are irrelevant at inference time.  This thin script loads a checkpoint, runs
    the model in eval mode, and decodes the output — it can be called from a shell
    pipeline without importing anything from the training path.

DECODING STRATEGY:
    Temperature sampling with a low default temperature (0.1).  Near-zero temperature
    approximates greedy decoding (argmax) but avoids the sharp cliff that can make
    argmax unstable when two logits are very close.  For production use, temperature
    should be 0 (pure greedy) or a small positive number.

    WHY not beam search:
        A single forward-pass per image is already fast.  Beam search would multiply
        latency by the beam width for typically marginal gain on printed Nepali text.
        Use beam search in post-processing if you need it.

USAGE:
    # Basic OCR:
    PYTHONPATH=. python scripts/generate.py \\
        --checkpoint checkpoints/ocr.pt \\
        --image path/to/image.png

    # With temperature and save to file:
    PYTHONPATH=. python scripts/generate.py \\
        --checkpoint checkpoints/ocr.pt \\
        --image path/to/image.png \\
        --temperature 0.0 \\
        --save_txt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

# ImageNet normalisation — must match the training preprocessing in ocr_dataset.py.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def preprocess_image(image_path: str, img_size: int) -> torch.Tensor:
    """
    Load and preprocess a single image to match training preprocessing.

    Args:
        image_path: Path to the image file.
        img_size:   Target square resolution (must match the model's img_size).

    Returns:
        FloatTensor [1, 3, img_size, img_size] ready for model input.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)   # [1, 3, H, W]


@torch.no_grad()
def generate(
    model,
    image_tensor: torch.Tensor,
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """
    Autoregressively decode text from an image tensor.

    Args:
        model:          Akshara in eval mode.
        image_tensor:   [1, 3, H, W] preprocessed image on the model's device.
        tokenizer:      HuggingFace tokenizer.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature:    Sampling temperature.  0 or very small ≈ greedy.

    Returns:
        Decoded string (BOS/EOS stripped).
    """
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    tokens = torch.tensor([[bos]], dtype=torch.long, device=image_tensor.device)

    for _ in range(max_new_tokens):
        logits = model.generate_step(image_tensor, tokens)   # [1, vocab_size]

        if temperature <= 1e-6:
            # Pure greedy — fastest and most deterministic.
            next_tok = logits.argmax(dim=-1, keepdim=True)
        else:
            # Temperature sampling.
            probs = torch.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_tok], dim=1)

        if next_tok.item() == eos:
            break

    ids = tokens[0].tolist()
    ids = [t for t in ids if t not in (bos, eos)]
    return tokenizer.decode(ids, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Akshara — inference on a single image"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a trained OCR checkpoint (checkpoints/ocr.pt)"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input image file"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Sampling temperature (0 = greedy; default: 0.1)"
    )
    parser.add_argument(
        "--save_txt", action="store_true",
        help="If set, write the decoded text to a .txt file alongside the image"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (default: inferred from checkpoint config)"
    )
    args = parser.parse_args()

    # --- validate inputs ---
    if not os.path.exists(args.checkpoint):
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.image):
        print(f"error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # --- load checkpoint ---
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_config = ck.get("config", {})

    device = args.device or saved_config.get("device", "cpu")
    # Fall back to CPU if CUDA requested but not available.
    if device == "cuda" and not torch.cuda.is_available():
        print("[generate] CUDA not available, falling back to CPU", file=sys.stderr)
        device = "cpu"

    # --- reconstruct model from saved config ---
    from src.models.vlm import Akshara
    model = Akshara(
        vocab_size  = saved_config.get("vocab_size",  248077),
        n_embed     = saved_config.get("n_embed",     768),
        n_heads     = saved_config.get("n_heads",     12),
        n_kv_heads  = saved_config.get("n_kv_heads",  3),
        n_layers    = saved_config.get("n_layers",    12),
        max_seq_len = saved_config.get("max_seq_len", 512),
        attn_every  = saved_config.get("attn_every",  4),
        img_size    = saved_config.get("img_size",    448),
        patch_size  = saved_config.get("patch_size",  14),
        vision_dim  = saved_config.get("vision_dim",  384),
        vit_layers  = saved_config.get("vit_layers",  12),
        vit_heads   = saved_config.get("vit_heads",   6),
        vit_pretrained = False,  # weights come from the checkpoint
    )
    model.load_state_dict(ck["model_state_dict"])
    model = model.to(device)
    model.eval()

    img_size = saved_config.get("img_size", 448)

    # --- tokenizer ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

    # --- preprocess + decode ---
    image_tensor = preprocess_image(args.image, img_size).to(device)
    decoded_text = generate(
        model, image_tensor, tokenizer,
        max_new_tokens=512,
        temperature=args.temperature,
    )

    # --- output ---
    print(decoded_text)

    if args.save_txt:
        txt_path = os.path.splitext(args.image)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(decoded_text)
        print(f"[generate] saved → {txt_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
