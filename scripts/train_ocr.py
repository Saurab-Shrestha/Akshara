"""
OCR fine-tuning script for the Nepali VLM (vision + language).

WHY two-phase training (freeze then unfreeze):
    For the first 1000 steps the vision encoder is frozen and only the connector
    + decoder are updated.  This matters because:

    1. The decoder arrives from pretrain.pt with good Nepali language priors.
       Random-init gradients from an untrained vision encoder would corrupt those
       weights before the encoder has any useful gradient signal.

    2. The connector (MLP) needs to learn the right projection from ViT patch
       embeddings to the decoder's residual stream dimension before the encoder
       itself gets adjusted.  Freezing the encoder first makes this a simpler
       regression problem.

    After 1000 steps both modules are unfrozen and the full model learns jointly.

EVALUATION METRIC — CER (Character Error Rate):
    CER = edit_distance(prediction, ground_truth) / len(ground_truth)

    WHY CER instead of loss / perplexity:
        The end-user cares about character accuracy, not token probability.
        A model that confidently produces the wrong character scores high
        perplexity but terrible CER.  Greedy decoding is used for simplicity;
        beam search would give slightly better CER but is much slower.

CHECKPOINT FORMAT:
    Same as pretrain.py — {step, model_state_dict, optimizer_state_dict, config}.
    This lets you resume or evaluate with a single torch.load call.

USAGE:
    # Full training from pretrained decoder:
    PYTHONPATH=. python scripts/train_ocr.py --config configs/ocr_finetune.json

    # Smoke test:
    PYTHONPATH=. python scripts/train_ocr.py --config configs/smoke/ocr_finetune.json

    # Start from scratch (no pretrained decoder):
    PYTHONPATH=. python scripts/train_ocr.py --config configs/ocr_finetune.json --pretrain_ckpt null
"""

from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn

from config.config import OCRFinetuneConfig
from config.loader import load_config
from src.data.ocr_dataset import get_ocr_dataloader
from src.models.vlm import Akshara


# ---------------------------------------------------------------------------
# Helpers (identical pattern to pretrain.py for consistency)
# ---------------------------------------------------------------------------

def cosine_lr(step, warmup_steps, max_steps, lr, min_lr) -> float:
    """Cosine LR with linear warmup — see pretrain.py for rationale."""
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def amp_context(use_amp: bool, amp_dtype, device: str):
    if not use_amp or device == "cpu" or amp_dtype is None:
        return nullcontext()
    device_type = "cuda" if device.startswith("cuda") else device
    if device_type not in ("cuda", "mps"):
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.amp.autocast(device_type=device_type, dtype=dtype)


def _strip_html(html: str) -> str:
    """Remove HTML tags, leaving plain text for CER comparison."""
    import re
    return re.sub(r"<[^>]+>", "", html).strip()


def _cer(pred: str, ref: str) -> float:
    """
    Character Error Rate via dynamic-programming edit distance.

    Both pred and ref should be plain text (HTML tags stripped).
    Returns a float in [0, ∞).
    """
    if not ref:
        return 0.0 if not pred else 1.0
    n, m = len(ref), len(pred)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == pred[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m] / n


@torch.no_grad()
def greedy_decode(model: Akshara, image_tensor: torch.Tensor, tokenizer, max_new: int = 128) -> str:
    """
    Greedy decode a single image tensor to text.

    WHY greedy and not beam search:
        Speed.  Beam search gives ~1-3% CER improvement but is 5× slower and adds
        significant code complexity.  For evaluation during training, greedy is fine.

    Args:
        model:        The Akshara model in eval mode.
        image_tensor: [1, 3, H, W] float32 tensor (already normalised).
        tokenizer:    HuggingFace tokenizer for decoding token ids.
        max_new:      Maximum number of tokens to generate.

    Returns:
        Decoded string (without BOS/EOS tokens).
    """
    model.eval()
    eos = tokenizer.eos_token_id
    bos = tokenizer.bos_token_id or eos

    # Start with BOS token.
    tokens = torch.tensor([[bos]], dtype=torch.long, device=image_tensor.device)

    for _ in range(max_new):
        logits = model.generate_step(image_tensor, tokens)   # [1, vocab]
        next_tok = logits.argmax(dim=-1, keepdim=True)        # [1, 1]
        tokens = torch.cat([tokens, next_tok], dim=1)
        if next_tok.item() == eos:
            break

    # Strip BOS and EOS from the output ids before decoding.
    ids = tokens[0].tolist()
    ids = [t for t in ids if t not in (bos, eos)]
    return tokenizer.decode(ids, skip_special_tokens=True)


@torch.no_grad()
def evaluate_cer(
    model: Akshara,
    dev_loader,
    tokenizer,
    cfg: OCRFinetuneConfig,
    n_iters: int,
) -> float:
    """
    Average CER over ``n_iters`` batches of the dev set.

    We evaluate one sample per batch to keep evaluation fast — the first image
    in each batch is decoded and compared to its ground-truth text.
    """
    model.eval()
    total_cer = 0.0
    count = 0
    for i, (images, input_ids, target_ids) in enumerate(dev_loader):
        if i >= n_iters:
            break
        images = images.to(cfg.device)
        # Decode only the first sample in the batch for speed.
        single_image = images[:1]
        pred_text = greedy_decode(model, single_image, tokenizer, max_new=cfg.max_seq_len)
        # Ground truth: decode target_ids[0], strip padding and HTML tags.
        gt_ids = target_ids[0].tolist()
        gt_ids = [t for t in gt_ids if t != tokenizer.eos_token_id]
        gt_text = _strip_html(tokenizer.decode(gt_ids, skip_special_tokens=True))
        pred_plain = _strip_html(pred_text)
        total_cer += _cer(pred_plain, gt_text)
        count += 1
    model.train()
    return total_cer / max(1, count)


def save_checkpoint(path, model, optimizer, step, cfg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(cfg),
    }, path)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    # --- parse CLI ---
    parser = argparse.ArgumentParser(description="Akshara — OCR fine-tuning")
    parser.add_argument("--config", type=str, default="configs/ocr_finetune.json")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume OCR training from")
    parser.add_argument("--pretrain_ckpt", type=str, default=None,
                        help="Pretrained decoder checkpoint (overrides config)")
    parser.add_argument("--train_path",  type=str,   default=None)
    parser.add_argument("--dev_path",    type=str,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--grad_accum",  type=int,   default=None)
    parser.add_argument("--train_steps", type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--out_ckpt",    type=str,   default=None)
    parser.add_argument("--seed",        type=int,   default=None)
    parser.add_argument("--device",      type=str,   default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config", "resume") and v is not None}
    cfg = load_config(OCRFinetuneConfig, args.config, overrides=overrides)

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = cfg.device
    print(f"[train_ocr] config loaded | device={device} | steps={cfg.train_steps}")

    # --- tokenizer ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

    # --- model ---
    model = Akshara(
        vocab_size=cfg.vocab_size,
        n_embed=cfg.n_embed,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        n_layers=cfg.n_layers,
        max_seq_len=cfg.max_seq_len,
        attn_every=cfg.attn_every,
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        vision_dim=cfg.vision_dim,
        vit_layers=cfg.vit_layers,
        vit_heads=cfg.vit_heads,
    )
    if cfg.use_gradient_checkpointing:
        model.set_gradient_checkpointing(True)

    # --- warm-start decoder from pretrain checkpoint ---
    pretrain_ckpt = cfg.pretrain_ckpt
    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        ck = torch.load(pretrain_ckpt, map_location="cpu", weights_only=False)
        # Load only the decoder weights; vision encoder starts from scratch.
        model.decoder.load_state_dict(ck["model_state_dict"], strict=True)
        print(f"[train_ocr] loaded decoder weights from {pretrain_ckpt}")
    elif pretrain_ckpt:
        print(f"[train_ocr] warning: pretrain_ckpt={pretrain_ckpt} not found — starting from scratch")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_ocr] total params: {n_params:,} (~{n_params / 1e6:.1f}M)")

    # --- optional resume (full VLM checkpoint) ---
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        start_step = ck.get("step", 0)
        print(f"[train_ocr] resumed from {args.resume} at step {start_step}")

    # --- optimizer ---
    decay_params   = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,   "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.95), fused=(device == "cuda" and torch.cuda.is_available()))

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])

    # --- data ---
    train_loader = get_ocr_dataloader(
        cfg.train_path, tokenizer, cfg.batch_size, cfg.img_size, cfg.max_seq_len, shuffle=True
    )
    dev_loader = get_ocr_dataloader(
        cfg.dev_path, tokenizer, cfg.batch_size, cfg.img_size, cfg.max_seq_len, shuffle=False
    ) if os.path.exists(cfg.dev_path) else None

    os.makedirs(cfg.log_dir, exist_ok=True)
    train_iter = iter(train_loader)

    # Phase 1: freeze the vision encoder for the first 1000 steps.
    UNFREEZE_AT = 1000
    encoder_frozen = False
    if start_step < UNFREEZE_AT:
        model.freeze_encoder()
        encoder_frozen = True
        print(f"[train_ocr] vision encoder FROZEN until step {UNFREEZE_AT}")

    # --- training loop ---
    model.train()
    t0 = time.perf_counter()
    accum_loss = 0.0
    tokens_per_step = cfg.batch_size * cfg.max_seq_len * cfg.grad_accum

    for step in range(start_step, cfg.train_steps):

        # Phase 2 transition: unfreeze encoder after 1000 steps.
        if encoder_frozen and step >= UNFREEZE_AT:
            model.unfreeze_all()
            encoder_frozen = False
            print(f"[train_ocr] vision encoder UNFROZEN at step {step}")

        lr = cosine_lr(step, cfg.warmup_steps, cfg.train_steps, cfg.lr, cfg.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(cfg.grad_accum):
            try:
                images, input_ids, target_ids = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, input_ids, target_ids = next(train_iter)

            images     = images.to(device)
            input_ids  = input_ids.to(device)
            target_ids = target_ids.to(device)

            with amp_context(cfg.use_amp, cfg.amp_dtype, device):
                _, loss = model(images, input_ids, target_ids)
                loss = loss / cfg.grad_accum

            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % 20 == 0:
            dt = time.perf_counter() - t0
            tok_s = tokens_per_step * 20 / dt if step > start_step else 0.0
            t0 = time.perf_counter()
            print(f"step {step:>6d} | loss {accum_loss:.4f} | lr {lr:.2e} | {tok_s:,.0f} tok/s")

        if step > start_step and step % cfg.eval_steps == 0 and dev_loader is not None:
            cer = evaluate_cer(model, dev_loader, tokenizer, cfg, cfg.eval_iters)
            print(f"  [eval] step {step} | CER {cer:.4f} ({cer * 100:.2f}%)")

        if step > start_step and step % cfg.save_every == 0:
            save_checkpoint(cfg.out_ckpt, model, optimizer, step, cfg)
            print(f"  [ckpt] saved → {cfg.out_ckpt} (step {step})")

    save_checkpoint(cfg.out_ckpt, model, optimizer, cfg.train_steps, cfg)
    print(f"[train_ocr] done. Final checkpoint → {cfg.out_ckpt}")


if __name__ == "__main__":
    main()
