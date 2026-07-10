"""
Language pretraining for the Akshara decoder.

WHY pretrain before OCR fine-tuning:
    The HybridDecoder is trained here on raw Nepali text (no images) so it
    acquires a strong prior on Devanagari script, word-piece boundaries, and
    sentence structure.  When OCR fine-tuning starts, the decoder already knows
    *what* valid Nepali looks like; the vision encoder only needs to teach it
    *which* characters appear in the image.  Without this stage the model must
    simultaneously learn language and vision, which is harder and slower.

TRAINING RECIPE:
    - Cosine LR schedule with linear warmup (standard for transformers).
    - Gradient accumulation to simulate a larger effective batch on T4 16GB.
    - bf16 AMP (autocast) to halve VRAM for activations.
    - Gradient checkpointing to trade compute for memory on the HybridDecoder.
    - Periodic eval (perplexity) and checkpoint saving.

CHECKPOINT FORMAT:
    {
        "step": int,
        "model_state_dict": ...,
        "optimizer_state_dict": ...,
        "config": vars(cfg),         # serialised as plain dict for portability
    }

USAGE:
    # Full pretraining:
    PYTHONPATH=. python scripts/pretrain.py --config configs/pretrain.json

    # Smoke test (10 steps on CPU, tiny model):
    PYTHONPATH=. python scripts/pretrain.py --config configs/smoke/pretrain.json

    # Override individual fields:
    PYTHONPATH=. python scripts/pretrain.py --config configs/pretrain.json --lr 5e-4 --batch_size 8
"""

from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.config import PretrainConfig
from config.loader import load_config
from src.data.text_dataset import get_text_dataloader
from src.models.hybrid_decoder import HybridDecoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    lr: float,
    min_lr: float,
) -> float:
    """
    Cosine LR schedule with linear warmup.

    WHY cosine: smoothly decays from peak to min_lr without a sharp drop,
    giving the optimiser time to converge into the loss basin.
    """
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def amp_context(use_amp: bool, amp_dtype: str | None, device: str):
    """
    Return an autocast context manager, or a no-op if AMP is disabled.

    WHY bf16 over fp16:
        bf16 has the same exponent range as fp32, so gradients don't need a
        GradScaler (which adds code complexity and occasional instability).
        fp16 is available as a fallback for older GPUs (pre-Ampere) that don't
        support bf16.
    """
    if not use_amp or device == "cpu" or amp_dtype is None:
        return nullcontext()
    device_type = "cuda" if device.startswith("cuda") else device
    if device_type not in ("cuda", "mps"):
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.amp.autocast(device_type=device_type, dtype=dtype)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    loader_iter,
    cfg: PretrainConfig,
    n_iters: int,
    device: str,
) -> float:
    """Average cross-entropy loss over ``n_iters`` batches (no grad)."""
    model.eval()
    losses = []
    for _ in range(n_iters):
        try:
            input_ids, targets = next(loader_iter)
        except StopIteration:
            break
        input_ids = input_ids.to(device)
        targets   = targets.to(device)
        with amp_context(cfg.use_amp, cfg.amp_dtype, device):
            _, loss = model(input_ids, targets)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def save_checkpoint(path: str, model: nn.Module, optimizer, step: int, cfg: PretrainConfig) -> None:
    """Save a portable checkpoint dict to ``path``."""
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
    parser = argparse.ArgumentParser(description="Akshara — language pretraining")
    parser.add_argument("--config", type=str, default="configs/pretrain.json",
                        help="Path to stage JSON (default: configs/pretrain.json)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume training from")
    # Allow any PretrainConfig field to be overridden via --field value
    parser.add_argument("--train_path",  type=str,   default=None)
    parser.add_argument("--dev_path",    type=str,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--grad_accum",  type=int,   default=None)
    parser.add_argument("--train_steps", type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--min_lr",      type=float, default=None)
    parser.add_argument("--out_ckpt",    type=str,   default=None)
    parser.add_argument("--seed",        type=int,   default=None)
    parser.add_argument("--device",      type=str,   default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config", "resume") and v is not None}
    cfg = load_config(PretrainConfig, args.config, overrides=overrides)

    # --- reproducibility ---
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = cfg.device
    print(f"[pretrain] config loaded | device={device} | steps={cfg.train_steps}")
    print(f"[pretrain] effective batch = {cfg.batch_size} × {cfg.grad_accum} = "
          f"{cfg.batch_size * cfg.grad_accum} sequences/step")

    # --- tokenizer ---
    # Lazy import so the project works without transformers installed (e.g. for
    # config / data tests).
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

    # --- model ---
    model = HybridDecoder(
        vocab_size=cfg.vocab_size,
        n_embed=cfg.n_embed,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        n_layers=cfg.n_layers,
        max_seq_len=cfg.max_seq_len,
        attn_every=cfg.attn_every,
    )
    if cfg.use_gradient_checkpointing:
        model.set_gradient_checkpointing(True)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] model params: {n_params:,} (~{n_params / 1e6:.1f}M)")

    # --- optional resume (load before DataParallel so keys match) ---
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        start_step = ck.get("step", 0)
        print(f"[pretrain] resumed from {args.resume} at step {start_step}")

    # --- optimizer: build before DataParallel (references same param tensors) ---
    decay_params   = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,   "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.95), fused=(device == "cuda"))

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])

    # --- multi-GPU: wrap in DataParallel if multiple GPUs available ---
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0
    if n_gpus > 1:
        print(f"[pretrain] {n_gpus}× GPU detected — using DataParallel (effective batch ×{n_gpus})")
        model = nn.DataParallel(model)
    # raw_model: always the unwrapped model (for state_dict saving)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    # --- data ---
    train_loader = get_text_dataloader(
        cfg.train_path, tokenizer, cfg.batch_size, cfg.max_seq_len, shuffle=True
    )
    dev_loader = get_text_dataloader(
        cfg.dev_path, tokenizer, cfg.batch_size, cfg.max_seq_len, shuffle=False
    ) if os.path.exists(cfg.dev_path) else None

    train_iter = iter(train_loader)

    # --- log dir + TensorBoard ---
    os.makedirs(cfg.log_dir, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=cfg.log_dir)

    # --- training loop ---
    model.train()
    t0          = time.perf_counter()
    run_start   = time.perf_counter()
    tokens_per_step = cfg.batch_size * cfg.max_seq_len * cfg.grad_accum
    accum_loss  = 0.0

    for step in range(start_step, cfg.train_steps):
        lr = cosine_lr(step, cfg.warmup_steps, cfg.train_steps, cfg.lr, cfg.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(cfg.grad_accum):
            try:
                input_ids, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, targets = next(train_iter)

            input_ids = input_ids.to(device)
            targets   = targets.to(device)

            with amp_context(cfg.use_amp, cfg.amp_dtype, device):
                _, loss = model(input_ids, targets)
                loss = loss / cfg.grad_accum

            loss.backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # --- logging every 20 steps ---
        if step % 20 == 0:
            dt    = time.perf_counter() - t0
            tok_s = tokens_per_step * 20 / dt if step > start_step else 0.0
            t0    = time.perf_counter()

            steps_done      = max(1, step - start_step)
            elapsed         = time.perf_counter() - run_start
            secs_per_step   = elapsed / steps_done
            eta_secs        = (cfg.train_steps - step) * secs_per_step
            eta_str         = f"{eta_secs/3600:.1f}h" if eta_secs > 3600 else f"{eta_secs/60:.0f}m"

            print(f"step {step:>6d} | loss {accum_loss:.4f} | lr {lr:.2e} | "
                  f"gnorm {grad_norm:.2f} | {tok_s:,.0f} tok/s | "
                  f"elapsed {elapsed/3600:.1f}h | eta {eta_str}")

            writer.add_scalar("train/loss",      accum_loss,      step)
            writer.add_scalar("train/lr",         lr,              step)
            writer.add_scalar("train/grad_norm",  grad_norm.item(), step)
            writer.add_scalar("train/tok_per_sec", tok_s,          step)

        # --- periodic eval ---
        if step > start_step and step % cfg.eval_steps == 0:
            if dev_loader is not None:
                dev_iter = iter(dev_loader)
                dev_loss = estimate_loss(model, dev_iter, cfg, cfg.eval_iters, device)
                train_iter_eval = iter(train_loader)
                train_loss = estimate_loss(model, train_iter_eval, cfg, cfg.eval_iters, device)
                train_iter = iter(train_loader)
                ppl = math.exp(min(dev_loss, 20))
                print(f"  [eval] step {step} | train {train_loss:.4f} | dev {dev_loss:.4f} | ppl {ppl:.2f}")
                writer.add_scalar("eval/train_loss", train_loss, step)
                writer.add_scalar("eval/dev_loss",   dev_loss,   step)
                writer.add_scalar("eval/perplexity", ppl,        step)

        # --- periodic checkpoint ---
        if step > start_step and step % cfg.save_every == 0:
            save_checkpoint(cfg.out_ckpt, raw_model, optimizer, step, cfg)
            print(f"  [ckpt] saved → {cfg.out_ckpt} (step {step})")

    # --- final checkpoint ---
    save_checkpoint(cfg.out_ckpt, model, optimizer, cfg.train_steps, cfg)
    writer.close()
    print(f"[pretrain] done. Final checkpoint → {cfg.out_ckpt}")


if __name__ == "__main__":
    main()
