"""
Language pretraining for the Akshara decoder.

WHY pretrain before OCR fine-tuning:
    The HybridDecoder is trained here on raw Nepali text (no images) so it
    acquires a strong prior on Devanagari script, word-piece boundaries, and
    sentence structure.  When OCR fine-tuning starts, the decoder already knows
    *what* valid Nepali looks like; the vision encoder only needs to teach it
    *which* characters appear in the image.  Without this stage the model must
    simultaneously learn language and vision, which is harder and slower.

TRAINING RECIPE (v2):
    - Own 48k BPE tokenizer (cfg.tokenizer) + packed uint16 .bin shards.
    - Muon on hidden matrices + AdamW on embeddings/norms (cfg.optimizer).
    - LR schedule: cosine (v1) or WSD (v2): warmup, a long flat phase, then a
      linear decay over the last decay_frac of steps on the anneal_path data.
    - bf16 autocast on A100; FLA Triton kernel for GDN.
    - Periodic eval (perplexity), checkpoint saving, HF Hub backup.

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
from src.optim import build_optimizer


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


def wsd_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float,
           decay_frac: float) -> float:
    """
    Warmup-Stable-Decay (MiniCPM, 2024): linear warmup, hold at ``lr``, then decay
    linearly to ``min_lr`` over the last ``decay_frac`` of steps.

    WHY: the flat phase does not depend on the total length, so a run can be
    extended mid-way (only the decay start moves), and the decay phase is where
    the high-quality annealing data goes. Matches or beats cosine at equal compute.
    """
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    start = decay_start(warmup_steps, max_steps, decay_frac)
    if step < start:
        return lr
    return lr + (min_lr - lr) * min(1.0, (step - start) / max(1, max_steps - start))


def decay_start(warmup_steps: int, max_steps: int, decay_frac: float) -> int:
    return max(warmup_steps, max_steps - int(decay_frac * max_steps))


def lr_at(step: int, cfg) -> float:
    if cfg.schedule == "wsd":
        return wsd_lr(step, cfg.warmup_steps, cfg.train_steps, cfg.lr, cfg.min_lr, cfg.decay_frac)
    if cfg.schedule == "cosine":
        return cosine_lr(step, cfg.warmup_steps, cfg.train_steps, cfg.lr, cfg.min_lr)
    raise ValueError(f"unknown schedule {cfg.schedule!r}")


def data_path_at(step: int, cfg) -> str:
    """The anneal set replaces the main corpus once the WSD decay starts."""
    if (cfg.schedule == "wsd" and cfg.anneal_path
            and step >= decay_start(cfg.warmup_steps, cfg.train_steps, cfg.decay_frac)):
        if os.path.exists(cfg.anneal_path):
            return cfg.anneal_path
        print(f"[pretrain] WARNING anneal_path {cfg.anneal_path} missing; decaying on train_path")
    return cfg.train_path


def unwrap(model: nn.Module) -> nn.Module:
    """Strip DataParallel / torch.compile wrappers so state_dict keys carry no
    'module.' / '_orig_mod.' prefix and load into a plain HybridDecoder."""
    while True:
        if isinstance(model, nn.DataParallel):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def amp_context(use_amp: bool, amp_dtype: str | None, device: str):
    """
    Return an autocast context manager, or a no-op if AMP is disabled.

    WHY bf16 over fp16:
        bf16 has the same exponent range as fp32, so gradients don't need a
        GradScaler. fp16 is for pre-Ampere GPUs (Kaggle T4/P100) without fast
        bf16; there small gradients underflow, so the loop scales the loss
        with a GradScaler (see main).
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
        losses.append(loss.mean().item())  # .mean(): DataParallel returns per-GPU losses
    model.train()
    return sum(losses) / max(1, len(losses))


def save_checkpoint(path: str, model: nn.Module, optimizer, step: int, cfg: PretrainConfig,
                    scaler=None) -> None:
    """Save a portable checkpoint dict to ``path``, then optionally push to HF Hub."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "step": step,
        "model_state_dict": unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": vars(cfg),
    }, path)
    _push_to_hub(path, cfg, step)


def _push_to_hub(path: str, cfg: PretrainConfig, step: int) -> None:
    """Upload the checkpoint to HF Hub if cfg.hf_repo is set. Never raises —
    a failed upload logs a warning and training continues (the local copy
    still exists, and the next save retries)."""
    repo = getattr(cfg, "hf_repo", None)
    if not repo:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo, repo_type="model", exist_ok=True, private=True)
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=os.path.basename(path),
            repo_id=repo,
            repo_type="model",
            commit_message=f"checkpoint step {step}",
        )
        print(f"  [hub] pushed {os.path.basename(path)} → {repo} (step {step})")
    except Exception as e:
        print(f"  [hub] upload failed (training continues, local copy kept): {e}")


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
    parser.add_argument("--hf_repo",     type=str,   default=None,
                        help="HF Hub repo (e.g. user/akshara-pretrain) to back up checkpoints to")
    parser.add_argument("--seed",        type=int,   default=None)
    parser.add_argument("--device",      type=str,   default=None)
    parser.add_argument("--tokenizer",   type=str,   default=None)
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
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, trust_remote_code=True)
    assert len(tokenizer) <= cfg.vocab_size, \
        f"tokenizer {cfg.tokenizer} has {len(tokenizer)} tokens > vocab_size {cfg.vocab_size}"

    # --- model ---
    model = HybridDecoder(
        vocab_size=cfg.vocab_size,
        n_embed=cfg.n_embed,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        n_layers=cfg.n_layers,
        max_seq_len=cfg.max_seq_len,
        attn_every=cfg.attn_every,
        use_fla=cfg.use_fla,
        qk_norm=cfg.qk_norm,
        gdn_impl=cfg.gdn_impl,
        rope_theta=cfg.rope_theta,
    )
    if cfg.use_gradient_checkpointing:
        model.set_gradient_checkpointing(True)
    model.ce_chunk = cfg.ce_chunk
    model.z_loss = cfg.z_loss
    model = model.to(device)

    # --- optional resume (before compile / DataParallel so keys match) ---
    start_step = 0
    ck = None
    if cfg.init_ckpt and not (args.resume and os.path.exists(args.resume)):
        # New stage from earlier weights (e.g. 2k → 16k context): weights only.
        # RoPE tables are rebuilt from this config's max_seq_len / rope_theta.
        init = torch.load(cfg.init_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        model.load_state_dict({k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in init.items()})
        print(f"[pretrain] initialised weights from {cfg.init_ckpt}")
        del init
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = {k.removeprefix("_orig_mod.").removeprefix("module."): v
              for k, v in ck["model_state_dict"].items()}
        model.load_state_dict(sd)
        start_step = ck.get("step", 0)
        print(f"[pretrain] resumed from {args.resume} at step {start_step}")

    # --- optimizer: built on the plain model (same param tensors as any wrapper) ---
    optimizer = build_optimizer(model, cfg, device)
    if ck is not None and "optimizer_state_dict" in ck:
        optimizer.load_state_dict(ck["optimizer_state_dict"])

    # fp16 only: scale the loss so small gradients don't underflow to zero; the
    # scale backs off when a step overflows (that step is skipped). Disabled =
    # every call below is a pass-through, so bf16/fp32 use the same code path.
    use_scaler = cfg.use_amp and cfg.amp_dtype == "fp16" and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if use_scaler and ck is not None and ck.get("scaler_state_dict"):
        scaler.load_state_dict(ck["scaler_state_dict"])
    opts = getattr(optimizer, "opts", (optimizer,))  # GradScaler needs real torch optimizers
    del ck

    if cfg.compile:
        torch.set_float32_matmul_precision("high")
        print("[pretrain] compiling model with torch.compile...")
        model = torch.compile(model)
        print("[pretrain] compilation done")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] model params: {n_params:,} (~{n_params / 1e6:.1f}M)")

    # --- multi-GPU: wrap in DataParallel if multiple GPUs available ---
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0
    if n_gpus > 1:
        print(f"[pretrain] {n_gpus}× GPU detected — using DataParallel (effective batch ×{n_gpus})")
        model = nn.DataParallel(model)
    raw_model = unwrap(model)  # for state_dict saving

    # --- data ---
    cur_path = data_path_at(start_step, cfg)
    train_loader = get_text_dataloader(
        cur_path, tokenizer, cfg.batch_size, cfg.max_seq_len, shuffle=True
    )
    dev_loader = get_text_dataloader(
        cfg.dev_path, tokenizer, cfg.batch_size, cfg.max_seq_len, shuffle=False
    ) if os.path.exists(cfg.dev_path) else None

    train_iter = iter(train_loader)

    # --- log dir + TensorBoard (optional — falls back to no-op if unavailable) ---
    os.makedirs(cfg.log_dir, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=cfg.log_dir)
    except Exception:
        class _NoOpWriter:
            def add_scalar(self, *a, **kw): pass
            def close(self): pass
        writer = _NoOpWriter()
        print("[pretrain] TensorBoard unavailable — console logging only")

    # --- training loop ---
    model.train()
    t0          = time.perf_counter()
    run_start   = time.perf_counter()
    tokens_per_step = cfg.batch_size * cfg.max_seq_len * cfg.grad_accum
    accum_loss  = 0.0

    for step in range(start_step, cfg.train_steps):
        lr = lr_at(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Keep the last pre-decay weights: resuming from them with a larger
        # train_steps continues the flat phase on more data, then decays again.
        # Step in the name: an extended run decays later and gets its own file.
        predecay = f"{os.path.splitext(cfg.out_ckpt)[0]}_predecay_{step}.pt"
        if (cfg.schedule == "wsd" and step == decay_start(cfg.warmup_steps, cfg.train_steps, cfg.decay_frac)
                and not os.path.exists(predecay)):
            save_checkpoint(predecay, raw_model, optimizer, step, cfg, scaler)
            print(f"  [ckpt] pre-decay checkpoint → {predecay} (step {step})")

        if (path := data_path_at(step, cfg)) != cur_path:
            print(f"[pretrain] step {step}: LR decay starts, switching data {cur_path} → {path}")
            cur_path = path
            train_loader = get_text_dataloader(path, tokenizer, cfg.batch_size, cfg.max_seq_len, shuffle=True)
            train_iter = iter(train_loader)

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
                # DataParallel returns one loss per GPU — reduce to scalar
                loss = loss.mean() / cfg.grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        for o in opts:
            scaler.unscale_(o)  # clip true gradients, not scaled ones
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for o in opts:
            scaler.step(o)      # skipped if the grads hold inf/nan
        scaler.update()

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
                ppl = math.exp(min(dev_loss, 20))
                print(f"  [eval] step {step} | train {train_loss:.4f} | dev {dev_loss:.4f} | ppl {ppl:.2f}")
                writer.add_scalar("eval/train_loss", train_loss, step)
                writer.add_scalar("eval/dev_loss",   dev_loss,   step)
                writer.add_scalar("eval/perplexity", ppl,        step)

        # --- periodic checkpoint ---
        if step > start_step and step % cfg.save_every == 0:
            # step+1: this step is done — resume should start at the next one
            save_checkpoint(cfg.out_ckpt, raw_model, optimizer, step + 1, cfg, scaler)
            print(f"  [ckpt] saved → {cfg.out_ckpt} (step {step})")

    # --- final checkpoint ---
    save_checkpoint(cfg.out_ckpt, raw_model, optimizer, cfg.train_steps, cfg, scaler)
    writer.close()
    print(f"[pretrain] done. Final checkpoint → {cfg.out_ckpt}")


if __name__ == "__main__":
    main()
