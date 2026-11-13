"""
End-to-end smoke test for scripts/pretrain.py: a tiny model trains a few steps
on CPU from generated .bin data, using the v2 recipe (Muon, WSD, anneal switch),
then resumes from its own checkpoint. Catches breakage before a GPU session is
spent on it.

    PYTHONPATH=. pytest -q tests/
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pretrain import decay_start, unwrap, wsd_lr  # noqa: E402
from src.models.hybrid_decoder import HybridDecoder  # noqa: E402

TINY = dict(vocab_size=49152, n_embed=64, n_heads=2, n_kv_heads=1, n_layers=4,
            max_seq_len=32, attn_every=4, use_fla=False, qk_norm=True, gdn_impl="native")


def test_wsd_schedule():
    kw = dict(warmup_steps=10, max_steps=100, lr=1.0, min_lr=0.1, decay_frac=0.2)
    assert decay_start(10, 100, 0.2) == 80
    assert wsd_lr(0, **kw) == 0.0 and wsd_lr(5, **kw) == 0.5
    assert wsd_lr(10, **kw) == wsd_lr(79, **kw) == 1.0          # stable phase
    assert abs(wsd_lr(90, **kw) - 0.55) < 1e-9                   # halfway through decay
    assert abs(wsd_lr(100, **kw) - 0.1) < 1e-9


def test_unwrap_strips_compile_prefix():
    m = HybridDecoder(**TINY)
    keys = set(unwrap(torch.compile(m)).state_dict())             # compile is lazy: no compilation here
    assert keys == set(m.state_dict()) and not any(k.startswith("_orig_mod.") for k in keys)


def test_checkpoint_loads_into_longer_context():
    """RoPE tables are not saved, so 2k weights load into a 16k / higher-theta model."""
    sd = HybridDecoder(**TINY).state_dict()
    assert not any(k.startswith("freqs_") for k in sd)
    long = HybridDecoder(**dict(TINY, max_seq_len=256, rope_theta=1e6))
    long.load_state_dict(sd)
    assert long.freqs_cos.shape[0] == 256 * 4


def test_chunked_ce_matches_full():
    torch.manual_seed(0)
    m = HybridDecoder(**dict(TINY, vocab_size=1000))
    ids, t = torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32))
    t[0, -5:] = -100
    _, full = m(ids, t)
    full.backward()
    g_full = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    m.ce_chunk = 20                                               # uneven last chunk on purpose
    logits, chunked = m(ids, t)
    chunked.backward()
    assert logits is None and torch.allclose(full, chunked, atol=1e-6)
    assert all(torch.allclose(a, p.grad, atol=1e-6) for a, p in zip(g_full, m.parameters()))


def test_z_loss_and_no_wd_embed():
    from types import SimpleNamespace
    from src.optim import build_optimizer
    torch.manual_seed(0)
    m = HybridDecoder(**dict(TINY, vocab_size=1000))
    ids, t = torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32))
    _, base = m(ids, t)
    m.z_loss = 1e-2
    logits, with_z = m(ids, t)
    z = 1e-2 * torch.logsumexp(logits.float(), -1).square().mean()
    assert torch.allclose(with_z, base + z, atol=1e-5)
    m.ce_chunk = 20                                               # chunked path adds the same z-loss
    assert torch.allclose(m(ids, t)[1], with_z, atol=1e-5)

    cfg = SimpleNamespace(optimizer="muon", lr=1e-3, weight_decay=0.1, no_wd_embed=True)
    adamw = build_optimizer(m, cfg, "cpu").opts[1]
    emb = m.token_embed.weight
    assert all(id(p) != id(emb) for p in adamw.param_groups[0]["params"])   # decayed group
    assert any(id(p) == id(emb) for p in adamw.param_groups[1]["params"])   # no-decay group


def _bin(path: Path, n_tokens: int, seed: int):
    rng = np.random.default_rng(seed)
    rng.integers(16, 2000, n_tokens, dtype=np.uint16).tofile(path)


def test_pretrain_runs_switches_to_anneal_and_resumes(tmp_path):
    for name, seed in [("train.bin", 0), ("val.bin", 1), ("anneal.bin", 2)]:
        _bin(tmp_path / name, 20_000, seed)
    ckpt = tmp_path / "ck.pt"
    cfg = dict(TINY, device="cpu", tokenizer=str(ROOT / "tokenizer/akshara-bpe"),
               train_path=str(tmp_path / "train.bin"), dev_path=str(tmp_path / "val.bin"),
               anneal_path=str(tmp_path / "anneal.bin"),
               batch_size=2, grad_accum=1, train_steps=12, warmup_steps=2,
               schedule="wsd", decay_frac=0.25, optimizer="muon", lr=1e-3, min_lr=1e-4,
               eval_steps=6, eval_iters=1, save_every=6, use_amp=False,
               use_gradient_checkpointing=False, compile=False, ce_chunk=16, no_wd_embed=True, z_loss=1e-4,
               out_ckpt=str(ckpt), log_dir=str(tmp_path / "logs"), hf_repo=None)
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))

    def run(*extra):
        r = subprocess.run([sys.executable, "scripts/pretrain.py", "--config", str(tmp_path / "cfg.json"), *extra],
                           cwd=ROOT, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        return r.stdout

    out = run()
    assert "switching data" in out and "anneal.bin" in out        # decay starts at step 9
    assert "[eval]" in out

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert ck["step"] == 12
    pre = torch.load(tmp_path / "ck_predecay_9.pt", map_location="cpu", weights_only=False)
    assert pre["step"] == 9                                        # = decay start, before any decay step
    HybridDecoder(**TINY).load_state_dict(ck["model_state_dict"])  # plain keys, loads strictly

    # resume past the decay start: must begin on the anneal data straight away
    cfg["train_steps"] = 16
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    out = run("--resume", str(ckpt))
    assert "resumed from" in out and "at step 12" in out
    assert torch.load(ckpt, map_location="cpu", weights_only=False)["step"] == 16
