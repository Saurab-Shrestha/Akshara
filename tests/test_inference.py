"""
Cached inference: GDN math, KV cache, rollback, sampling, speculative decoding.

    PYTHONPATH=. pytest -q tests/test_inference.py
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.gdn_block import recurrent_gated_delta_rule  # noqa: E402
from src.models.hybrid_decoder import HybridDecoder, probs_from_logits  # noqa: E402
from src.speculative import speculative_generate  # noqa: E402

CFG = dict(vocab_size=300, n_embed=64, n_heads=4, n_kv_heads=2, n_layers=8, max_seq_len=64,
           attn_every=4, qk_norm=True, gdn_impl="native")


def model(seed, **kw):
    torch.manual_seed(seed)
    m = HybridDecoder(**{**CFG, **kw}).double().eval()
    with torch.no_grad():                     # make the random model's outputs less flat
        m.token_embed.weight.mul_(20)
    return m


def full_logits(m, ids):
    return m(ids)[0]


def test_gdn_matches_fla_reference():
    """Transcription of fla.ops.gated_delta_rule.naive (FLA 0.6) — the CUDA kernel's math."""
    torch.manual_seed(0)
    B, T, H, K = 2, 7, 3, 8
    q, k = (torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1) for _ in range(2))
    v, g, beta = torch.randn(B, T, H, K), torch.nn.functional.logsigmoid(torch.randn(B, T, H)), torch.rand(B, T, H)
    h0 = torch.randn(B, H, K, K)
    o, S, _ = recurrent_gated_delta_rule(q, k, v, g, beta, h0)

    qq, kk, vv, bb, gg = (x.transpose(1, 2) for x in (q, k, v, beta, g))
    h, ref = h0.clone(), torch.zeros(B, H, T, K)
    for i in range(T):
        h = h * gg[:, :, i].exp()[..., None, None]
        b_v = (vv[:, :, i] - (h * kk[:, :, i, :, None]).sum(-2)) * bb[:, :, i, None]
        h = h + kk[:, :, i, :, None] * b_v[:, :, None, :]
        ref[:, :, i] = torch.einsum("bhd,bhdm->bhm", qq[:, :, i] * K ** -0.5, h)
    assert torch.allclose(o, ref.transpose(1, 2), atol=1e-5) and torch.allclose(S, h, atol=1e-5)


def test_cached_forward_matches_full_forward():
    m = model(0)
    ids = torch.randint(0, 300, (2, 20))
    cache = m.new_cache()
    parts = [m.forward_cached(ids[:, :8], cache)]                       # prefill
    parts += [m.forward_cached(ids[:, t:t + 1], cache) for t in range(8, 14)]   # 1-token steps
    parts += [m.forward_cached(ids[:, 14:], cache)]                     # multi-token chunk on a cache
    assert torch.allclose(torch.cat(parts, 1), full_logits(m, ids), atol=1e-8)


def test_rollback():
    m = model(1)
    ids = torch.randint(0, 300, (1, 16))
    a = m.new_cache()
    m.forward_cached(ids[:, :10], a)
    m.forward_cached(ids[:, 10:15], a, keep_states=True)
    m.rollback(a, 2)                                                    # keep tokens 10, 11
    got = m.forward_cached(ids[:, 15:16], a)
    want = full_logits(m, torch.cat([ids[:, :12], ids[:, 15:16]], 1))[:, -1:]
    assert torch.allclose(got, want, atol=1e-8)
    m.forward_cached(ids[:, :3], b := m.new_cache())
    m.forward_cached(ids[:, 3:6], b, keep_states=True)
    m.rollback(b, 0)
    assert b["pos"] == 3 and torch.allclose(m.forward_cached(ids[:, 6:7], b),
                                            full_logits(m, torch.cat([ids[:, :3], ids[:, 6:7]], 1))[:, -1:])


def test_generate_greedy_matches_uncached():
    m = model(2)
    ids = torch.randint(0, 300, (2, 5))
    out = m.generate(ids, 12, temperature=0.0)
    ref = ids
    for _ in range(12):
        ref = torch.cat([ref, full_logits(m, ref)[:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(out, ref)


def test_speculative_greedy_equals_target():
    target, draft = model(3), model(4, n_layers=4)
    for seed in range(3):
        ids = torch.randint(0, 300, (1, 4), generator=torch.Generator().manual_seed(seed))
        spec, stats = speculative_generate(target, draft, ids, 20, k=4, temperature=0.0)
        assert torch.equal(spec, target.generate(ids, 20, temperature=0.0))
        assert stats["target_calls"] <= 21
    _, stats = speculative_generate(target, target, ids, 20, k=4)      # draft == target
    assert stats["accepted"] == stats["proposed"]


def test_speculative_sampling_matches_target_distribution():
    """First generated token: speculative sampling must reproduce the target's
    distribution even though the draft proposes from a different one."""
    target, draft = model(5, vocab_size=8), model(6, vocab_size=8, n_layers=4)
    ids = torch.tensor([[1, 2, 3]])
    want = probs_from_logits(full_logits(target, ids)[:, -1], 1.0)[0]
    torch.manual_seed(0)
    n = 2000
    counts = torch.bincount(torch.cat([speculative_generate(target, draft, ids, 1, k=3, temperature=1.0)[0][0, -1:]
                                       for _ in range(n)]), minlength=8).double() / n
    assert (counts - want).abs().max() < 0.04, (counts, want)


def test_top_p():
    p = probs_from_logits(torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]])), 1.0, top_p=0.75)
    assert torch.allclose(p, torch.tensor([[0.625, 0.375, 0.0, 0.0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA kernels need CUDA")
def test_fla_kernel_matches_reference_on_gpu():
    """Same weights through FLA's chunk kernel and the PyTorch recurrence (run on Kaggle/A100)."""
    from src.models.gdn_block import _HAS_FLA, GatedDeltaNetLayer
    if not _HAS_FLA:
        pytest.skip("flash-linear-attention not installed")
    torch.manual_seed(0)
    layer = GatedDeltaNetLayer(256, 4, use_fla=True, impl="native").cuda()
    x = torch.randn(2, 300, 256, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fast, S_fast = layer(x, return_state=True)
    layer.use_fla = False
    ref, S_ref = layer(x, return_state=True)
    assert (fast.float() - ref).abs().max() < 5e-2 and (S_fast.float() - S_ref).abs().max() < 5e-2
