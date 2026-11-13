"""
l
=====================

WHY THIS EXISTS
---------------
This is the core innovation in Surya v2's architecture. Instead of standard
attention (which is O(T²) — expensive on long documents), GatedDeltaNet
maintains a fixed-size memory matrix that runs in O(T) time.

Three ideas combine here:

1. LINEAR ATTENTION (the base idea)
   Standard attention: output_t = sum(softmax(q·k) * v) for all past tokens
   Linear attention:   output_t = S_t · q_t   where S_t is a running sum
                       S_t = S_{t-1} + v_t ⊗ k_t  (outer product accumulation)
   Problem: S keeps accumulating everything — no forgetting. Old info piles up.

2. DELTA RULE (DeltaNet's fix)
   Instead of blindly adding v_t ⊗ k_t, first ERASE what S already knows
   about k_t, THEN write the new v_t:

       erase:  S_t = S_{t-1} - β_t * (S_{t-1} · k_t) ⊗ k_t
       write:  S_t = S_t + β_t * v_t ⊗ k_t

   Combined:   S_t = S_{t-1} + β_t * (v_t - S_{t-1}·k_t) ⊗ k_t
   (the code applies the forget gate below first, then the delta rule,
   exactly as FLA's kernel does: see recurrent_gated_delta_rule)

   This means: "replace what S thinks k_t means with what v_t says it means"
   β_t (beta) controls how aggressively to update (learned, 0→1).

3. GATING (Mamba2's fix applied to DeltaNet)
   Add a forget gate α_t that can wipe the entire state when needed:

       S_t = α_t * S_{t-1} + β_t * (v_t - S_{t-1}·k_t) ⊗ k_t

   α_t ≈ 1 → remember everything (default)
   α_t ≈ 0 → forget everything (start fresh for a new document)

   This is what makes GatedDeltaNet handle document boundaries and topic
   changes gracefully — crucial for OCR of multi-page documents.

THE STATE MATRIX
----------------
S has shape: (n_heads, d_head, d_head)
- n_heads parallel memories, each a d_head × d_head matrix
- For our config: 12 heads, d_head=64 → 12 × 64 × 64 = 49,152 values per token
- Standard attention KV cache: grows with T → unbounded
- GDN state: fixed size regardless of T → bounded memory at inference

READ/WRITE ANALOGY
------------------
Think of S as a whiteboard with n_heads sections:
  Write: β * outer(v, k)  → "write v at the location k points to"
  Erase: -β * outer(S·k, k) → "erase whatever was at k's location first"
  Read:  S · q            → "read what's stored at q's location"

OUTPUT
------
After reading: o_t = S_t · q_t
Then normalize + project back to d_model (like attention's output projection).

HARDWARE NOTE
-------------
recurrent_gated_delta_rule is a plain Python loop over T: correct, slow, and
used on CPU/Mac and for decoding. With use_fla=True on CUDA the same math runs
in FLA's chunked Triton kernel (chunk_gated_delta_rule) on the same weights.

IN THE 3:1 HYBRID
-----------------
This block replaces 3 out of every 4 TransformerBlocks in the decoder.
Every 4th layer is a standard TransformerBlock for exact recall.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rms_norm import RMSNorm
from src.models.swiglu import SwiGLU

# Optional FLA (flash-linear-attention) Triton kernels, CUDA only.
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk
    from fla.layers import GatedDeltaNet as _FLA_GatedDeltaNet
    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False


def recurrent_gated_delta_rule(q, k, v, g, beta, state=None, all_states=False):
    """
    Reference gated delta rule, same math as FLA's chunk kernel
    (fla.ops.gated_delta_rule.naive), so one set of weights runs on both.

        S_t = exp(g_t)·S_{t-1}                        decay (forget gate α = e^g)
        S_t = S_t + k_t ⊗ β_t·(v_t − k_tᵀS_t)         delta rule: replace what S
                                                       stores under key k_t with v_t
        o_t = (q_t/√K)ᵀ S_t                           read

    q, k: (B, T, H, K) L2-normalised · v: (B, T, H, V) · g, beta: (B, T, H)
    state: (B, H, K, V) or None. Returns o (B, T, H, V), final state, and with
    all_states=True the list of states after every token (speculative rollback).
    """
    B, T, H, K = q.shape
    dt = torch.promote_types(q.dtype, torch.float32)      # ≥ fp32 for the running memory
    q, k, v, g, beta = (t.to(dt) for t in (q, k, v, g, beta))
    q = q * K ** -0.5
    S = state.to(dt) if state is not None else q.new_zeros(B, H, K, v.shape[-1])
    outs, states = [], []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        u = beta[:, t, :, None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], S))
        S = S + k[:, t, :, :, None] * u[:, :, None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], S))
        if all_states:
            states.append(S)
    return torch.stack(outs, dim=1), S, states


class GatedDeltaNetLayer(nn.Module):
    """
    The GDN recurrence layer, replaces self-attention.
    Input/output shape: (batch, seq_len, d_model), like attention.

    impl="native" (v2): our own projections; on CUDA with FLA installed the
    recurrence runs in FLA's chunked Triton kernel, everywhere else in
    recurrent_gated_delta_rule. Same weights either way, so a checkpoint
    trained on a GPU loads and decodes on a Mac.
    impl="fla_layer" (v1 checkpoints): FLA's whole GatedDeltaNet module, own
    weights, GPU only, no single-step decode.
    """

    def __init__(self, d_model: int, n_heads: int, use_fla: bool = False, impl: str = "fla_layer"):
        super().__init__()
        assert d_model % n_heads == 0
        assert impl in ("native", "fla_layer"), impl
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.use_fla = use_fla and _HAS_FLA
        self.impl    = impl

        if impl == "fla_layer" and self.use_fla:
            self._fla_layer = _FLA_GatedDeltaNet(
                mode="chunk", hidden_size=d_model, head_dim=self.d_head, num_heads=n_heads,
                expand_v=1.0, use_gate=True, use_short_conv=False,
            )
            return

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        # β: how strongly to overwrite memory, one scalar per head per token
        self.w_beta = nn.Linear(d_model, n_heads, bias=False)
        # α = sigmoid(·): forget gate. bias +4 → α≈0.98 at init, memory survives
        # ~50 tokens instead of halving every step.
        self.w_alpha = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.w_alpha.bias, 4.0)
        self.out_norm = RMSNorm(d_model)
        self.wo       = nn.Linear(d_model, d_model, bias=False)

    def _project(self, x):
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_head
        q = F.normalize(self.wq(x).view(B, T, H, D), dim=-1)
        k = F.normalize(self.wk(x).view(B, T, H, D), dim=-1)
        v = self.wv(x).view(B, T, H, D)
        dt = torch.promote_types(x.dtype, torch.float32)   # gates in ≥ fp32
        g = F.logsigmoid(self.w_alpha(x).to(dt))           # log α
        beta = torch.sigmoid(self.w_beta(x).to(dt))
        return q, k, v, g, beta

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None,
                return_state: bool = False, all_states: bool = False):
        """
        x: (B, T, d_model). state: (B, H, K, V) memory from earlier tokens or None.
        return_state → (out, final_state); all_states → (out, [state after each token]).
        """
        if self.impl == "fla_layer" and self.use_fla:
            assert not (return_state or all_states or state is not None), \
                "v1 fla_layer GDN has no cached decoding"
            return self._fla_layer(x)[0]

        B, T, _ = x.shape
        q, k, v, g, beta = self._project(x)
        if self.use_fla and x.is_cuda and not all_states:
            o, S = _fla_chunk(q, k, v.to(q.dtype), g, beta.to(q.dtype), initial_state=state,
                              output_final_state=return_state)
            states = None
        else:
            o, S, states = recurrent_gated_delta_rule(q, k, v, g, beta, state, all_states)
        out = self.wo(self.out_norm(o.reshape(B, T, -1).type_as(x)))
        if all_states:
            return out, states
        return (out, S) if return_state else out


class GDNBlock(nn.Module):
    """GatedDeltaNetLayer + SwiGLU MLP, pre-norm residuals. Drop-in for TransformerBlock."""

    def __init__(self, n_embed: int, n_heads: int, use_fla: bool = False, impl: str = "fla_layer"):
        super().__init__()
        self.norm1 = RMSNorm(n_embed)
        self.gdn   = GatedDeltaNetLayer(n_embed, n_heads, use_fla=use_fla, impl=impl)
        self.norm2 = RMSNorm(n_embed)
        self.mlp   = SwiGLU(n_embed)

    def forward(self, x: torch.Tensor, cos: torch.Tensor = None, sin: torch.Tensor = None,
                cache: dict | None = None, keep_states: bool = False) -> torch.Tensor:
        """
        cos/sin unused (same signature as TransformerBlock). With ``cache`` (a dict,
        updated in place) the GDN memory continues from cache["S"]; keep_states also
        stores the memory after every token in cache["S_steps"] for rollback.
        """
        if cache is None:
            h = self.gdn(self.norm1(x))
        elif keep_states:
            cache["S_base"] = cache.get("S")
            h, cache["S_steps"] = self.gdn(self.norm1(x), cache.get("S"), all_states=True)
            cache["S"] = cache["S_steps"][-1]
        else:
            h, cache["S"] = self.gdn(self.norm1(x), cache.get("S"), return_state=True)
        x = x + h
        return x + self.mlp(self.norm2(x))


# ── self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    batch, seq_len, n_embed = 2, 12, 256
    n_heads = 8

    block = GDNBlock(n_embed=n_embed, n_heads=n_heads)
    x     = torch.randn(batch, seq_len, n_embed)
    out   = block(x)

    assert out.shape == x.shape, f"shape mismatch: {out.shape}"

    # Causal check: output at t=0 should not change when t=5 changes
    # (GDN processes left-to-right, t=0 state is computed before t=5 exists)
    x2 = x.clone()
    x2[:, 5, :] += 100.0
    out2 = block(x2)
    diff = (out[:, 0, :] - out2[:, 0, :]).abs().max().item()
    assert diff < 1e-4, f"causal violation: t=0 affected by t=5 (diff={diff})"

    params = sum(p.numel() for p in block.parameters())
    print("GDNBlock self-check passed.")
    print(f"  input shape  : {x.shape}")
    print(f"  output shape : {out.shape}")
    print(f"  causal check : t=0 unaffected by t=5 ✅ (diff={diff:.2e})")
    print(f"  params       : {params:,}")

    # Compare param count vs TransformerBlock at same size
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.models.transformer_block import TransformerBlock
    tb = TransformerBlock(n_embed, n_heads, n_kv_heads=2, max_seq_len=seq_len)
    tb_params = sum(p.numel() for p in tb.parameters())
    print(f"\n  GDNBlock params      : {params:,}")
    print(f"  TransformerBlock params: {tb_params:,}")
    print(f"  GDN is {params/tb_params:.2f}x the size of standard block")
