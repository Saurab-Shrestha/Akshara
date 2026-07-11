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
This naive sequential implementation is correct but slow (Python loop over T).
On Kaggle T4 (CUDA), replace this with fla-org/flash-linear-attention's
GatedDeltaNet which uses a chunked Triton kernel: O(T) amortized, fast.
Swap: `from fla.layers import GatedDeltaNet` and use it directly.
This PyTorch version is for understanding and Mac/CPU development.

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

# Optional: FLA chunked Triton kernel for GDN (fast on GPU, not needed on CPU/Mac).
# Install with: pip install flash-linear-attention
try:
    from fla.layers import GatedDeltaNet as _FLA_GatedDeltaNet
    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False


class GatedDeltaNetLayer(nn.Module):
    """
    The GDN recurrence layer — replaces self-attention.
    Input/output shape: (batch, seq_len, d_model)  [same as attention]
    """

    def __init__(self, d_model: int, n_heads: int, use_fla: bool = False):
        """
        Args:
            d_model:  embedding dimension
            n_heads:  number of parallel memory heads
            use_fla:  if True and flash-linear-attention is installed,
                      use FLA's chunked Triton kernel instead of the
                      pure-Python recurrence (much faster on GPU).
        """
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.use_fla = use_fla and _HAS_FLA

        if self.use_fla:
            self._fla_layer = _FLA_GatedDeltaNet(
                mode="chunk",
                hidden_size=d_model,
                head_dim=d_model // n_heads,
                num_heads=n_heads,
                expand_v=1.0,
                use_gate=True,
                use_short_conv=False,
            )
            return

        # Project input to q, k, v — same as attention
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        # β (beta): how aggressively to apply the delta update
        # One scalar per head per token → sigmoid → (0, 1)
        self.w_beta = nn.Linear(d_model, n_heads, bias=False)

        # α (alpha): forget gate — how much of old state to keep
        # One scalar per head per token → sigmoid → (0, 1)
        # bias=+4 so alpha starts at sigmoid(4)≈0.98: memory survives ~50 tokens
        # at init instead of halving every step (sigmoid(0)=0.5 → memoryless).
        self.w_alpha = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.w_alpha.bias, 4.0)

        # Output norm + projection (stabilizes the state output)
        self.out_norm = RMSNorm(d_model)
        self.wo       = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, state: dict | None = None) -> torch.Tensor:
        """
        Full-sequenze forward with initial state (for training or prefill).

        When ``use_fla=True`` and FLA is installed, delegates to FLA's
        chunked Triton kernel for much faster GPU training.

        Args:
            x:     (batch, seq_len, d_model)
            state: optional dict with 'S' — (B, H, D, D) initial memory.
                   If None, starts from zero (standard training path).
        Returns:
            (batch, seq_len, d_model)
        """
        if self.use_fla:
            out, _, _ = self._fla_layer(x)
            if getattr(self, '_capture_state', False):
                self._final_state = None
            return out

        B, T, _ = x.shape
        H, D = self.n_heads, self.d_head

        q = self.wq(x).view(B, T, H, D).float()
        k = self.wk(x).view(B, T, H, D).float()
        v = self.wv(x).view(B, T, H, D).float()
        k = F.normalize(k, dim=-1)
        beta  = torch.sigmoid(self.w_beta(x)).unsqueeze(-1).float()
        alpha = torch.sigmoid(self.w_alpha(x)).unsqueeze(-1).float()

        S = (state['S'].float() if state is not None and 'S' in state
             else torch.zeros(B, H, D, D, device=x.device, dtype=torch.float32))

        outputs = []
        for t in range(T):
            kt = k[:, t]; vt = v[:, t]; qt = q[:, t]
            bt = beta[:, t]; at = alpha[:, t]
            S_kt = torch.matmul(S, kt.unsqueeze(-1)).squeeze(-1)
            delta = torch.matmul(
                (vt - S_kt).unsqueeze(-1), kt.unsqueeze(-2))
            S = at.unsqueeze(-1) * S + bt.unsqueeze(-1) * delta
            outputs.append(torch.matmul(S, qt.unsqueeze(-1)).squeeze(-1))

        out = torch.stack(outputs, dim=1)
        out = out.contiguous().view(B, T, -1).type_as(x)

        if getattr(self, '_capture_state', False):
            self._final_state = {'S': S.detach().clone()}

        return self.wo(self.out_norm(out))

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: dict | None = None) -> tuple[torch.Tensor, dict]:
        """
        Single-timestep forward for autoregressive decoding.

        Processes one token through the GDN recurrence, updating the
        memory matrix S in-place.  This is O(1) per step (vs O(T) for
        the full forward loop).

        Falls back to the pure-Python recurrence in all cases (FLA's
        kernel does not expose a single-step interface).

        Args:
            x_t:   (B, d_model) — single token embedding
            state: dict with 'S' (B, H, D, D) or None (starts zero)

        Returns:
            (output_t, new_state) where output_t is (B, d_model)
        """
        B = x_t.shape[0]
        H, D = self.n_heads, self.d_head

        q_t = self.wq(x_t).view(B, H, D).float()
        k_t = self.wk(x_t).view(B, H, D).float()
        v_t = self.wv(x_t).view(B, H, D).float()
        k_t = F.normalize(k_t, dim=-1)
        beta_t  = torch.sigmoid(self.w_beta(x_t)).unsqueeze(-1).float()
        alpha_t = torch.sigmoid(self.w_alpha(x_t)).unsqueeze(-1).float()

        S = (state['S'].float() if state is not None and 'S' in state
             else torch.zeros(B, H, D, D, device=x_t.device, dtype=torch.float32))

        S_kt = torch.matmul(S, k_t.unsqueeze(-1)).squeeze(-1)
        delta = torch.matmul((v_t - S_kt).unsqueeze(-1), k_t.unsqueeze(-2))
        S = alpha_t.unsqueeze(-1) * S + beta_t.unsqueeze(-1) * delta

        o_t = torch.matmul(S, q_t.unsqueeze(-1)).squeeze(-1)
        out = o_t.reshape(B, -1).type_as(x_t)
        out = self.wo(self.out_norm(out))
        return out, {'S': S}


class GDNBlock(nn.Module):
    """
    Full GDN block = GatedDeltaNetLayer + SwiGLU MLP + residuals + pre-norm.
    Drop-in replacement for TransformerBlock in the 3:1 hybrid decoder.
    """

    def __init__(self, n_embed: int, n_heads: int, use_fla: bool = False):
        """
        Args:
            n_embed:  embedding dimension
            n_heads:  number of GDN memory heads
            use_fla:  use FLA Triton kernel if available (GPU training speedup)
        """
        super().__init__()
        self.norm1 = RMSNorm(n_embed)
        self.gdn   = GatedDeltaNetLayer(n_embed, n_heads, use_fla=use_fla)
        self.norm2 = RMSNorm(n_embed)
        self.mlp   = SwiGLU(n_embed)

    def forward(self, x: torch.Tensor, cos: torch.Tensor = None, sin: torch.Tensor = None,
                state: dict | None = None) -> torch.Tensor:
        """
        Full-sequence forward.  Accepts optional initial GDN state.

        When ``_capture_state`` is set on the GDN layer before calling, the
        final memory state is saved in ``_final_state`` for cached generation.

        Args:
            x:         (batch, seq_len, n_embed)
            cos, sin:  unused (interface compat with TransformerBlock)
            state:     optional GDN state dict from a prior forward

        Returns:
            (batch, seq_len, n_embed)
        """
        x = x + self.gdn(self.norm1(x), state=state)
        x = x + self.mlp(self.norm2(x))
        return x

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: dict | None = None) -> tuple[torch.Tensor, dict]:
        """
        Single-timestep forward for autoregressive decoding.

        Args:
            x_t:   (B, n_embed) — single token
            state: GDN state dict or None

        Returns:
            (output_t, new_state)
        """
        gdn_out, new_state = self.gdn.step(self.norm1(x_t), state)
        x_t = x_t + gdn_out
        x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t, new_state


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
