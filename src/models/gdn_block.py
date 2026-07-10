"""
Gated DeltaNet Block
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


class GatedDeltaNetLayer(nn.Module):
    """
    The GDN recurrence layer — replaces self-attention.
    Input/output shape: (batch, seq_len, d_model)  [same as attention]
    """

    def __init__(self, d_model: int, n_heads: int):
        """
        Args:
            d_model:  embedding dimension
            n_heads:  number of parallel memory heads
        """
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        # Project input to q, k, v — same as attention
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        # β (beta): how aggressively to apply the delta update
        # One scalar per head per token → sigmoid → (0, 1)
        self.w_beta = nn.Linear(d_model, n_heads, bias=False)

        # α (alpha): forget gate — how much of old state to keep
        # One scalar per head per token → sigmoid → (0, 1)
        self.w_alpha = nn.Linear(d_model, n_heads, bias=False)

        # Output norm + projection (stabilizes the state output)
        self.out_norm = RMSNorm(d_model)
        self.wo       = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_head

        # Project to q, k, v  →  reshape to (B, T, H, D)
        q = self.wq(x).view(B, T, H, D)
        k = self.wk(x).view(B, T, H, D)
        v = self.wv(x).view(B, T, H, D)

        # L2-normalize k so the state matrix doesn't grow unboundedly
        # k is the "address" in memory — unit vectors work best as addresses
        k = F.normalize(k, dim=-1)

        # Gates: (B, T, H) → unsqueeze for broadcasting → (B, T, H, 1)
        beta  = torch.sigmoid(self.w_beta(x)).unsqueeze(-1)   # update strength
        alpha = torch.sigmoid(self.w_alpha(x)).unsqueeze(-1)  # forget strength

        # ── Sequential recurrence over time ──────────────────────────────────
        # S: (B, H, D, D) — the memory matrix, one per head per batch
        # Initialized to zero at the start of each sequence
        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            kt = k[:, t]     # (B, H, D)
            vt = v[:, t]     # (B, H, D)
            qt = q[:, t]     # (B, H, D)
            bt = beta[:, t]  # (B, H, 1)
            at = alpha[:, t] # (B, H, 1)

            # What does S currently associate with kt?
            # S: (B, H, D, D), kt: (B, H, D) → (B, H, D, 1) → squeeze → (B, H, D)
            S_kt = torch.matmul(S, kt.unsqueeze(-1)).squeeze(-1)  # (B, H, D)

            # Delta update: erase S's current kt-memory, write vt instead
            # outer product: (B, H, D, 1) @ (B, H, 1, D) = (B, H, D, D)
            delta = torch.matmul(
                (vt - S_kt).unsqueeze(-1),  # (B, H, D, 1)
                kt.unsqueeze(-2)             # (B, H, 1, D)
            )

            # Gated state update:
            # α scales old state (forget), β scales the new delta (write)
            # at, bt: (B, H, 1) → unsqueeze → (B, H, 1, 1) for matrix broadcast
            S = at.unsqueeze(-1) * S + bt.unsqueeze(-1) * delta  # (B, H, D, D)

            # Read: project current state onto the query direction
            ot = torch.matmul(S, qt.unsqueeze(-1)).squeeze(-1)  # (B, H, D)

            outputs.append(ot)

        # Stack time steps: list of T × (B, H, D)  →  (B, T, H, D)
        out = torch.stack(outputs, dim=1)

        # Merge heads: (B, T, H, D) → (B, T, d_model)
        out = out.contiguous().view(B, T, -1)

        # Normalize and project
        return self.wo(self.out_norm(out))


class GDNBlock(nn.Module):
    """
    Full GDN block = GatedDeltaNetLayer + SwiGLU MLP + residuals + pre-norm.
    Drop-in replacement for TransformerBlock in the 3:1 hybrid decoder.
    """

    def __init__(self, n_embed: int, n_heads: int):
        """
        Args:
            n_embed:  embedding dimension
            n_heads:  number of GDN memory heads
        """
        super().__init__()
        self.norm1 = RMSNorm(n_embed)
        self.gdn   = GatedDeltaNetLayer(n_embed, n_heads)
        self.norm2 = RMSNorm(n_embed)
        self.mlp   = SwiGLU(n_embed)

    def forward(self, x: torch.Tensor, cos: torch.Tensor = None, sin: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:         (batch, seq_len, n_embed)
            cos, sin:  unused (GDN doesn't use RoPE). Accepted for interface
                       compatibility with TransformerBlock.
        Returns:
            (batch, seq_len, n_embed)
        """
        # GDN sub-layer with residual
        x = x + self.gdn(self.norm1(x))
        # MLP sub-layer with residual
        x = x + self.mlp(self.norm2(x))
        return x


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
