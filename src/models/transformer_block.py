"""
Transformer Block
=================

WHY THIS EXISTS
---------------
A transformer block is one "layer" of the decoder. The full decoder
stacks N of these blocks in sequence, each one refining the token
representations a little more.

Each block does exactly two things, in order:
  1. Attention  — let tokens look at each other (communication)
  2. MLP        — process each token independently (computation)

Both are wrapped with:
  - RMSNorm BEFORE (pre-norm): stabilizes inputs, trains faster
  - Residual (+) AFTER: preserves information from previous layers

PRE-NORM vs POST-NORM
---------------------
The original "Attention is All You Need" paper used post-norm
(normalize AFTER attention + residual). This caused training
instability at large scale.

Modern LLMs all use pre-norm: normalize BEFORE feeding into
attention/MLP. The residual connection then bypasses the norm,
keeping a clean gradient highway from output to input.

  pre-norm:  x = x + attn(norm(x))   ← norm sees clean x
  post-norm: x = norm(x + attn(x))   ← norm sees x + noisy attn output

BLOCK STRUCTURE
---------------
  Input x
    │
    ├─ RMSNorm → Attention ─────────────── (+) → x
    │                                       │
    └─ RMSNorm → SwiGLU MLP ───────────── (+) → x
    │
  Output x  (same shape as input)
"""

import torch
import torch.nn as nn

from src.models.attention import MultiHeadAttention
from src.models.swiglu import SwiGLU
from src.models.rms_norm import RMSNorm


class TransformerBlock(nn.Module):

    def __init__(self, n_embed: int, n_heads: int, n_kv_heads: int, max_seq_len: int):
        """
        Args:
            n_embed:     embedding dimension
            n_heads:     number of query attention heads
            n_kv_heads:  number of key/value heads (GQA)
            max_seq_len: maximum sequence length (passed to attention for causal mask)
        """
        super().__init__()
        # Pre-norm before attention
        self.norm1 = RMSNorm(n_embed)
        self.attn  = MultiHeadAttention(n_embed, n_heads, n_kv_heads, max_seq_len)

        # Pre-norm before MLP
        self.norm2 = RMSNorm(n_embed)
        self.mlp   = SwiGLU(n_embed)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (batch, seq_len, n_embed)
            freqs_cis:  RoPE rotation factors, passed through to attention

        Returns:
            (batch, seq_len, n_embed) — same shape, refined representations
        """
        # Attention sub-layer with residual
        # norm1(x) is normalized, attention runs on it, result added back to raw x
        x = x + self.attn(self.norm1(x), freqs_cis)

        # MLP sub-layer with residual
        x = x + self.mlp(self.norm2(x))

        return x


# ── self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.models.rope import precompute_freqs_cis

    batch, seq_len, n_embed = 2, 12, 256
    n_heads, n_kv_heads     = 8, 2

    block     = TransformerBlock(n_embed, n_heads, n_kv_heads, max_seq_len=seq_len)
    freqs_cis = precompute_freqs_cis(dim=n_embed // n_heads, max_seq_len=seq_len)

    x   = torch.randn(batch, seq_len, n_embed)
    out = block(x, freqs_cis)

    assert out.shape == x.shape, f"shape mismatch: {out.shape}"

    params = sum(p.numel() for p in block.parameters())
    print("TransformerBlock self-check passed.")
    print(f"  input  shape : {x.shape}")
    print(f"  output shape : {out.shape}")
    print(f"  params       : {params:,}")
