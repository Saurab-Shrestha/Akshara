"""
Multi-Head Attention with RoPE + Causal Masking
================================================

WHY THIS EXISTS
---------------
Attention is the mechanism that lets every token in a sequence look at
every other token and decide what to "pay attention to."

Without attention, each token is processed in isolation — the model has
no way to resolve that "छ" at the end of "नेपाल राम्रो छ" refers to
the whole phrase, not just itself.

HOW ATTENTION WORKS (step by step)
------------------------------------
For each token, we create three vectors from its embedding:
  Q (Query)  — "what am I looking for?"
  K (Key)    — "what do I contain?"
  V (Value)  — "what do I actually pass forward if attended to?"

Then for each token i:
  1. Compute score(i, j) = dot(Q_i, K_j) / sqrt(head_dim)
     → how relevant is token j to token i?
  2. Apply causal mask: set score(i, j) = -inf for j > i
     → token i cannot look at future tokens (no cheating)
  3. Softmax over scores → attention weights (sum to 1)
  4. Output = sum of V_j weighted by attention weights
     → a blend of all past tokens, weighted by relevance

MULTI-HEAD
----------
Instead of one set of Q/K/V, we run H parallel "heads" each with
dim = n_embed / n_heads. Each head can specialize:
  - Head 1 might track subject-verb agreement
  - Head 2 might track conjunct character dependencies
  - Head 3 might track punctuation boundaries
  etc.

Outputs from all heads are concatenated and projected back to n_embed.

WHERE ROPE FITS IN
------------------
Before computing dot(Q, K), we rotate Q and K using RoPE.
This injects position into the similarity score without touching V.
After rotation: dot(Q_i, K_j) naturally encodes the distance (i - j).

GROUPED QUERY ATTENTION (GQA)
------------------------------
Standard attention: each of the H query heads has its own K and V head.
Total K/V heads = H.

GQA (used in Qwen, LLaMA-3, Mistral): multiple query heads SHARE one
K/V head. If H=8 query heads and n_kv_heads=2, then 4 query heads
share each K/V head.

Why: K and V are large memory consumers (the KV cache at inference).
GQA reduces that memory 4× with minimal accuracy loss.
We use n_kv_heads = n_heads // 4 (one KV head per 4 query heads).

SHAPES THROUGH THE LAYER
-------------------------
Input x:        (batch, seq_len, n_embed)
Q, K, V proj:   (batch, seq_len, n_heads * head_dim)    [Q]
                (batch, seq_len, n_kv_heads * head_dim) [K, V]
After reshape:  (batch, n_heads, seq_len, head_dim)     [Q]
                (batch, n_kv_heads, seq_len, head_dim)  [K, V]
After attention:(batch, n_heads, seq_len, head_dim)
After concat:   (batch, seq_len, n_embed)
After out proj: (batch, seq_len, n_embed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rope import apply_rope


class MultiHeadAttention(nn.Module):

    def __init__(self, n_embed: int, n_heads: int, n_kv_heads: int, max_seq_len: int):
        """
        Args:
            n_embed:     total embedding dimension
            n_heads:     number of query heads
            n_kv_heads:  number of key/value heads (< n_heads for GQA)
            max_seq_len: maximum sequence length (for causal mask)
        """
        super().__init__()
        assert n_embed % n_heads == 0, "n_embed must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = n_embed // n_heads

        # How many query heads share each KV head
        self.n_rep = n_heads // n_kv_heads

        # Projections — bias=False is standard in modern LLMs (saves params, works fine)
        self.wq = nn.Linear(n_embed, n_heads    * self.head_dim, bias=False)
        self.wk = nn.Linear(n_embed, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(n_embed, n_kv_heads * self.head_dim, bias=False)

        # Output projection: concatenated heads → n_embed
        self.wo = nn.Linear(n_heads * self.head_dim, n_embed, bias=False)

        # Causal mask: lower-triangular matrix of True.
        # Built for 3× max_seq_len to cover visual prefix + text sequence
        # without recomputing at runtime.
        _mask_len = max_seq_len * 3
        mask = torch.tril(torch.ones(_mask_len, _mask_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (batch, seq_len, n_embed)
            cos: (seq_len, head_dim // 2) — RoPE cosine table
            sin: (seq_len, head_dim // 2) — RoPE sine table

        Returns:
            (batch, seq_len, n_embed)
        """
        B, T, _ = x.shape

        # ── Project to Q, K, V ──────────────────────────────────────────────
        q = self.wq(x)  # (B, T, n_heads * head_dim)
        k = self.wk(x)  # (B, T, n_kv_heads * head_dim)
        v = self.wv(x)  # (B, T, n_kv_heads * head_dim)

        # Reshape into (B, T, n_heads, head_dim) for RoPE and attention
        q = q.view(B, T, self.n_heads,    self.head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        # ── Apply RoPE to Q and K (not V — V carries content, not position) ─
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # ── Expand K and V for GQA ──────────────────────────────────────────
        # Each KV head needs to be repeated n_rep times to match query heads
        # (B, T, n_kv_heads, head_dim) → (B, T, n_heads, head_dim)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # ── Rearrange for batch matrix multiply: put heads before seq_len ───
        # (B, T, n_heads, head_dim) → (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ── Scaled dot-product attention ─────────────────────────────────────
        # scores: (B, n_heads, T, T)
        scale  = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask: positions where mask is False get -inf
        # (future tokens become invisible after softmax)
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))

        # Softmax → attention weights (each row sums to 1)
        weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        # (B, n_heads, T, T) × (B, n_heads, T, head_dim) → (B, n_heads, T, head_dim)
        out = torch.matmul(weights, v)

        # ── Merge heads and project ──────────────────────────────────────────
        # (B, n_heads, T, head_dim) → (B, T, n_heads * head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.wo(out)  # (B, T, n_embed)


# ── self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.models.rope import precompute_freqs_cis

    batch, seq_len, n_embed = 2, 16, 256
    n_heads, n_kv_heads     = 8, 2   # GQA: 4 query heads per KV head

    attn      = MultiHeadAttention(n_embed, n_heads, n_kv_heads, max_seq_len=seq_len)
    cos, sin = precompute_freqs_cis(dim=n_embed // n_heads, max_seq_len=seq_len)

    x   = torch.randn(batch, seq_len, n_embed)
    out = attn(x, cos, sin)

    assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"

    # Causal check: output at position 0 should not change when we alter position 5
    # because position 0 cannot attend to position 5 (future)
    x2 = x.clone()
    x2[:, 5, :] += 100.0          # large change at position 5
    out2 = attn(x2, cos, sin)
    diff_at_pos0 = (out[:, 0, :] - out2[:, 0, :]).abs().max().item()
    assert diff_at_pos0 < 1e-4, f"position 0 was affected by position 5 — causal mask broken! diff={diff_at_pos0}"

    print("MultiHeadAttention self-check passed.")
    print(f"  input  shape : {x.shape}")
    print(f"  output shape : {out.shape}")
    print(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={n_embed // n_heads}")
    print(f"  GQA ratio    : {n_heads // n_kv_heads} query heads per KV head")
    print(f"  causal check : position 0 unaffected by position 5 ✅ (diff={diff_at_pos0:.2e})")
    params = sum(p.numel() for p in attn.parameters())
    print(f"  params       : {params:,}")
