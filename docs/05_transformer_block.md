# 05 · Transformer Block

One full standard layer: pre-norm attention + pre-norm SwiGLU MLP + residuals.

**File:** [`src/models/transformer_block.py`](../src/models/transformer_block.py)

---

## The code

```python
class TransformerBlock(nn.Module):

    def __init__(self, n_embed, n_heads, n_kv_heads, max_seq_len):
        super().__init__()
        self.norm1 = RMSNorm(n_embed)
        self.attn  = MultiHeadAttention(n_embed, n_heads, n_kv_heads, max_seq_len)
        self.norm2 = RMSNorm(n_embed)
        self.mlp   = SwiGLU(n_embed)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.norm1(x), freqs_cis)   # attention sub-layer
        x = x + self.mlp(self.norm2(x))                # MLP sub-layer
        return x
```

That's the entire file. 15 lines of logic.

---

## What each line does

**`self.norm1 = RMSNorm(n_embed)`**
Pre-norm before attention. x enters at full scale; attention sees a normalized view.

**`self.attn = MultiHeadAttention(...)`**
GQA + RoPE + causal attention. The "communication" step — tokens see each other.

**`x = x + self.attn(self.norm1(x), freqs_cis)`**
Pre-norm: `self.norm1(x)` normalizes before attn. The `+` adds the residual.
The original x is preserved along the residual path — gradient flows freely.

**`self.norm2 = RMSNorm(n_embed)`**
Pre-norm before MLP. Independent from norm1 (separate learned weights).

**`self.mlp = SwiGLU(n_embed)`**
The "computation" step — each token processed independently, no cross-token info.

**`x = x + self.mlp(self.norm2(x))`**
Same residual pattern. Output of the block is the same shape as input.

---

## Parameter count

At our config (n_embed=768, n_heads=12, n_kv_heads=3, max_seq_len=512):

| Component | Params |
|---|---|
| RMSNorm × 2 | 768 × 2 = 1,536 |
| MultiHeadAttention (wq, wk, wv, wo) | 768×768 + 768×192 + 768×192 + 768×768 = 1,327,104 |
| SwiGLU (w_gate, w_hidden, w_out) | 3 × 768 × 2,048 = 4,718,592 |
| **Total** | **~705k** |

Multiply by 3 TransformerBlocks in the hybrid decoder = ~2.1M for all attention layers.

---

## Role in the 3:1 hybrid

In `HybridDecoder`, this block appears at positions `[3, 7, 11]` (every 4th
layer). It's the "exact recall checkpoint" — after 3 GDN layers may have
compressed some information, this full-attention layer can look back at the
exact token sequence and correct any drift.

---

## Self-check result

```
TransformerBlock self-check passed.
  input  shape : (2, 12, 256)
  output shape : (2, 12, 256)
  params       : 705,024
```

---

**Next:** [06 · GDN Block](06_gdn_block.md)
