# 04 · Multi-Head Attention with GQA

The full attention implementation: GQA, RoPE, causal mask — all wired together.

**File:** [`src/models/attention.py`](../src/models/attention.py)

---

## Class signature

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_heads, n_kv_heads, max_seq_len):
```

| Parameter | Value (our config) | What it controls |
|---|---|---|
| `n_embed` | 768 | Total embedding dimension |
| `n_heads` | 12 | Query heads — how many parallel attention patterns |
| `n_kv_heads` | 3 | KV heads — shared across query groups (GQA) |
| `max_seq_len` | 512 | Causal mask size |

---

## Projection matrices

```python
self.wq = nn.Linear(n_embed, n_embed,               bias=False)  # 768 → 768
self.wk = nn.Linear(n_embed, n_kv_heads * head_dim, bias=False)  # 768 → 192
self.wv = nn.Linear(n_embed, n_kv_heads * head_dim, bias=False)  # 768 → 192
self.wo = nn.Linear(n_embed, n_embed,               bias=False)  # 768 → 768
```

`head_dim = 768 / 12 = 64`

Q has full 12-head dimension (12 × 64 = 768).
K and V only have 3-head dimension (3 × 64 = 192) — 4× smaller KV matrices.

---

## GQA expansion

Q has 12 heads, K/V have 3. Before computing attention scores, K and V are
expanded so shapes match:

```python
n_rep = n_heads // n_kv_heads  # = 4

# k: (B, 3, T, 64) → (B, 12, T, 64)
k = k.repeat_interleave(n_rep, dim=1)
v = v.repeat_interleave(n_rep, dim=1)
```

Query heads 0-3 all use KV head 0. Query heads 4-7 use KV head 1. Etc.
`repeat_interleave` repeats each row `n_rep` times (not a simple tile).

---

## Forward pass, step by step

```python
def forward(self, x, freqs_cis):
    B, T, _ = x.shape

    # 1. Project to Q, K, V
    q = self.wq(x).view(B, T, n_heads,    head_dim).transpose(1, 2)  # (B, 12, T, 64)
    k = self.wk(x).view(B, T, n_kv_heads, head_dim).transpose(1, 2)  # (B,  3, T, 64)
    v = self.wv(x).view(B, T, n_kv_heads, head_dim).transpose(1, 2)  # (B,  3, T, 64)

    # 2. Apply RoPE to Q and K (not V)
    q = apply_rope(q, freqs_cis)
    k = apply_rope(k, freqs_cis)

    # 3. Expand K, V from 3 heads to 12 (GQA)
    k = k.repeat_interleave(4, dim=1)   # (B, 12, T, 64)
    v = v.repeat_interleave(4, dim=1)

    # 4. Attention scores
    scale  = head_dim ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, 12, T, T)

    # 5. Causal mask
    scores = scores.masked_fill(~causal_mask[:T, :T], float('-inf'))

    # 6. Softmax → weighted average of V
    attn   = torch.softmax(scores, dim=-1)
    out    = torch.matmul(attn, v)         # (B, 12, T, 64)

    # 7. Merge heads + output project
    out = out.transpose(1, 2).reshape(B, T, n_embed)
    return self.wo(out)
```

---

## Causal mask

Registered as a buffer (not a parameter — it's not learned):

```python
mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
self.register_buffer("causal_mask", mask)
```

During forward, sliced to current T:
```python
scores.masked_fill(~self.causal_mask[:T, :T], float('-inf'))
```

`~` inverts the lower-triangular mask → the upper triangle is `True` → filled
with `-inf` → softmax output is 0 for all future positions.

---

## Verification

```
Causal check: output at position 0 with seq [A, B, C, D, E, F]
After changing token 5 to random noise:
  diff at position 0 = 0.00e+00  ✅
```

Token 0 is processed before tokens 1-5 exist in the causal attention view.
Changing any future token must not change past outputs.

---

**Next:** [05 · Transformer Block](05_transformer_block.md)
