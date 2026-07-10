# Attention — Letting Tokens See Each Other

The attention mechanism is the core operation of a transformer. It lets each
token gather information from other tokens in the sequence.

---

## The problem it solves

Imagine reading the Nepali word `राम्रो` and needing to know whether it refers
to a person or an adjective. That depends on the surrounding words. A token
processed in isolation can't make that judgment.

Attention gives each token a "query" — what am I looking for? — and every other
token a "key" — what do I have? — and a "value" — what should I send back if
you select me?

---

## The formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

Step by step:
1. **QK^T** — dot product of query with all keys. High score = relevant token.
2. **/ √d_k** — scale down so softmax doesn't saturate (gradients vanish if logits are too large).
3. **softmax** — convert scores to probabilities that sum to 1.
4. **· V** — weighted average of values. Tokens with high attention scores contribute more.

---

## Multi-head attention

One attention head looks for one kind of relationship (e.g. subject-verb). We
want multiple relationship types simultaneously. The solution: run H independent
attention heads in parallel, each with smaller dimension d_head = d_model / H.

```
d_model = 768    (total embedding dimension)
n_heads = 12     (parallel attention heads)
d_head  = 64     (per-head dimension)
```

Each head has its own Q, K, V projection matrices. Results are concatenated and
projected back to d_model.

---

## Causal masking

During OCR, the decoder generates text left-to-right. Token at position 5 must
not see token at position 6 (that's cheating — it hasn't been generated yet).

We enforce this by masking the attention matrix with a lower-triangular matrix:

```
Position:  0  1  2  3  4
Token 0: [ ✓  ✗  ✗  ✗  ✗ ]   ← can only see itself
Token 1: [ ✓  ✓  ✗  ✗  ✗ ]
Token 2: [ ✓  ✓  ✓  ✗  ✗ ]
Token 3: [ ✓  ✓  ✓  ✓  ✗ ]
Token 4: [ ✓  ✓  ✓  ✓  ✓ ]   ← can see all past tokens
```

Positions marked ✗ are set to `-inf` before softmax → they become 0 in the
attention weights → zero contribution to the output.

In code:
```python
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
scores = scores.masked_fill(~mask, float('-inf'))
```

---

## Grouped Query Attention (GQA)

Standard multi-head attention: 12 query heads, 12 key heads, 12 value heads.
That means 36 weight matrices. KV is the expensive part at inference — you
cache it, and it grows with sequence length.

GQA: share K and V across groups of queries.

```
n_heads    = 12   (query heads)
n_kv_heads = 3    (KV heads — 4 query heads share each KV head)
```

```
Q heads:   [0  1  2  3]  [4  5  6  7]  [8  9  10 11]
                │               │               │
KV head:      [0]            [1]            [2]
```

This cuts KV parameter count by 4×. At inference on long documents, the KV
cache (the main memory bottleneck) shrinks by 4×. Quality barely changes — the
queries still have full expressiveness.

---

## RoPE: position inside attention

How does attention know token 5 is further from token 0 than token 1 is?

Classic solution: add a position embedding to the input. Problem: it contaminates
the token's semantic meaning and doesn't generalize beyond the training length.

**RoPE (Rotary Position Embedding)** embeds position differently: it rotates
the Q and K vectors by an angle proportional to position. The dot product
between a rotated Q at position `m` and a rotated K at position `n` naturally
contains a term `cos(m - n)` — the attention score depends on the *relative*
distance, not absolute positions.

Key properties:
- Position information only affects Q and K — V keeps the clean semantic signal
- The rotation magnitude scales by dimension (slow-varying base frequencies)
- Generalizes well to lengths beyond training

See [Building Blocks doc](../03_building_blocks.md) for the implementation.

---

## What we built

`src/models/attention.py` — `MultiHeadAttention(n_embed, n_heads, n_kv_heads, max_seq_len)`:
- Projects input to Q, K, V with bias=False (standard for modern LLMs)
- Applies RoPE to Q and K only
- Expands K, V with `repeat_interleave` so GQA shapes match for matmul
- Applies causal mask
- Verified: position 0 is completely unaffected by position 5 (diff = 0.00e+00)

---

**Next:** [Transformer Block](../foundations/transformer.md)
