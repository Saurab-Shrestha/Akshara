# Gated DeltaNet — Linear Attention That Actually Works

Standard attention is O(T²) in sequence length — as the document gets longer,
compute grows quadratically. A 2-page Nepali document with 4,096 tokens costs
4× more than a 1-page document with 2,048 tokens. Gated DeltaNet replaces
this with O(T) recurrence using a fixed-size memory matrix.

---

## Three ideas combined

### 1. Linear attention (the base)

Instead of computing all pairwise attention scores, maintain a running state
matrix `S` that accumulates key-value associations:

```
Standard: output_t = softmax(q_t · K^T / √d) · V    — looks at all T past tokens
Linear:   S_t = S_{t-1} + v_t ⊗ k_t              — accumulates into fixed matrix
          output_t = S_t · q_t                    — reads from matrix
```

`v_t ⊗ k_t` is an outer product: a d×d matrix saying "associate value v at key k."
The state S is a d×d matrix regardless of T. Once full, it overwrites old info.

**Problem:** S grows without bound — new information piles on top of old. The
model can't selectively update or forget.

---

### 2. Delta rule (DeltaNet's fix)

Before writing `v_t` at address `k_t`, first **erase** what S currently stores
at `k_t`. Then write the new value. This is targeted write, not accumulate:

```
What S currently thinks k_t means:  S_kt = S · k_t
Error (what needs updating):        Δ = v_t - S_kt
Delta update:                       S = S + β · outer(Δ, k_t)

Which simplifies to:
S_t = S_{t-1} + β_t · outer(v_t - S_{t-1}·k_t, k_t)
```

`β_t` (beta) is learned, per-token: how aggressively to apply this update?
β → 0: keep old state. β → 1: full replacement.

The name "DeltaNet" comes from the delta learning rule — the same principle
used in the Widrow-Hoff perceptron rule (1960s), now applied to a matrix memory.

---

### 3. Gating (forget mechanism)

Add an alpha gate `α_t` that can scale down the old state before applying the
delta update:

```
S_t = α_t · S_{t-1} + β_t · outer(v_t - α_t · S_{t-1}·k_t, k_t)
```

`α_t → 1`: keep everything (default for middle of a paragraph)
`α_t → 0`: wipe the slate clean (useful at document boundaries)

This is the "Gated" part of Gated DeltaNet. It's similar to how GRU and LSTM
gates control information flow — but operating on a matrix rather than a vector.

**Why this matters for OCR:** A multi-page document contains multiple distinct
text regions. When the model moves to a new region (new line, new paragraph),
the alpha gate can flush the accumulated context and start fresh. Without this,
old character patterns bleed into new lines.

---

## State matrix visualization

```
S: shape (n_heads, d_head, d_head)

For our config: 12 heads, d_head = 64

One head's state matrix = 64×64 = 4,096 values

Think of each row as a "memory slot" identified by the key direction.
Writing to slot k updates the row aligned with k.
Reading from slot q retrieves a weighted sum of rows aligned with q.
```

Compared to standard attention's KV cache:
- Standard KV cache at token T: (T, n_heads, d_head) × 2 → **grows with T**
- GDN state: (n_heads, d_head, d_head) → **fixed size always**

At T=4,096 with 12 heads and d_head=64:
- Standard KV: 4,096 × 12 × 64 × 2 = **6.3M floats**
- GDN state: 12 × 64 × 64 = **49,152 floats** — 128× smaller

---

## Why 3:1 hybrid (not pure GDN)?

GDN's fixed-size state compresses information. For most OCR tasks (reading
text character by character, one line at a time), the compression is fine.
But occasionally, exact verbatim recall is needed — spelling out rare character
sequences, copying a digit that appeared 50 tokens ago.

One exact-attention layer every 4 layers acts as a **recall checkpoint**:

```
Layer 0-2: GDN  → fast, good enough for most content
Layer 3:   Attn → exact recall, fixes any GDN drift
Layer 4-6: GDN  → fast again, starts from corrected state
Layer 7:   Attn → exact recall again
...
```

The attention layer has full O(T²) visibility of all past tokens. It can
copy any token verbatim if needed. 25% of layers being exact is enough.

This pattern is empirically validated: Surya v2 (OCR), Qwen3-Next (NLP),
Kimi Linear (long-context) all use the 3:1 ratio at 650M–7B scale.

---

## Implementation: CPU vs GPU

Our implementation (`src/models/gdn_block.py`) loops over T in Python:

```python
for t in range(T):
    S_kt = S @ kt    # read
    delta = outer(vt - S_kt, kt)
    S = alpha * S + beta * delta  # write
    ot = S @ qt      # query
```

This is **correct** but **slow** — Python loop overhead is high for long sequences.

On Kaggle T4 (CUDA): swap `GatedDeltaNetLayer` with `fla.layers.GatedDeltaNet`
from [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention).
Their Triton kernel fuses all T steps into one GPU pass — same math, 10–50× faster.

The swap is one import change. Our pure-PyTorch version is for understanding
and for Mac/CPU development (Triton requires CUDA).

---

**Back to build stages:** [06 · GDN Block](../06_gdn_block.md)
