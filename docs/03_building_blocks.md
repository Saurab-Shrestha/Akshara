# 03 · Building Blocks

Three small components that every modern LLM uses instead of the original
transformer's equivalents: RMSNorm (replaces LayerNorm), RoPE (replaces
learned position embeddings), SwiGLU (replaces ReLU FFN).

---

## RMSNorm — faster normalization

**File:** [`src/models/rms_norm.py`](../src/models/rms_norm.py)

### What LayerNorm does (the old way)

LayerNorm normalizes by subtracting the mean and dividing by the standard
deviation, then applies a learned scale (γ) and shift (β):

```
mean  = x.mean(-1, keepdim=True)
std   = x.std(-1, keepdim=True)
x_norm = (x - mean) / (std + eps)
out   = γ * x_norm + β
```

### What RMSNorm does (the modern way)

Drop the mean subtraction (the re-centering step) and the β (shift) parameter.
Just normalize by the root mean square:

```python
rms_inv = x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
out = weight * (x * rms_inv)
```

**Why?** Empirically, the mean subtraction does little for training stability
but costs compute and an extra learned parameter. RMSNorm is 10-15% faster and
matches or beats LayerNorm on downstream performance. LLaMA, Qwen, and Surya v2
all use it.

**Self-check:** output RMS ≈ 1.0 (values are normalized to unit scale).

---

## RoPE — position without position embeddings

**File:** [`src/models/rope.py`](../src/models/rope.py)

### Why not learned position embeddings?

The classic approach: add a learned vector `pos_embed[position]` to each token.
Problems:
- Extra 512 × 768 = 393k learned parameters
- Doesn't generalize beyond training length
- Pollutes token semantics (position and content mixed in one vector)

### How RoPE works

RoPE encodes position by **rotating** the query and key vectors in 2D planes.
Each pair of dimensions gets rotated by an angle proportional to position `m`
and frequency `θ_i`:

```
angle = m / (10000 ^ (2i / d))

[x_{2i}, x_{2i+1}] → [x_{2i}·cos(angle) - x_{2i+1}·sin(angle),
                       x_{2i}·sin(angle) + x_{2i+1}·cos(angle)]
```

The key insight: when you compute `Q · K^T` after rotation, the dot product
between Q at position `m` and K at position `n` contains `cos(m - n)`. The
attention score naturally depends on **relative distance**, not absolute position.

```python
# precompute once
freqs_cis = precompute_freqs_cis(dim=64, max_seq_len=512)
# shape: (512, 32)  — complex numbers encoding the rotation angles

# apply to q and k (not v!)
q_rotated = apply_rope(q, freqs_cis)
k_rotated = apply_rope(k, freqs_cis)
```

**Self-check:** magnitude preserved (diff 4.77e-07 ≈ float32 precision), shape unchanged.

---

## SwiGLU — gated feed-forward network

**File:** [`src/models/swiglu.py`](../src/models/swiglu.py)

### The original FFN

The standard transformer FFN is two linear layers with a ReLU in between:

```
x → Linear(d_model, 4·d_model) → ReLU → Linear(4·d_model, d_model)
```

**Problem with ReLU:** any neuron that receives a negative input is clamped to
zero and produces zero gradient. "Dead neurons" — the model can't recover them.

### SwiGLU

Two branches: one computes a gate, the other computes values. The gate
(using SiLU = sigmoid-weighted linear unit) multiplies element-wise with values:

```python
# Three projections (not two):
gate_out    = F.silu(w_gate(x))   # the gate branch, smooth + non-zero gradient everywhere
hidden_out  = w_hidden(x)         # the value branch
return w_out(gate_out * hidden_out)
```

SiLU is `x · sigmoid(x)` — always differentiable, negative inputs → small
non-zero output (no dead neurons).

The hidden dimension is scaled to `8/3 × d_model` (rounded to multiple of 64)
to match the parameter count of the original 4× FFN while using three matrices
instead of two.

```python
hidden_dim = int(n_embed * 8/3)
hidden_dim = (hidden_dim + 63) // 64 * 64   # round up to multiple of 64
```

Why `8/3`? Three matrices at `8/3 × d` ≈ two matrices at `4 × d` in total FLOPs.
The rounding to 64 is for GPU memory alignment (CUDA tensor cores work in
multiples of 8, often 64 for efficiency).

LLaMA, Qwen, and Surya v2 all use SwiGLU.

---

## Putting them together

These three components appear in every block of the decoder:

```
TransformerBlock / GDNBlock
├── RMSNorm (before attention or GDN)
├── attention or GDN sub-layer
├── residual (+)
├── RMSNorm (before MLP)
├── SwiGLU MLP
└── residual (+)

The rotation from RoPE is applied inside attention to Q and K.
```

---

**Next:** [04 · Attention](04_attention.md)
