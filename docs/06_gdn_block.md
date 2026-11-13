# 06 · GDN Block

The Gated DeltaNet block — the key architectural innovation. Replaces 75% of
attention layers with O(T) recurrence.

**Files:** [`src/models/gdn_block.py`](../src/models/gdn_block.py)

Read the [Gated DeltaNet foundation doc](foundations/gdn.md) first if you
haven't — it covers the theory. This doc focuses on the implementation.

---

## Two classes

```
GatedDeltaNetLayer   ← the recurrence (replaces self-attention)
GDNBlock             ← wraps the layer with norm + SwiGLU + residuals
                        (drop-in for TransformerBlock)
```

---

## GatedDeltaNetLayer — the recurrence

### Projections

```python
self.wq = nn.Linear(d_model, d_model, bias=False)   # queries (for reading)
self.wk = nn.Linear(d_model, d_model, bias=False)   # keys (for addressing)
self.wv = nn.Linear(d_model, d_model, bias=False)   # values (to write)
self.w_beta  = nn.Linear(d_model, n_heads, bias=False)  # update gate (scalar/head)
self.w_alpha = nn.Linear(d_model, n_heads, bias=False)  # forget gate (scalar/head)
self.out_norm = RMSNorm(d_model)
self.wo = nn.Linear(d_model, d_model, bias=False)   # output projection
```

Same Q/K/V structure as attention. Two extra projections for the gates (one
scalar per head, not per dimension — very cheap).

### Forward pass

```python
def forward(self, x):
    B, T, _ = x.shape
    H, D = self.n_heads, self.d_head

    q = self.wq(x).view(B, T, H, D)
    k = self.wk(x).view(B, T, H, D)
    v = self.wv(x).view(B, T, H, D)

    k = F.normalize(k, dim=-1)   # L2-normalize keys (unit vectors = stable addresses)

    beta  = torch.sigmoid(self.w_beta(x)).unsqueeze(-1)   # (B, T, H, 1)
    alpha = torch.sigmoid(self.w_alpha(x)).unsqueeze(-1)  # (B, T, H, 1)

    S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)

    outputs = []
    for t in range(T):
        kt = k[:, t]    # (B, H, D)
        vt = v[:, t]
        qt = q[:, t]
        bt = beta[:, t]   # (B, H, 1)
        at = alpha[:, t]

        # Read: what does S think k_t means?
        S_kt = torch.matmul(S, kt.unsqueeze(-1)).squeeze(-1)   # (B, H, D)

        # Delta: erase current, write new
        delta = torch.matmul(
            (vt - S_kt).unsqueeze(-1),   # (B, H, D, 1)
            kt.unsqueeze(-2)              # (B, H, 1, D)
        )                                 # → (B, H, D, D) outer product

        # Gated update
        S = at.unsqueeze(-1) * S + bt.unsqueeze(-1) * delta

        # Query the updated state
        ot = torch.matmul(S, qt.unsqueeze(-1)).squeeze(-1)
        outputs.append(ot)

    out = torch.stack(outputs, dim=1)   # (B, T, H, D)
    out = out.contiguous().view(B, T, -1)  # merge heads

    return self.wo(self.out_norm(out))
```

### Why L2-normalize k?

K vectors are used as "addresses" in the state matrix S. If K vectors have
large magnitude, the outer product `v ⊗ k` will dominate S and cause the
matrix to grow unboundedly. Unit-norm keys guarantee the outer product has
bounded magnitude, keeping S numerically stable across many timesteps.

Q and V are not normalized — Q controls what we *read*, magnitude matters.
V controls what value we *write*, magnitude should be preserved.

---

## GDNBlock — the full layer

```python
class GDNBlock(nn.Module):
    def __init__(self, n_embed, n_heads):
        self.norm1 = RMSNorm(n_embed)
        self.gdn   = GatedDeltaNetLayer(n_embed, n_heads)
        self.norm2 = RMSNorm(n_embed)
        self.mlp   = SwiGLU(n_embed)

    def forward(self, x, freqs_cis=None):
        x = x + self.gdn(self.norm1(x))   # GDN sub-layer
        x = x + self.mlp(self.norm2(x))   # MLP sub-layer
        return x
```

`freqs_cis` is accepted but ignored — GDN doesn't use RoPE. Position is
implicit in the recurrence order (t=0 is processed before t=1). The parameter
exists so `GDNBlock` and `TransformerBlock` have the same interface, letting
`HybridDecoder` call both with the same `layer(x, freqs_cis)` pattern.

---

## Causality guarantee

The sequential loop `for t in range(T)` processes tokens in order. State `S`
at time `t` is computed from tokens `0..t-1` only. Token `t=0`'s output uses
only the initialized zero state — it cannot see any future token.

```
Self-check result:
  causal check : t=0 unaffected by t=5  ✅  (diff=0.00e+00)
```

---

## Parameter comparison

At n_embed=256, n_heads=8:

```
GDNBlock params      : 526,848
TransformerBlock params: 393,216
GDN is 1.34x the size of standard block
```

GDN is slightly larger because of the extra `w_beta`, `w_alpha`, `out_norm`
projections. The compute is lower (O(T) vs O(T²)), but per-step VRAM during
the sequential loop is higher (holding S). Triton kernel would hide this cost.

---

## CPU vs GPU implementation

| Implementation | Correctness | Speed (T=512) | Platforms |
|---|---|---|---|
| This PyTorch loop | ✅ | ~100ms | Any |
| fla `chunk_gated_delta_rule` kernel | ✅ | ~2ms | CUDA only |

Both run on **the same weights** (v2, `gdn_impl: "native"`). The layer computes
q, k, v, β and the forget gate g = log α with its own projections, then hands
them to FLA's chunked kernel on CUDA (`use_fla: true`) or to
`recurrent_gated_delta_rule` elsewhere. The PyTorch loop follows FLA's naive
reference exactly — decay first, then the delta rule, q and k L2-normalised,
q scaled by 1/√d — and `tests/test_inference.py` checks it against that
reference. So a model trained on a GPU loads and decodes on a Mac.

For decoding, the layer also returns its memory S (B, H, K, V): generation
carries S forward one token at a time instead of re-reading the prefix.

v1 checkpoints (`gdn_impl: "fla_layer"`) used FLA's whole `GatedDeltaNet`
module, which has different weights; they still load, but only with FLA on CUDA.

---

**Next:** [07 · Hybrid Decoder](07_hybrid_decoder.md)
