# 07 · Hybrid Decoder

> **v2 note:** the numbers in this chapter are the original 12-layer, 248k-vocab
> walkthrough. The v2 decoder is **32 layers × 1024 wide (24 GDN + 8 attention),
> 48k own vocab, QK-norm → ~443M params (393M body, 50M embedding)**. The 3:1
> mechanics described here are unchanged. See [ROADMAP_V2.md](ROADMAP_V2.md).


The full text model: 3:1 GDN + Attention hybrid with weight tying and vision prefix support.

**Files:** [`src/models/hybrid_decoder.py`](../src/models/hybrid_decoder.py),
[`src/models/decoder.py`](../src/models/decoder.py) (pure-attention reference)

---

## Why two decoders?

`decoder.py` — standard transformer decoder. All 12 layers are `TransformerBlock`.
Used as a reference and for ablation (comparing GDN vs pure attention).

`hybrid_decoder.py` — the real architecture. 9 GDN + 3 Attention = 3:1 ratio.
This is what we train.

---

## Layer layout

```python
def _build_layers(n_layers, n_embed, n_heads, n_kv_heads, max_seq_len, attn_every=4):
    layers = []
    for i in range(n_layers):
        if (i + 1) % attn_every == 0:
            layers.append(TransformerBlock(n_embed, n_heads, n_kv_heads, max_seq_len))
        else:
            layers.append(GDNBlock(n_embed, n_heads))
    return nn.ModuleList(layers)
```

`(i+1) % 4 == 0` hits at i = 3, 7, 11 — exactly every 4th layer.

```
Layer  0: GDNBlock
Layer  1: GDNBlock
Layer  2: GDNBlock
Layer  3: TransformerBlock  ← recall checkpoint
Layer  4: GDNBlock
Layer  5: GDNBlock
Layer  6: GDNBlock
Layer  7: TransformerBlock  ← recall checkpoint
Layer  8: GDNBlock
Layer  9: GDNBlock
Layer 10: GDNBlock
Layer 11: TransformerBlock  ← recall checkpoint
```

---

## Model structure

```python
class HybridDecoder(nn.Module):

    def __init__(self, vocab_size, n_embed, n_heads, n_kv_heads, n_layers, max_seq_len, attn_every=4):
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.layers      = _build_layers(...)
        self.norm        = RMSNorm(n_embed)
        self.lm_head     = nn.Linear(n_embed, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embed.weight

        # RoPE factors — precomputed, used by TransformerBlocks
        head_dim = n_embed // n_heads
        freqs_cis = precompute_freqs_cis(dim=head_dim, max_seq_len=max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)
```

---

## Forward pass

```python
def forward(self, token_ids, targets=None, vision_prefix=None):
    B, T = token_ids.shape
    x = self.token_embed(token_ids)      # (B, T, 768)

    if vision_prefix is not None:
        x = torch.cat([vision_prefix, x], dim=1)  # prepend visual tokens

    freqs_cis = self.freqs_cis[:x.shape[1]]

    for layer in self.layers:
        if self.gradient_checkpointing and self.training:
            x = ckpt.checkpoint(layer, x, freqs_cis, use_reentrant=False)
        else:
            x = layer(x, freqs_cis)   # GDNBlock ignores freqs_cis; TransformerBlock uses it

    x = self.norm(x)
    logits = self.lm_head(x)            # (B, total_len, vocab_size)

    if targets is not None:
        if vision_prefix is not None:
            n_visual = vision_prefix.shape[1]
            logits   = logits[:, n_visual:, :]   # don't supervise visual tokens

        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1).long())

    return logits, loss
```

Key decision: when `vision_prefix` is prepended, we compute loss **only on
the text portion**. Visual tokens are not text — asking the model to predict
them as vocabulary tokens would be nonsensical.

---

## Generation

The naive loop re-runs the whole prefix for every new token:

```python
for _ in range(max_new_tokens):
    logits, _ = self(token_ids)                    # O(T) work per token → O(T²) total
    next_id = sample(logits[:, -1], temperature)
    token_ids = torch.cat([token_ids, next_id], dim=1)
```

The real `generate()` keeps a **cache** instead:

```python
cache = self.new_cache()                           # per layer: KV cache or GDN memory
logits = self.forward_cached(prompt, cache)[:, -1] # prefill: run the prompt once
for _ in range(max_new_tokens):
    next_id = sample(logits, temperature, top_k, top_p)
    logits = self.forward_cached(next_id, cache)[:, -1]   # one token, one step
```

- **Attention layers** store every past token's (already rotated) K and V; a
  new token attends to them without recomputing them.
- **GDN layers** store only their fixed-size memory S (H × d × d), whatever the
  length — that's the hybrid's inference advantage.
- `rollback(cache, n)` drops tokens again (slice K/V, restore S). Speculative
  decoding (`src/speculative.py`) uses it: a small draft model proposes k
  tokens, this model checks them in one `forward_cached` call, and rejected
  ones are rolled back.

`tests/test_inference.py` checks that cached logits equal a full forward pass
and that greedy speculative output equals plain greedy output.

**Temperature:**
- `temperature=1.0` → sample normally from the distribution
- `temperature=0.1` → confident, near-greedy (good for OCR — text should be verbatim)
- `temperature=2.0` → chaotic, diverse (bad for OCR)

For OCR inference, use `temperature=0.1` or greedy decoding (`argmax`).

---

## Default configuration

```python
DEFAULT_CONFIG = dict(
    vocab_size  = 248_044,   # Qwen3.5-0.8B tokenizer
    n_embed     = 768,
    n_heads     = 12,
    n_kv_heads  = 3,
    n_layers    = 12,
    max_seq_len = 512,
    attn_every  = 4,         # 3:1 GDN:Attention ratio
)
```

---

## Parameter count

| Component | Params |
|---|---|
| token_embed (= lm_head, tied) | 248,044 × 768 = **190.5M** |
| 9 × GDNBlock | 9 × ~807k = 7.3M |
| 3 × TransformerBlock | 3 × ~705k = 2.1M |
| 2 × RMSNorm | ~1k |
| **Total unique** | **~273M** |

The embedding table dominates. The 12 transformer-style layers cost only
~9.4M combined — the bulk of the model's "intelligence" lives in the embedding
space, not the layer count.

---

## Inspection helper

```python
print(model.layer_summary())
# Layer  0: GDNBlock
# Layer  1: GDNBlock
# Layer  2: GDNBlock
# Layer  3: TransformerBlock
# ...
```

---

## Self-check results

```
Layer layout: 9 GDN + 3 Attention = 12 layers  ✅
Ratio: 9:3 = 3:1  ✅
logits shape: (2, 16, 1000)
loss: 6.9082  (expect ~log(1000) = 6.91)  ✅
generate: (1, 9) shape  ✅
Real config unique params: 273.0M
```

---

## What comes next

The HybridDecoder is a text-only model. To make it do OCR, we add:

1. **Vision Encoder** (Stage 08) — ViT-S/16, encodes a document image into patch tokens
2. **Connector** (Stage 09) — 2-layer MLP, maps patch-dim → decoder-dim
3. **Full VLM** (Stage 10) — wires all three together
4. **Synthetic Data** (Stage 11) — generates Nepali text images for training
5. **Training** (Stage 12) — pretraining on text corpus, then OCR fine-tuning

The decoder's `vision_prefix` parameter is already wired to accept visual tokens —
stages 08-10 just need to provide them.

---

**Next:** Stage 08 · Vision Encoder *(coming)*
