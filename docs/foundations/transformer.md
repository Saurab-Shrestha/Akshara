# Transformer — How Layers Stack

A transformer decoder is a sequence of identical blocks. Each block does two
things: communicate (attention) and compute (MLP). Stacking 12 of them gives
the model enough representational power to learn Nepali OCR.

---

## One block, two sub-layers

```
Input x  (B, T, 768)
    │
    ├─ RMSNorm → Attention ──────────(+)→ x
    │                                 ↑
    │                          residual adds back original x
    │
    ├─ RMSNorm → SwiGLU MLP ────────(+)→ x
    │
Output x  (B, T, 768)   ← same shape as input
```

**Sub-layer 1 — Attention:** Each token gathers information from other tokens.
Tokens can "talk" to each other here.

**Sub-layer 2 — MLP:** Each token is processed independently (no cross-token
communication). This is where facts are stored and refined.

---

## Residual connections: the gradient highway

Every sub-layer is wrapped as `x = x + sublayer(x)`.

Why? Without residuals, gradients (the signals that update weights during
training) must pass through every layer to reach the earlier ones. 12 layers
deep, the gradient shrinks to near zero — the model can't learn.

With residuals, the gradient has a direct path: it flows straight back along
the `+x` branch without going through the heavy sub-layer computation.

```
Forward:  x_12 = x_11 + attn_11(norm(x_11))
                             ↑
Backward: grad flows straight down this identity branch to x_11
```

This is what makes training 12+ layer networks stable.

---

## Pre-norm: normalize before, not after

Original transformer (2017): `x = norm(x + sublayer(x))` — norm after residual.
Modern transformers (LLaMA, Qwen, Surya v2): `x = x + sublayer(norm(x))` — norm before.

Why pre-norm? At the start of training, weights are random and sublayer outputs
are chaotic. Pre-norm applies normalization to the clean `x` — the sublayer
receives a stabilized input. Post-norm normalizes `x + chaos`, which is harder.

Pre-norm trains significantly faster and more stably at large scale.

---

## The training loop

```python
optimizer.zero_grad()    # clear old gradients
logits, loss = model(ids, targets)   # forward pass
loss.backward()          # compute gradients (chain rule)
optimizer.step()         # update weights
```

**zero_grad** — gradients accumulate by default in PyTorch. You must clear them
each step or they'll add up from previous steps.

**forward** — the model makes predictions. Loss = how wrong were we? (cross-entropy
over the vocabulary — measures bits needed to encode the true token).

**backward** — PyTorch traces the computation graph backwards, computing
∂loss/∂weight for every parameter. This is automatic differentiation.

**step** — the optimizer (AdamW) uses the gradients to nudge weights in the
direction that reduces loss.

---

## Gradient checkpointing

Training stores every intermediate activation (each layer's output) to use
during backprop. 12 layers × (batch × seq_len × 768) floats ≈ enormous.

Gradient checkpointing: **don't save** the activations. During backprop, rerun
the forward pass for each layer to recompute what's needed.

Trade: ~20% more compute, ~40% less VRAM. Essential for training on Kaggle T4.

```python
model.set_gradient_checkpointing(True)  # call before training
```

---

## Weight tying

The model has two giant matrices touching vocabulary:
1. `token_embed`: maps token IDs → dense vectors (190M params)
2. `lm_head`: maps dense vectors → next-token logits (would be 190M params)

They do opposite operations. It turns out tying them (sharing the same matrix)
works as well or better than separate matrices, and halves the vocab parameter
cost:

```python
self.lm_head.weight = self.token_embed.weight
```

GPT-2, LLaMA, and Qwen all use this. Our model goes from ~388M to ~198M unique
params with this single line.

---

**Next:** [Gated DeltaNet](gdn.md) — the recurrent layer that replaces 75% of attention
