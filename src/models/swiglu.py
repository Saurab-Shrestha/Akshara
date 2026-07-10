"""
SwiGLU — Swish-Gated Linear Unit
==================================

WHY THIS EXISTS
---------------
Inside every transformer block, after the attention layer, there is an MLP
(also called the feed-forward network, FFN). Its job is to process each token
position independently — adding capacity for the model to "think" beyond
what attention captures.

The repo's MLP is simple:
    hidden = ReLU(x @ W1)   # expand
    output = hidden @ W2     # compress

ReLU has a hard problem: it sets all negative values to exactly zero.
Once a neuron outputs zero, it contributes nothing to the gradient during
backprop — it's "dead". A large fraction of neurons can go dead and stay
dead, wasting model capacity.

SwiGLU fixes this with a gating mechanism:

    gate   = x @ W_gate      # one projection — "how much to let through"
    hidden = x @ W_hidden    # another projection — "what to let through"
    output = (gate * sigmoid(gate)) * hidden   # gate controls hidden

The gate uses the Swish function (x * sigmoid(x)) which is smooth and
never fully zero — no dead neurons.

Compared to ReLU MLP:
  - More expressive (two separate learned projections)
  - No dead neurons
  - Slightly larger (3 weight matrices instead of 2) but reduced hidden
    dim compensates: repo uses 4×, SwiGLU uses 8/3× (≈2.67×) to keep
    the same parameter count
  - Used in: LLaMA, Qwen, PaLM, Mistral, and our model

FORMULA
-------
  gate   = W_gate(x)              # (batch, seq, hidden_dim)
  hidden = W_hidden(x)            # (batch, seq, hidden_dim)
  after_gate = F.silu(gate) * hidden   # element-wise gating
  output = W_out(after_gate)      # project back to n_embed

  where silu(x) = x * sigmoid(x)  [Sigmoid Linear Unit = Swish]

PARAMETER COUNT
---------------
  hidden_dim = int(n_embed * 8/3)   # ≈ 2.67 × n_embed
  W_gate:   n_embed   → hidden_dim
  W_hidden: n_embed   → hidden_dim
  W_out:    hidden_dim → n_embed

  vs repo MLP:
  W1: n_embed → 4*n_embed
  W2: 4*n_embed → n_embed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):

    def __init__(self, n_embed: int):
        """
        Args:
            n_embed: embedding dimension (input and output size)
        """
        super().__init__()
        # hidden_dim = 8/3 × n_embed, rounded to nearest multiple of 64
        # (multiple of 64 keeps matrix ops aligned for GPU efficiency)
        hidden_dim = int(n_embed * 8 / 3)
        hidden_dim = (hidden_dim + 63) // 64 * 64  # round up to multiple of 64

        # Gate projection — determines how much of hidden to pass through
        self.w_gate = nn.Linear(n_embed, hidden_dim, bias=False)

        # Hidden projection — the actual content being gated
        self.w_hidden = nn.Linear(n_embed, hidden_dim, bias=False)

        # Output projection — compress back to n_embed
        self.w_out = nn.Linear(hidden_dim, n_embed, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_embed)
        Returns:
            (batch, seq_len, n_embed)
        """
        # silu(gate) acts as a smooth, learned filter
        # multiplying with hidden selects which features pass through
        return self.w_out(F.silu(self.w_gate(x)) * self.w_hidden(x))


# ── self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    batch, seq_len, n_embed = 2, 6, 128

    mlp = SwiGLU(n_embed=n_embed)
    x   = torch.randn(batch, seq_len, n_embed)
    out = mlp(x)

    assert out.shape == x.shape, "output shape must match input shape"

    # Count parameters
    params = sum(p.numel() for p in mlp.parameters())
    hidden_dim = int(n_embed * 8 / 3)
    hidden_dim = (hidden_dim + 63) // 64 * 64
    expected = n_embed * hidden_dim * 3   # w_gate + w_hidden + w_out
    assert params == expected, f"param count mismatch: {params} vs {expected}"

    print("SwiGLU self-check passed.")
    print(f"  input  shape : {x.shape}")
    print(f"  output shape : {out.shape}")
    print(f"  hidden dim   : {hidden_dim}  (= n_embed × 8/3, rounded to ×64)")
    print(f"  params       : {params:,}")

    # Compare with repo's MLP param count for same n_embed
    repo_mlp_params = n_embed * (4 * n_embed) + (4 * n_embed) * n_embed
    print(f"  repo MLP params (4×): {repo_mlp_params:,}")
    print(f"  difference    : {params - repo_mlp_params:+,}")
