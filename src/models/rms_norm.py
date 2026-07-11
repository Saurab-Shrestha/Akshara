"""
RMSNorm — Root Mean Square Layer Normalization
===============================================

WHY THIS EXISTS
---------------
Every transformer layer needs normalization to keep activations from growing
too large or shrinking too small during training. Without it, gradients either
explode or vanish and the model stops learning.

The original transformer used LayerNorm, which:
  1. Subtracts the mean  (centers the values)
  2. Divides by std      (scales the values)
  3. Applies γ and β     (learned scale + shift)

RMSNorm (2019, Zhang & Sennrich) removes step 1 and β entirely.
It turns out centering (subtracting mean) provides no benefit in practice —
the learned γ already handles it implicitly. Removing it makes each norm
~15% faster and slightly simpler.

Used in: LLaMA, Qwen, Mistral, Gemma, and our model.

FORMULA
-------
  RMS(x) = sqrt( mean(x²) + ε )     # ε prevents division by zero
  output  = γ * (x / RMS(x))

where γ (self.weight) is a learned per-dimension scale, initialized to 1.

SHAPE
-----
Input and output are identical shape: (batch, seq_len, dim)
The normalization is applied across the last dimension (dim) independently
for each token position.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Args:
            dim: the size of the last dimension of the input tensor (= n_embed)
            eps: small value added before sqrt to avoid division by zero
                 (1e-5: representable in bf16-scale statistics; 1e-8 is not)
        """
        super().__init__()
        self.eps = eps
        # γ — one learnable scale per dimension, initialized to 1 (identity at start)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            normalized tensor, same shape as x
        """
        # Statistics in fp32 even under bf16 autocast (LLaMA-style):
        # mean of 768 squares in bf16 loses precision and destabilizes training.
        xf = x.float()
        rms_inv = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (xf * rms_inv).type_as(x)


# ── self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # A batch of 2 sequences, each 4 tokens long, embedding size 8
    x = torch.randn(2, 4, 8)
    norm = RMSNorm(dim=8)
    out = norm(x)

    assert out.shape == x.shape, "output shape must match input shape"

    # After normalization each token's RMS should be close to 1
    # (γ is all-ones at init so it doesn't change this)
    rms = out.pow(2).mean(-1).sqrt()
    assert (rms - 1.0).abs().max().item() < 0.01, "RMS should be ~1.0 after norm"

    print("RMSNorm self-check passed.")
    print(f"  input shape : {x.shape}")
    print(f"  output shape: {out.shape}")
    print(f"  output RMS  : {rms[0].tolist()}")  # should all be ~1.0
