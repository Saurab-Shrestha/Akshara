"""
Connector — Vision-to-Language Bridge
======================================

WHY THIS EXISTS
---------------
The vision encoder outputs 384-dim patch tokens (ViT-S).
The decoder expects 768-dim tokens (our HybridDecoder).

These need to match. A simple linear projection (384 → 768) would work, but
a 2-layer MLP gives the model capacity to learn a non-linear translation
between the vision encoder's feature space and the decoder's language space.

This is the same connector design used in LLaVA and InternVL:
  vision_out (384) → Linear → GeLU → Linear → decoder_in (768)

WHY NOT JUST RESIZE THE ENCODER?
---------------------------------
You could make the vision encoder 768-dim to match the decoder. Then you need:
  - ViT-B/16 (768 dim, 86M params) instead of ViT-S (384 dim, 22M)
  - 4× more compute in the encoder

With ViT-S + connector:
  - Encoder: 22M params
  - Connector: 384×768×2 ≈ 590k params (tiny)
  - Total: ~22.6M params for the visual processing path

With ViT-B:
  - Encoder: 86M params
  - No connector needed
  - 4× more params for the same final output shape

The connector approach is more efficient on T4 VRAM.

WHY GELU NOT SWIGLU HERE?
--------------------------
SwiGLU uses 3 weight matrices (gate, hidden, out). For a 2-layer bridge, the
extra complexity isn't warranted — GeLU with 2 matrices suffices.
The connector is 0.6M params either way. No dead-neuron risk at this scale.
"""

import torch
import torch.nn as nn

from src.models.rms_norm import RMSNorm


class Connector(nn.Module):
    """
    2-layer MLP that bridges vision encoder output to decoder input dimension.

    Input:  (batch, n_patches, vision_dim)   e.g. (B, 196, 384)
    Output: (batch, n_patches, decoder_dim)  e.g. (B, 196, 768)

    The n_patches dimension is unchanged — every patch token gets projected.
    """

    def __init__(self, vision_dim: int, decoder_dim: int):
        """
        Args:
            vision_dim:  output dimension of the vision encoder (384 for ViT-S)
            decoder_dim: input dimension of the decoder (768 for our config)
        """
        super().__init__()

        # Hidden dim: 2× vision_dim is standard (same expansion ratio as FFNs)
        hidden_dim = vision_dim * 2

        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, decoder_dim, bias=True),
        )

        # Output norm: visual tokens must enter the decoder's residual stream
        # at the same scale as text embeddings (std≈0.02 per dim). Without it
        # the visual prefix arrives ~5× hotter and drowns out the text.
        # RMSNorm output has per-dim RMS ≈ 1; token embeds have per-dim std 0.02,
        # so init the learnable gain to 0.02 to match. It's learnable — the model
        # can turn the visual signal up as training progresses.
        self.out_norm = RMSNorm(decoder_dim)
        with torch.no_grad():
            self.out_norm.weight.fill_(0.02)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_tokens: (batch, n_patches, vision_dim)
        Returns:
            (batch, n_patches, decoder_dim) — ready to prepend to decoder input
        """
        return self.out_norm(self.net(vision_tokens))


# ── default config ────────────────────────────────────────────────────────────
DEFAULT_CONFIG = dict(
    vision_dim  = 384,   # ViT-S/16 output
    decoder_dim = 768,   # HybridDecoder input
)


# ── self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    connector = Connector(**DEFAULT_CONFIG)

    total = sum(p.numel() for p in connector.parameters())
    print(f"Connector self-check")
    print(f"  params       : {total:,}  ({total/1e6:.2f}M)")

    # Simulate vision encoder output
    B, n_patches, vision_dim = 2, 196, DEFAULT_CONFIG["vision_dim"]
    vision_tokens = torch.randn(B, n_patches, vision_dim)

    out = connector(vision_tokens)

    expected_shape = (B, n_patches, DEFAULT_CONFIG["decoder_dim"])
    assert out.shape == expected_shape, f"shape mismatch: {out.shape}"

    print(f"  input shape  : {tuple(vision_tokens.shape)}")
    print(f"  output shape : {tuple(out.shape)}  ✅")
    print(f"  dim bridge   : {vision_dim} → {DEFAULT_CONFIG['decoder_dim']}")
