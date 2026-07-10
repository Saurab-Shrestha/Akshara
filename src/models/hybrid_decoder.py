"""
Hybrid Decoder — Gated DeltaNet 3:1
=====================================

WHY THIS EXISTS
---------------
This replaces decoder.py with the actual Surya v2 architecture:
a 3:1 hybrid of GDN layers and standard attention layers.

The problem each layer type solves:

  GDNBlock (3 out of 4 layers):
    - O(T) recurrence — fast on long documents
    - Fixed-size memory — bounded RAM at inference
    - Learns to forget stale content (α gate)
    - Learns to update specific memory slots (β gate)
    - Weakness: compressed state can miss rare exact patterns

  TransformerBlock (1 out of 4 layers):
    - O(T²) exact attention — sees every past token precisely
    - Acts as a "checkpoint" — corrects any drift from GDN compression
    - Ensures verbatim character recall (critical for OCR)
    - Expensive but only 25% of layers

THE 3:1 PATTERN (12 layers total)
-----------------------------------
  Layer  0: GDNBlock          ┐
  Layer  1: GDNBlock          │ group 0
  Layer  2: GDNBlock          │
  Layer  3: TransformerBlock  ┘ ← exact recall checkpoint

  Layer  4: GDNBlock          ┐
  Layer  5: GDNBlock          │ group 1
  Layer  6: GDNBlock          │
  Layer  7: TransformerBlock  ┘ ← exact recall checkpoint

  Layer  8: GDNBlock          ┐
  Layer  9: GDNBlock          │ group 2
  Layer 10: GDNBlock          │
  Layer 11: TransformerBlock  ┘ ← exact recall checkpoint

WHY THIS RATIO
--------------
The 3:1 ratio is empirically validated at 650M–7B scale (Surya v2, Qwen3.5,
Kimi Linear). At our smaller scale (~265M), the ratio might need adjustment.
If you find the model struggles with exact recall during training, try 2:1
(2 GDN + 1 attention, repeat) — more frequent checkpoints.

KAGGLE NOTE
-----------
On Kaggle T4 (CUDA), for actual training speed, swap GatedDeltaNetLayer
with fla-org's implementation:

    from fla.layers import GatedDeltaNet
    # Use in place of our GatedDeltaNetLayer inside GDNBlock

Our PyTorch version is correct but runs the T-loop in Python (slow for long
sequences). FLA's Triton kernel does the same math in one GPU pass.

PARAMETER COUNT
---------------
  12 layers: 9 GDNBlock (807k each) + 3 TransformerBlock (705k each) ≈ 9.4M
  Token embedding (248k × 768)                                        ≈ 190M
  Total unique                                                         ≈ 200M
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from src.models.gdn_block import GDNBlock
from src.models.transformer_block import TransformerBlock
from src.models.rms_norm import RMSNorm
from src.models.rope import precompute_freqs_cis


def _build_layers(
    n_layers:    int,
    n_embed:     int,
    n_heads:     int,
    n_kv_heads:  int,
    max_seq_len: int,
    attn_every:  int = 4,
) -> nn.ModuleList:
    """
    Build the 3:1 hybrid layer stack.

    Args:
        n_layers:   total number of blocks
        n_embed:    embedding dimension
        n_heads:    query heads (used by both GDN and attention)
        n_kv_heads: KV heads (GQA, used only by attention blocks)
        max_seq_len:causal mask size (attention blocks only)
        attn_every: place a standard attention block every N layers (default 4 = 3:1 ratio)

    Returns:
        ModuleList of alternating GDNBlock and TransformerBlock
    """
    layers = []
    for i in range(n_layers):
        if (i + 1) % attn_every == 0:
            # Every 4th layer: exact attention checkpoint
            layers.append(TransformerBlock(n_embed, n_heads, n_kv_heads, max_seq_len))
        else:
            # All other layers: fast GDN recurrence
            layers.append(GDNBlock(n_embed, n_heads))
    return nn.ModuleList(layers)


class HybridDecoder(nn.Module):

    def __init__(
        self,
        vocab_size:  int,
        n_embed:     int,
        n_heads:     int,
        n_kv_heads:  int,
        n_layers:    int,
        max_seq_len: int,
        attn_every:  int = 4,
    ):
        """
        Args:
            vocab_size:  tokenizer vocabulary size
            n_embed:     hidden/embedding dimension
            n_heads:     attention/GDN query heads
            n_kv_heads:  KV heads for GQA (attention blocks only)
            n_layers:    total layers
            max_seq_len: maximum sequence length
            attn_every:  full attention every N layers (4 = 3:1 ratio)
        """
        super().__init__()
        self.n_embed     = n_embed
        self.max_seq_len = max_seq_len
        self.gradient_checkpointing = False

        self.token_embed = nn.Embedding(vocab_size, n_embed)

        self.layers = _build_layers(
            n_layers, n_embed, n_heads, n_kv_heads, max_seq_len, attn_every
        )

        self.norm    = RMSNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        # Weight tying: share embedding and lm_head weights
        self.lm_head.weight = self.token_embed.weight

        # Precompute RoPE tables (real-valued cos/sin — no complex tensors so
        # DataParallel buffer replication works correctly).
        # ×3 covers visual prefix (≤784 patches) + full text sequence.
        head_dim       = n_embed // n_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=head_dim, max_seq_len=max_seq_len * 3)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_gradient_checkpointing(self, enabled: bool):
        self.gradient_checkpointing = enabled

    def forward(
        self,
        token_ids:     torch.Tensor,
        targets:       torch.Tensor = None,
        vision_prefix: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            token_ids:     (batch, seq_len) integer token IDs
            targets:       (batch, seq_len) next-token targets (optional)
            vision_prefix: (batch, n_patches, n_embed) visual tokens (optional)
        Returns:
            logits: (batch, total_len, vocab_size)
            loss:   scalar or None
        """
        B, T = token_ids.shape
        x = self.token_embed(token_ids)

        if vision_prefix is not None:
            x = torch.cat([vision_prefix, x], dim=1)

        total_len = x.shape[1]
        cos = self.freqs_cos[:total_len]
        sin = self.freqs_sin[:total_len]

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                def _make_fwd(l, c, s):
                    def _inner(inp):
                        return l(inp, c, s)
                    return _inner
                x = ckpt.checkpoint(_make_fwd(layer, cos, sin), x, use_reentrant=False)
            else:
                x = layer(x, cos, sin)

        x      = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if vision_prefix is not None:
                n_visual        = vision_prefix.shape[1]
                logits_for_loss = logits[:, n_visual:, :]
            else:
                logits_for_loss = logits

            B, L, V = logits_for_loss.shape
            loss = torch.nn.functional.cross_entropy(
                logits_for_loss.reshape(B * L, V),
                targets.reshape(B * L).long(),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids:      torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 1.0,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ids_cond    = token_ids[:, -self.max_seq_len:]
            logits, _   = self(ids_cond)
            next_logits = logits[:, -1, :] / temperature
            probs       = torch.softmax(next_logits, dim=-1)
            next_id     = torch.multinomial(probs, num_samples=1)
            token_ids   = torch.cat([token_ids, next_id], dim=1)
        return token_ids

    def layer_summary(self) -> str:
        """Print which layer is GDN vs Attention — useful for debugging."""
        lines = []
        for i, layer in enumerate(self.layers):
            kind = "GDNBlock      " if isinstance(layer, GDNBlock) else "TransformerBlock"
            lines.append(f"  Layer {i:2d}: {kind}")
        return "\n".join(lines)


# ── default config ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = dict(
    vocab_size  = 248_044,
    n_embed     = 768,
    n_heads     = 12,
    n_kv_heads  = 3,
    n_layers    = 12,
    max_seq_len = 512,
    attn_every  = 4,   # 3:1 ratio
)


# ── self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Tiny config for fast testing
    cfg = dict(vocab_size=1000, n_embed=128, n_heads=4,
               n_kv_heads=2, n_layers=8, max_seq_len=32, attn_every=4)

    model = HybridDecoder(**cfg)

    print("HybridDecoder layer layout:")
    print(model.layer_summary())

    # Count each type
    gdn_count  = sum(1 for l in model.layers if isinstance(l, GDNBlock))
    attn_count = sum(1 for l in model.layers if isinstance(l, TransformerBlock))
    print(f"\n  GDN blocks  : {gdn_count}")
    print(f"  Attn blocks : {attn_count}")
    print(f"  Ratio       : {gdn_count}:{attn_count} ({'3:1' if gdn_count==3*attn_count else 'other'})")

    # Forward pass
    B, T = 2, 16
    ids     = torch.randint(0, cfg["vocab_size"], (B, T))
    targets = torch.randint(0, cfg["vocab_size"], (B, T))

    logits, loss = model(ids, targets)
    assert logits.shape == (B, T, cfg["vocab_size"])
    assert loss is not None and loss.item() > 0

    print(f"\n  logits shape : {logits.shape}")
    print(f"  loss         : {loss.item():.4f}  "
          f"(expect ~{torch.log(torch.tensor(cfg['vocab_size'])).item():.2f})")

    # Generation
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=8)
    assert out.shape == (1, 9)
    print(f"  generate     : {out.shape} ✅")

    # Real config
    print(f"\nReal config (DEFAULT_CONFIG):")
    real = HybridDecoder(**DEFAULT_CONFIG)
    unique = sum(p.numel() for p in set(real.parameters()))
    print(f"  unique params: {unique/1e6:.1f}M")
    print(f"\nLayer layout:")
    print(real.layer_summary())
