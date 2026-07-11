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
    use_fla:     bool = False,
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
        use_fla:    use FLA Triton kernel for GDN layers (GPU training speedup)

    Returns:
        ModuleList of alternating GDNBlock and TransformerBlock
    """
    layers = []
    for i in range(n_layers):
        if (i + 1) % attn_every == 0:
            layers.append(TransformerBlock(n_embed, n_heads, n_kv_heads, max_seq_len))
        else:
            layers.append(GDNBlock(n_embed, n_heads, use_fla=use_fla))
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
        use_fla:     bool = False,
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
            use_fla:     use FLA Triton kernel for GDN layers
        """
        super().__init__()
        self.n_embed     = n_embed
        self.max_seq_len = max_seq_len
        self.gradient_checkpointing = False

        self.token_embed = nn.Embedding(vocab_size, n_embed)

        self.layers = _build_layers(
            n_layers, n_embed, n_heads, n_kv_heads, max_seq_len, attn_every,
            use_fla=use_fla,
        )

        self.norm    = RMSNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        # Weight tying: share embedding and lm_head weights
        self.lm_head.weight = self.token_embed.weight

        # Precompute RoPE tables (real-valued cos/sin — no complex tensors so
        # DataParallel buffer replication works correctly).
        # ×4 covers the visual prefix + full text at any supported resolution:
        # 448px → 1024 patches, 518px → 1369 patches; 1369 + 512 < 2048.
        head_dim       = n_embed // n_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=head_dim, max_seq_len=max_seq_len * 4)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        self.apply(self._init_weights)

        # Depth-scaled init on residual-branch output projections (GPT-2/LLaMA):
        # without this the residual stream variance grows linearly over
        # 2*n_layers sublayers.
        residual_scale = 0.02 / (2 * n_layers) ** 0.5
        for layer in self.layers:
            for name, p in layer.named_parameters():
                if name.endswith("wo.weight") or name.endswith("w_out.weight"):
                    nn.init.normal_(p, mean=0.0, std=residual_scale)

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
            # ignore_index=-100: datasets mark padding positions (everything
            # after the first EOS) as -100 so the model is never supervised
            # to spam EOS over padding.
            loss = torch.nn.functional.cross_entropy(
                logits_for_loss.reshape(B * L, V),
                targets.reshape(B * L).long(),
                ignore_index=-100,
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

    @torch.no_grad()
    def generate_cached(
        self,
        vision_prefix:  torch.Tensor,
        token_ids:      torch.Tensor,
        max_new_tokens: int = 256,
        temperature:    float = 0.0,
        eos_token_id:   int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with GDN state caching.

        The visual prefix is processed once through all layers; GDN memory
        states are cached after the prefix.  For each decode step, GDN
        layers only process the new token (using cached state — O(1) per
        step instead of O(T)), while attention layers still attend to
        the full accumulated sequence.

        Per-layer hidden state buffers grow as tokens are generated
        (12 × ~3k × 768 = ~28 MB at max length — negligible).

        Args:
            vision_prefix: (batch, n_patches, n_embed) — from the connector
            token_ids:     (batch, T) — initial text tokens (at least BOS)
            max_new_tokens: safety cap
            temperature:   0.0 = greedy
            eos_token_id:  stop at this token

        Returns:
            (batch, T + generated) — full sequence including input tokens
        """
        device = vision_prefix.device
        B = vision_prefix.shape[0]

        # ── Pre-fill: process visual prefix, build GDN states & hidden buffer ──
        T_v = vision_prefix.shape[1]
        cos = self.freqs_cos[:T_v]
        sin = self.freqs_sin[:T_v]

        # layer_out[li] = output of layer li (layer_out[0] = input to layer 0)
        layer_out = [vision_prefix]
        x = vision_prefix
        for layer in self.layers:
            if isinstance(layer, GDNBlock):
                layer.gdn._capture_state = True
                x = layer(x, cos, sin)
            else:
                x = layer(x, cos, sin)
            layer_out.append(x)

        # Collect captured GDN states
        gdn_states: list[dict | None] = []
        for layer in self.layers:
            if isinstance(layer, GDNBlock):
                gdn_states.append(getattr(layer.gdn, '_final_state', None))
                layer.gdn._capture_state = False
            else:
                gdn_states.append(None)

        text_tokens = token_ids
        finished    = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            x_t = self.token_embed(text_tokens[:, -1:])  # (B, 1, n_embed)
            gdn_idx = -1

            for li, layer in enumerate(self.layers):
                L = layer_out[li].shape[1]

                if isinstance(layer, GDNBlock):
                    gdn_idx += 1
                    new_out, new_state = layer.gdn.step(
                        layer.norm1(x_t).squeeze(1),
                        gdn_states[gdn_idx],
                    )
                    new_out = x_t.squeeze(1) + new_out
                    new_out = new_out + layer.mlp(layer.norm2(new_out))
                    gdn_states[gdn_idx] = new_state
                    new_out = new_out.unsqueeze(1)
                    layer_out[li + 1] = torch.cat([layer_out[li + 1], new_out], dim=1)
                    x_t = new_out
                else:
                    # GDN layers before this have already appended the new
                    # token's output to layer_out[li], so the full sequence
                    # is already in place — no need to cat x_t.
                    out = layer(layer_out[li], self.freqs_cos[:L], self.freqs_sin[:L])
                    layer_out[li + 1] = out
                    x_t = out[:, -1:, :]

            # LM head
            last_hid = self.norm(x_t)
            next_logit = self.lm_head(last_hid).squeeze(1)

            if temperature <= 1e-6:
                next_id = next_logit.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logit / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            # Force EOS for finished sequences (or 0 if eos_token_id not set)
            eos_val = eos_token_id if eos_token_id is not None else 0
            next_id = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_id, eos_val),
                next_id,
            )
            text_tokens = torch.cat([text_tokens, next_id], dim=1)
            if eos_token_id is not None:
                finished |= next_id.squeeze(1) == eos_token_id
                if finished.all():
                    break

        return text_tokens

    def layer_summary(self) -> str:
        """Print which layer is GDN vs Attention — useful for debugging."""
        lines = []
        for i, layer in enumerate(self.layers):
            kind = "GDNBlock      " if isinstance(layer, GDNBlock) else "TransformerBlock"
            lines.append(f"  Layer {i:2d}: {kind}")
        return "\n".join(lines)


# ── default config ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = dict(
    vocab_size  = 248_077,
    n_embed     = 768,
    n_heads     = 12,
    n_kv_heads  = 3,
    n_layers    = 12,
    max_seq_len = 512,
    attn_every  = 4,   # 3:1 ratio
    use_fla     = False,
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
