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

PARAMETER COUNT (v2 config: 32 layers, n_embed 1024, 48k vocab)
----------------------------------------------------------------
  32 layers: 24 GDNBlock + 8 TransformerBlock                ≈ 393.0M
  Token embedding (49,152 × 1024, tied with lm_head)          ≈  50.3M
  Total unique                                                ≈ 443.4M
  (v1 used Qwen's 248k vocab: the embedding alone was 190M.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    qk_norm:     bool = False,
    gdn_impl:    str = "fla_layer",
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
        qk_norm:    QK-norm in the attention blocks (v2)

    Returns:
        ModuleList of alternating GDNBlock and TransformerBlock
    """
    layers = []
    for i in range(n_layers):
        if (i + 1) % attn_every == 0:
            layers.append(TransformerBlock(n_embed, n_heads, n_kv_heads, max_seq_len, qk_norm))
        else:
            layers.append(GDNBlock(n_embed, n_heads, use_fla=use_fla, impl=gdn_impl))
    return nn.ModuleList(layers)


def probs_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                      top_p: float | None = None) -> torch.Tensor:
    """Next-token distribution. temperature 0 → one-hot argmax (greedy).
    top_k keeps the k likeliest tokens; top_p (nucleus) the smallest set whose
    probability reaches p."""
    logits = logits.float()
    if temperature <= 1e-6:
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        drop = cum - sorted_logits.softmax(-1) > top_p        # keep the token that crosses p
        logits = logits.masked_fill(drop.scatter(-1, idx, drop), float("-inf"))
    return logits.softmax(-1)


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
           top_p: float | None = None) -> torch.Tensor:
    """(B, vocab) logits → (B, 1) token ids."""
    return torch.multinomial(probs_from_logits(logits, temperature, top_k, top_p), 1)


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
        qk_norm:     bool = False,
        gdn_impl:    str = "fla_layer",
        rope_theta:  float = 10000.0,
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
            qk_norm:     QK-norm in attention blocks (v2; False keeps v1 checkpoints loadable)
            gdn_impl:    "native" (v2: own GDN weights, FLA kernel on CUDA, decodes
                         anywhere) or "fla_layer" (v1 checkpoints: FLA's module)
            rope_theta:  RoPE base. Raised for long-context extension (ABF).
        """
        super().__init__()
        self.n_embed     = n_embed
        self.max_seq_len = max_seq_len
        self.gradient_checkpointing = False
        self.ce_chunk = 0  # tokens per loss chunk in training; 0 = full logits (set by pretrain.py)
        self.z_loss = 0.0  # z-loss coefficient (PaLM): + z * logsumexp(logits)^2 per token

        self.token_embed = nn.Embedding(vocab_size, n_embed)

        self.layers = _build_layers(
            n_layers, n_embed, n_heads, n_kv_heads, max_seq_len, attn_every,
            use_fla=use_fla, qk_norm=qk_norm, gdn_impl=gdn_impl,
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
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=head_dim, max_seq_len=max_seq_len * 4,
                                                    base=rope_theta)
        # Not saved in checkpoints: rebuilt from max_seq_len / rope_theta, so a
        # checkpoint loads into a model with a longer context or another base.
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

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

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # older checkpoints saved the RoPE tables
        state_dict.pop(prefix + "freqs_cos", None)
        state_dict.pop(prefix + "freqs_sin", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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

        x = self.norm(x)

        # Chunked cross-entropy (Wijmans et al. 2024, "Cut Your Losses"): the full
        # logits tensor (B×T×49k, fp32 for the loss) dominates activation memory.
        # Each chunk's logits are recomputed in backward, so only one chunk lives
        # at a time. Training only; returns logits=None. Opt-in via ce_chunk > 0.
        if targets is not None and self.training and self.ce_chunk > 0:
            if vision_prefix is not None:
                x = x[:, vision_prefix.shape[1]:]
            h, t = x.reshape(-1, x.shape[-1]), targets.reshape(-1).long()

            def _chunk_loss(hc, tc):
                return self._token_loss_sum(self.lm_head(hc), tc)

            total = sum(ckpt.checkpoint(_chunk_loss, h[i:i + self.ce_chunk], t[i:i + self.ce_chunk],
                                        use_reentrant=False)
                        for i in range(0, h.shape[0], self.ce_chunk))
            return None, total / (t != -100).sum().clamp(min=1)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if vision_prefix is not None:
                n_visual        = vision_prefix.shape[1]
                logits_for_loss = logits[:, n_visual:, :]
            else:
                logits_for_loss = logits

            V = logits_for_loss.shape[-1]
            # -100 targets: datasets mark padding positions (everything after
            # the first EOS) so the model is never supervised to spam EOS.
            t = targets.reshape(-1).long()
            loss = self._token_loss_sum(logits_for_loss.reshape(-1, V), t) / (t != -100).sum().clamp(min=1)

        return logits, loss

    def _token_loss_sum(self, logits: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Summed cross-entropy (+ z-loss) over tokens whose target is not -100."""
        logits = logits.float()
        loss = F.cross_entropy(logits, t, ignore_index=-100, reduction="sum")
        if self.z_loss:
            lse = torch.logsumexp(logits, dim=-1)
            loss = loss + self.z_loss * (lse.square() * (t != -100)).sum()
        return loss

    # ── cached inference ───────────────────────────────────────────────────
    # cache = {"pos": tokens seen so far, "layers": one dict per layer}
    #   attention layer: {"k", "v"}  keys/values of every past token (KV cache)
    #   GDN layer:       {"S"}       the fixed-size memory matrix
    # Prefill runs the prompt once; each new token then costs one step.

    def new_cache(self) -> dict:
        return {"pos": 0, "layers": [{} for _ in self.layers]}

    @torch.no_grad()
    def forward_cached(self, token_ids: torch.Tensor | None, cache: dict,
                       embeds: torch.Tensor | None = None, keep_states: bool = False) -> torch.Tensor:
        """Run new tokens (and/or embeddings, e.g. a visual prefix placed before
        them) through the model, continuing from ``cache``. Returns logits for
        every new position: (B, T_new, vocab). keep_states → rollback() works."""
        parts = ([embeds] if embeds is not None else []) + \
                ([self.token_embed(token_ids)] if token_ids is not None else [])
        x = torch.cat(parts, dim=1)
        p, T = cache["pos"], x.shape[1]
        assert p + T <= self.freqs_cos.shape[0], f"context {p + T} > RoPE table {self.freqs_cos.shape[0]}"
        cos, sin = self.freqs_cos[p:p + T], self.freqs_sin[p:p + T]
        for layer, c in zip(self.layers, cache["layers"]):
            x = layer(x, cos, sin, cache=c, keep_states=keep_states)
        cache["base"], cache["pos"] = p, p + T
        return self.lm_head(self.norm(x))

    @staticmethod
    def rollback(cache: dict, n_keep: int) -> None:
        """After forward_cached(..., keep_states=True), keep only its first n_keep
        new tokens (speculative decoding drops rejected draft tokens)."""
        pos = cache["base"] + n_keep
        for c in cache["layers"]:
            if "k" in c:                                  # attention: slice the KV cache
                c["k"], c["v"] = c["k"][:, :, :pos], c["v"][:, :, :pos]
            if "S_steps" in c:                            # GDN: memory after token n_keep
                c["S"] = c["S_steps"][n_keep - 1] if n_keep else c["S_base"]
        cache["pos"] = pos

    @torch.no_grad()
    def generate(
        self,
        token_ids:      torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 1.0,
        top_k:          int | None = None,
        top_p:          float | None = None,
        eos_token_id:   int | None = None,
        vision_prefix:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Cached autoregressive generation: the prompt (plus optional visual
        prefix) is processed once, then every step feeds only the newest token.
        Returns (B, T + generated). Finished rows are padded with eos_token_id.
        """
        cache = self.new_cache()
        logits = self.forward_cached(token_ids, cache, embeds=vision_prefix)[:, -1]
        finished = torch.zeros(token_ids.shape[0], dtype=torch.bool, device=token_ids.device)
        for _ in range(max_new_tokens):
            next_id = sample(logits, temperature, top_k, top_p)
            if eos_token_id is not None:
                next_id = torch.where(finished[:, None], torch.full_like(next_id, eos_token_id), next_id)
                finished |= next_id[:, 0] == eos_token_id
            token_ids = torch.cat([token_ids, next_id], dim=1)
            if eos_token_id is not None and finished.all():
                break
            logits = self.forward_cached(next_id, cache)[:, -1]
        return token_ids

    def generate_cached(self, vision_prefix, token_ids, max_new_tokens=256, temperature=0.0,
                        eos_token_id=None):
        """VLM entry point (src/models/vlm.py): greedy by default."""
        return self.generate(token_ids, max_new_tokens, temperature, eos_token_id=eos_token_id,
                             vision_prefix=vision_prefix)

    def layer_summary(self) -> str:
        """Print which layer is GDN vs Attention — useful for debugging."""
        lines = []
        for i, layer in enumerate(self.layers):
            kind = "GDNBlock      " if isinstance(layer, GDNBlock) else "TransformerBlock"
            lines.append(f"  Layer {i:2d}: {kind}")
        return "\n".join(lines)


# ── default config ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = dict(   # v2 — mirrors configs/base.json
    vocab_size  = 49_152,
    n_embed     = 1024,
    n_heads     = 16,
    n_kv_heads  = 4,
    n_layers    = 32,
    max_seq_len = 1024,
    attn_every  = 4,   # 3:1 ratio
    use_fla     = False,
    qk_norm     = True,
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
