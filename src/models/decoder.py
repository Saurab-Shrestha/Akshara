"""
Decoder — The Language Model
=============================

WHY THIS EXISTS
---------------
This is the text-generation half of our VLM. Given a sequence of token
IDs, it predicts the probability of the next token at every position.

During OCR training, the decoder receives:
  - visual tokens from the vision encoder (the image)
  - text tokens generated so far
And predicts: what text token comes next?

During text-only pretraining (Stage 1), it receives only text tokens
and learns the structure of Nepali + English language. This pretrain
is crucial — a decoder that understands language will learn OCR much
faster than one starting from random weights.

ARCHITECTURE
------------
  Token Embedding      : token IDs → dense vectors (n_embed dimensions)
  [No position embed]  : RoPE handles position inside attention instead
  N × TransformerBlock : each block refines representations
  RMSNorm              : final normalization before prediction
  LM Head              : project n_embed → vocab_size (next-token logits)

OUR CONFIG (fits Kaggle T4 16GB)
---------------------------------
  vocab_size  : 248,044  (Qwen3.5-0.8B tokenizer)
  n_embed     : 768      (hidden dimension)
  n_heads     : 12       (query attention heads)
  n_kv_heads  : 3        (GQA: 4 query heads per KV head)
  n_layers    : 12       (transformer blocks stacked)
  max_seq_len : 512      (enough for a text line / short paragraph)
  head_dim    : 64       (= n_embed / n_heads)

PARAMETER COUNT (approximate)
-------------------------------
  Token embedding        : 248,044 × 768        ≈  190M   ← largest single piece
  Per block              : ~705K × 12            ≈    8M
  LM head (tied)         : shares embedding weights  ≈  0 extra
  Total                  :                       ≈  198M

  Wait — that's more than our 100M decoder target!
  The embedding table dominates because our vocab is 248k.
  The transformer compute (blocks) is only ~8M.

  Solution: WEIGHT TYING — the LM head shares weights with the token
  embedding. Prediction "un-embeds" using the same matrix that embeds.
  This is standard practice (GPT-2, LLaMA, Qwen all do it).
  No extra params for lm_head.

  With tying: total unique params ≈ 198M (embedding + 12 blocks).
  Without tying: 198M + 190M = 388M.

GRADIENT CHECKPOINTING
-----------------------
  Enabled via model.set_gradient_checkpointing(True).
  Trades ~20% more compute for ~40% less VRAM during training.
  Essential for Kaggle T4 with large batches.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from src.models.transformer_block import TransformerBlock
from src.models.rms_norm import RMSNorm
from src.models.rope import precompute_freqs_cis


class Decoder(nn.Module):

    def __init__(
        self,
        vocab_size:  int,
        n_embed:     int,
        n_heads:     int,
        n_kv_heads:  int,
        n_layers:    int,
        max_seq_len: int,
    ):
        """
        Args:
            vocab_size:  number of tokens in the tokenizer vocabulary
            n_embed:     embedding / hidden dimension
            n_heads:     number of query attention heads per block
            n_kv_heads:  number of KV heads per block (GQA)
            n_layers:    number of transformer blocks
            max_seq_len: maximum input sequence length
        """
        super().__init__()
        self.n_embed     = n_embed
        self.n_heads     = n_heads
        self.max_seq_len = max_seq_len
        self.gradient_checkpointing = False

        # Token embedding: ID → dense vector
        # vocab_size × n_embed matrix; each row is one token's embedding
        self.token_embed = nn.Embedding(vocab_size, n_embed)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, n_heads, n_kv_heads, max_seq_len)
            for _ in range(n_layers)
        ])

        # Final normalization before prediction
        self.norm = RMSNorm(n_embed)

        # LM head: n_embed → vocab_size logits (next-token scores)
        # bias=False is standard; we tie weights below
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        # Weight tying: lm_head shares the token_embed matrix
        # Embeds and un-embeds use the same learned vectors → fewer params,
        # often better performance (representations stay consistent)
        self.lm_head.weight = self.token_embed.weight

        # Precompute RoPE rotation factors for all positions up to max_seq_len
        # These are fixed math (not learned), stored as a buffer
        head_dim  = n_embed // n_heads
        freqs_cis = precompute_freqs_cis(dim=head_dim, max_seq_len=max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)

        # Initialize weights sensibly
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Small init variance prevents exploding activations at the start of training."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_gradient_checkpointing(self, enabled: bool):
        """Call model.set_gradient_checkpointing(True) before training on Kaggle."""
        self.gradient_checkpointing = enabled

    def forward(
        self,
        token_ids: torch.Tensor,
        targets:   torch.Tensor = None,
        vision_prefix: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            token_ids:     (batch, seq_len) integer token IDs
            targets:       (batch, seq_len) next-token targets for loss (optional)
            vision_prefix: (batch, n_patches, n_embed) visual tokens prepended
                           before text tokens. Used during VLM training (Stage 4+).
                           None during text-only pretraining (Stage 1).

        Returns:
            logits: (batch, total_seq_len, vocab_size)
            loss:   scalar cross-entropy loss, or None if targets not provided
        """
        B, T = token_ids.shape

        # Embed text tokens
        x = self.token_embed(token_ids)  # (B, T, n_embed)

        # Prepend visual tokens if provided (VLM mode)
        # Visual tokens come from the connector (Step 10); they are already
        # n_embed dimensional so we just concatenate in the sequence dimension
        if vision_prefix is not None:
            x = torch.cat([vision_prefix, x], dim=1)  # (B, n_patches + T, n_embed)

        total_len = x.shape[1]

        # Slice the precomputed RoPE factors for the actual sequence length
        freqs_cis = self.freqs_cis[:total_len]

        # Pass through all transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                # Recompute block activations in backward → saves ~40% VRAM
                # lambda needed because checkpoint requires positional args only
                x = ckpt.checkpoint(block, x, freqs_cis, use_reentrant=False)
            else:
                x = block(x, freqs_cis)

        # Final norm + project to vocabulary logits
        x      = self.norm(x)
        logits = self.lm_head(x)  # (B, total_len, vocab_size)

        # Compute loss only if targets are provided
        loss = None
        if targets is not None:
            # If visual prefix was prepended, we only supervise the text portion
            # (we don't ask the model to predict visual tokens from text)
            if vision_prefix is not None:
                n_visual = vision_prefix.shape[1]
                logits_for_loss = logits[:, n_visual:, :]  # text portion only
            else:
                logits_for_loss = logits

            B, L, V = logits_for_loss.shape
            loss = torch.nn.functional.cross_entropy(
                logits_for_loss.reshape(B * L, V),
                targets.reshape(B * L).long(),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Autoregressive generation: feed output back as input, one token at a time.

        Args:
            token_ids:      (1, seq_len) starting token IDs
            max_new_tokens: how many tokens to generate
            temperature:    >1 = more random, <1 = more greedy, 1 = standard

        Returns:
            (1, seq_len + max_new_tokens) extended token IDs
        """
        for _ in range(max_new_tokens):
            # Trim to max_seq_len if needed (sliding window)
            ids_cond = token_ids[:, -self.max_seq_len:]

            logits, _ = self(ids_cond)

            # Take logits at the last position (next token prediction)
            next_logits = logits[:, -1, :] / temperature

            # Sample from the distribution
            probs    = torch.softmax(next_logits, dim=-1)
            next_id  = torch.multinomial(probs, num_samples=1)

            token_ids = torch.cat([token_ids, next_id], dim=1)

        return token_ids


# ── default config for our Kaggle-friendly ~100M compute model ────────────────
DEFAULT_CONFIG = dict(
    vocab_size  = 248_044,  # Qwen3.5-0.8B tokenizer
    n_embed     = 768,
    n_heads     = 12,
    n_kv_heads  = 3,
    n_layers    = 12,
    max_seq_len = 512,
)


# ── self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Use a tiny config for fast testing (not the real config)
    cfg = dict(vocab_size=1000, n_embed=128, n_heads=4, n_kv_heads=2, n_layers=2, max_seq_len=64)
    model = Decoder(**cfg)

    # Count parameters
    total  = sum(p.numel() for p in model.parameters())
    unique = sum(p.numel() for p in set(model.parameters()))  # weight tying deduplicates
    print(f"Decoder self-check")
    print(f"  total param refs : {total:,}  (includes tied weight counted twice)")
    print(f"  unique params    : {unique:,}  (actual memory footprint)")

    # Forward pass
    B, T = 2, 16
    ids     = torch.randint(0, cfg["vocab_size"], (B, T))
    targets = torch.randint(0, cfg["vocab_size"], (B, T))

    logits, loss = model(ids, targets)
    assert logits.shape == (B, T, cfg["vocab_size"]), f"wrong logits shape: {logits.shape}"
    assert loss is not None and loss.item() > 0

    print(f"  logits shape     : {logits.shape}")
    print(f"  loss             : {loss.item():.4f}  (expect ~log({cfg['vocab_size']}) = {torch.log(torch.tensor(cfg['vocab_size'])).item():.2f})")

    # Generation
    start = torch.zeros(1, 1, dtype=torch.long)
    out   = model.generate(start, max_new_tokens=10)
    assert out.shape == (1, 11)
    print(f"  generate shape   : {out.shape}  ✅")

    # Real config param count
    print(f"\nReal config (DEFAULT_CONFIG):")
    real_model = Decoder(**DEFAULT_CONFIG)
    real_unique = sum(p.numel() for p in set(real_model.parameters()))
    print(f"  unique params    : {real_unique/1e6:.1f}M")
