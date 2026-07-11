"""
Akshara VLM — Full Model Assembly
=====================================

This is the end-to-end model. It wires three components built in earlier stages:

  VisionEncoder (ViT-S/16, Stage 08)
      ↓ 196 × 384 patch tokens
  Connector (2-layer MLP, Stage 09)
      ↓ 196 × 768 visual tokens (decoder's language)
  HybridDecoder (3:1 GDN, Stage 07)
      ↓ next-token logits → Nepali/English text

TRAINING STAGES
---------------
This model is trained in two stages:

  Stage 1: Language pretraining (decoder only)
    - Train HybridDecoder on Nepali + English text corpus
    - Vision encoder and connector are FROZEN (or untouched)
    - Teaches the decoder grammar, vocabulary, character patterns
    - Corpus: Wikipedia (ne), CC-100 (ne), OSCAR (ne), mixed English
    - WHY: A decoder that already understands Nepali learns OCR 10× faster

  Stage 2: OCR fine-tuning (full model, all weights)
    - Input: rendered text images (SynthTIGER or PIL)
    - Target: the ground-truth text string
    - All three components trained together end-to-end
    - Loss: cross-entropy on text tokens only (visual prefix excluded)
    - WHY end-to-end: vision encoder learns OCR-specific features
      (sharp edges, stroke patterns) rather than natural image features

FORWARD PASS
------------
  images      (B, 3, 224, 224)
      → encoder   → (B, 196, 384)
      → connector → (B, 196, 768)    ← visual prefix
      → decoder   ← (B, T)           ← text token IDs
          ↓
      (B, 196+T, vocab_size)          ← logits (only T portion supervised)

PARAMETER TOTAL
---------------
  VisionEncoder  : 21.6M
  Connector      :  0.9M
  HybridDecoder  : 273.0M  (dominated by 190M embedding table)
  Total          : ~296M
  Unique (tied)  : ~296M  (weight tying is inside HybridDecoder)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.vit import VisionEncoder
from src.models.connector import Connector
from src.models.hybrid_decoder import HybridDecoder


class Akshara(nn.Module):
    """
    End-to-end OCR VLM for Nepali (Devanagari) + English.
    """

    def __init__(
        self,
        # Vision encoder config (ViT-S/16)
        img_size:    int = 448,
        patch_size:  int = 16,
        in_channels: int = 3,
        vision_dim:  int = 384,
        vit_layers:  int = 12,
        vit_heads:   int = 6,
        # Decoder config
        vocab_size:  int = 248_077,
        n_embed:     int = 768,    # renamed from decoder_dim to match ModelConfig / HybridDecoder
        n_heads:     int = 12,
        n_kv_heads:  int = 3,
        n_layers:    int = 12,
        max_seq_len: int = 512,
        attn_every:  int = 4,
    ):
        super().__init__()

        self.encoder = VisionEncoder(
            img_size    = img_size,
            patch_size  = patch_size,
            in_channels = in_channels,
            embed_dim   = vision_dim,
            n_layers    = vit_layers,
            n_heads     = vit_heads,
        )

        self.connector = Connector(
            vision_dim  = vision_dim,
            decoder_dim = n_embed,
        )

        self.decoder = HybridDecoder(
            vocab_size  = vocab_size,
            n_embed     = n_embed,
            n_heads     = n_heads,
            n_kv_heads  = n_kv_heads,
            n_layers    = n_layers,
            max_seq_len = max_seq_len,
            attn_every  = attn_every,
        )

    def forward(
        self,
        images:    torch.Tensor,
        token_ids: torch.Tensor,
        targets:   torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            images:    (batch, 3, img_size, img_size) — document image crops
            token_ids: (batch, seq_len) — text token IDs (OCR target so far)
            targets:   (batch, seq_len) — next-token targets for loss

        Returns:
            logits: (batch, seq_len, vocab_size)  — text portion only
            loss:   scalar or None
        """
        # Vision path
        patch_tokens  = self.encoder(images)       # (B, n_patches, vision_dim)
        visual_prefix = self.connector(patch_tokens)  # (B, n_patches, decoder_dim)

        # Language path — decoder reads visual prefix then generates text
        logits, loss = self.decoder(
            token_ids     = token_ids,
            targets       = targets,
            vision_prefix = visual_prefix,
        )

        # logits shape from decoder: (B, n_patches + T, vocab_size)
        # Slice to text portion only — callers only care about text predictions
        n_visual = visual_prefix.shape[1]
        logits   = logits[:, n_visual:, :]   # (B, T, vocab_size)

        return logits, loss

    @torch.no_grad()
    def generate_step(
        self,
        images:    torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single generation step: encode image + run decoder → return next-token logits.

        Used by inference loops that manage the token buffer externally (generate.py,
        train_ocr.py CER evaluation). Encodes the image on every call; callers that
        need multiple steps should cache the visual prefix themselves using the lower-
        level encoder/connector/decoder API.

        Args:
            images:    (1, 3, img_size, img_size) — preprocessed image
            token_ids: (1, T) — token IDs generated so far

        Returns:
            (1, vocab_size) — logits for the next token position
        """
        logits, _ = self.forward(images, token_ids)
        return logits[:, -1, :]   # last position → next-token logits

    @torch.no_grad()
    def generate(
        self,
        images:         torch.Tensor,
        bos_token_id:   int,
        eos_token_id:   int,
        max_new_tokens: int = 256,
        temperature:    float = 0.0,
    ) -> torch.Tensor:
        """
        Run OCR inference on a batch of images.

        Args:
            images:         (batch, 3, img_size, img_size)
            bos_token_id:   start-of-sequence token
            eos_token_id:   stop generation when this token is produced
            max_new_tokens: safety limit on generated length
            temperature:    0.0 = greedy (default; right for verbatim OCR)

        Returns:
            token_ids: (batch, n_generated) — generated text token IDs
        """
        B = images.shape[0]
        device = images.device

        # Encode image once (no need to re-encode per generation step)
        patch_tokens  = self.encoder(images)
        visual_prefix = self.connector(patch_tokens)
        n_visual      = visual_prefix.shape[1]

        # Text budget: decoder positions cover visual prefix + text
        max_text = self.decoder.max_seq_len

        # Start with just the BOS token
        token_ids = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            ids_cond = token_ids[:, -max_text:]

            logits, _ = self.decoder(
                token_ids     = ids_cond,
                vision_prefix = visual_prefix,
            )

            next_logits = logits[:, -1, :]
            if temperature <= 1e-6:
                # Greedy — the right default for verbatim OCR
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs   = torch.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            # Sequences that already emitted EOS keep emitting EOS
            next_id = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_id, eos_token_id),
                next_id,
            )
            token_ids = torch.cat([token_ids, next_id], dim=1)

            finished |= next_id.squeeze(1) == eos_token_id
            if finished.all():
                break

        return token_ids

    def freeze_encoder(self):
        """Freeze vision encoder during Stage 1 language pretraining."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.connector.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze everything for Stage 2 end-to-end OCR fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True

    def set_gradient_checkpointing(self, enabled: bool):
        """Enable in decoder to save VRAM on Kaggle T4."""
        self.decoder.set_gradient_checkpointing(enabled)

    def param_summary(self) -> str:
        def count(module):
            return sum(p.numel() for p in module.parameters())

        enc  = count(self.encoder)
        conn = count(self.connector)
        dec  = count(self.decoder)
        # Unique counts (weight tying deduplicates)
        unique = sum(p.numel() for p in set(self.parameters()))

        return (
            f"  VisionEncoder  : {enc/1e6:.1f}M\n"
            f"  Connector      : {conn/1e6:.1f}M\n"
            f"  HybridDecoder  : {dec/1e6:.1f}M\n"
            f"  Total unique   : {unique/1e6:.1f}M"
        )


# ── default config ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = dict(
    img_size    = 448,
    patch_size  = 16,
    in_channels = 3,
    vision_dim  = 384,
    vit_layers  = 12,
    vit_heads   = 6,
    vocab_size  = 248_077,
    n_embed     = 768,
    n_heads     = 12,
    n_kv_heads  = 3,
    n_layers    = 12,
    max_seq_len = 512,
    attn_every  = 4,
)


# ── self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    print("Building Akshara VLM...")
    t0 = time.time()
    model = Akshara(**DEFAULT_CONFIG)
    print(f"  build time   : {time.time()-t0:.1f}s")
    print()
    print("Parameter breakdown:")
    print(model.param_summary())

    # Forward pass (training mode)
    B, T = 2, 16
    images    = torch.randn(B, 3, 448, 448)
    token_ids = torch.randint(0, 1000, (B, T))
    targets   = torch.randint(0, 1000, (B, T))

    logits, loss = model(images, token_ids, targets)

    n_patches = model.encoder.patch_embed.n_patches
    assert logits.shape == (B, T, DEFAULT_CONFIG["vocab_size"]), \
        f"logits shape wrong: {logits.shape}"
    assert loss is not None and loss.item() > 0

    print(f"\nForward pass:")
    print(f"  images shape  : {tuple(images.shape)}")
    print(f"  token_ids     : {tuple(token_ids.shape)}")
    print(f"  visual prefix : (B, {n_patches}, {DEFAULT_CONFIG['n_embed']})")
    print(f"  logits shape  : {tuple(logits.shape)}  ✅")
    print(f"  loss          : {loss.item():.4f}")

    # Freeze test
    model.freeze_encoder()
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total  = sum(1 for p in model.parameters())
    print(f"\nFreeze encoder:")
    print(f"  frozen params : {frozen}/{total} param groups  ✅")

    model.unfreeze_all()
    frozen_after = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f"  after unfreeze: {frozen_after} frozen  ✅")

    # Gradient checkpointing
    model.set_gradient_checkpointing(True)
    print(f"\nGradient checkpointing: enabled  ✅")
