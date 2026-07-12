"""
Dataclass-based configuration for the Akshara project.

WHY dataclasses instead of plain dicts:
    - Field names are type-checked and discoverable via IDE completion.
    - Inheritance lets PretrainConfig and OCRFinetuneConfig share all model architecture
      fields from ModelConfig without duplication.
    - ``dataclasses.fields()`` is used by the config loader to validate JSON keys.

THREE CONFIG LEVELS:
    ModelConfig          -- shared architecture knobs (identical across all stages).
    PretrainConfig       -- language-only pretraining (text tokens, no images).
    OCRFinetuneConfig    -- vision-language OCR fine-tuning on image+text pairs.

HARDWARE TARGET: Kaggle T4 (16 GB VRAM).
    The defaults are tuned to fit comfortably with bf16 AMP + gradient checkpointing.
    effective_batch = batch_size * grad_accum = 64 tokens for both stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Shared model architecture
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Architecture constants shared between language pretraining and OCR fine-tuning.

    These must stay identical across all stages because loading a pretrained
    checkpoint into a fine-tuning run requires the same model topology.

    DECODER (HybridDecoder, ~308M params):
        - n_embed=768, n_heads=12, n_kv_heads=3 (GQA 4:1 compression)
        - n_layers=16, attn_every=4  → 12 GDN blocks + 4 attention blocks (3:1 ratio)
        - max_seq_len=512  (fits a full A4 page of dense Nepali text)

    VISION ENCODER (ViT-S/16, ~21.6M params):
        - img_size=224, patch_size=16 → 14×14=196 patch tokens
        - vision_dim=384, vit_layers=12, vit_heads=6

    CONNECTOR (2-layer MLP, ~0.9M params):
        - projects vision_dim (384) → n_embed (768)
    """

    # Decoder / tokenizer
    vocab_size: int = 248077          # Qwen/Qwen3.5-0.8B: 248044 base + 33 special tokens
    n_embed: int = 768                # model dimension (residual stream width)
    n_heads: int = 12                 # number of query heads
    n_kv_heads: int = 3               # GQA: 12/3=4 queries share each KV head
    n_layers: int = 16                # total decoder layers
    max_seq_len: int = 512            # text tokens per crop (a dense paragraph fits)
    attn_every: int = 4               # full attention every N layers; others use GDN

    # Vision encoder (DINOv2-S/14, pretrained)
    img_size: int = 448               # crop canvas size (aspect-preserving pad)
    patch_size: int = 14              # DINOv2 patch size → 32×32=1024 patch tokens at 448px
    vision_dim: int = 384             # DINOv2-small hidden dimension
    vit_layers: int = 12              # (informational — architecture fixed by dinov2-small)
    vit_heads: int = 6                # (informational)
    vit_pretrained: bool = True       # init from facebook/dinov2-small weights

    # FLA (Flash Linear Attention) — GPU-accelerated GDN kernel
    use_fla: bool = False             # set True if flash-linear-attention is installed


# ---------------------------------------------------------------------------
# Stage 1: Language pretraining (text-only)
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig(ModelConfig):
    """
    Configuration for language-only pretraining with SmolLM-style data approach.

    WHY text-only first:
        Training the decoder on language before attaching the vision encoder
        gives it a strong prior on Devanagari syntax and script. When OCR fine-tuning
        starts, the model already knows how to produce valid Unicode Devanagari —
        the vision side only needs to learn *which* characters, not *what* characters
        look like in general.

    DATA APPROACH (SmolLM-inspired):
        Primary: FineWeb-2 (multilingual Devanagari: ne, hi, mr, sa) with quality filtering
        English: FineWeb-Edu (education-filtered, English only)
        Supplement: Wikipedia for all target languages
        Quality: EduScore >= 3 filtering on web data

    BATCH MATH (T4 16GB):
        batch_size=2, grad_accum=8 → effective_batch=16 sequences of 512 tokens
        = 8,192 tokens per optimizer step.  fp32 (T4 doesn't support bf16 natively).

    LR SCHEDULE: cosine decay with warmup.
        warmup_steps=500 prevents early divergence.
        min_lr=3e-5 is 10% of peak — standard ratio.
    """

    # Data paths (overridden by JSON / CLI)
    train_path: str = "data/corpus/train.jsonl"
    dev_path: str = "data/corpus/val.jsonl"

    # Batch / accumulation
    batch_size: int = 16
    grad_accum: int = 4               # effective batch = 16*4 = 64 sequences

    # Training duration
    train_steps: int = 50_000
    eval_steps: int = 500             # run eval every N optimizer steps
    eval_iters: int = 50              # batches averaged per eval
    warmup_steps: int = 2_000

    # Optimisation
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Checkpointing
    out_ckpt: str = "checkpoints/pretrain.pt"
    save_every: int = 1_000

    # Mixed precision + memory
    use_amp: bool = True
    amp_dtype: Optional[str] = "bf16"          # "bf16" or "fp16"; None = disabled
    use_gradient_checkpointing: bool = True     # recomputes activations in backward

    # Compilation
    compile: bool = False                # torch.compile — fuses GDN loop via Triton

    # Misc
    seed: int = 42
    device: str = "cuda"
    log_dir: str = "logs/pretrain"

    # Durable backup: if set (e.g. "username/akshara-pretrain"), every saved
    # checkpoint is also uploaded to this HF Hub repo. Survives the training
    # machine dying. Requires `huggingface-cli login` on the box first.
    hf_repo: Optional[str] = "Saurab0/akshara-pretrain"  # case-sensitive namespace!


# ---------------------------------------------------------------------------
# Stage 2: OCR fine-tuning (vision + language)
# ---------------------------------------------------------------------------

@dataclass
class OCRFinetuneConfig(ModelConfig):
    """
    Configuration for OCR fine-tuning on paired (image, Nepali text) data.

    WHY different batch_size/grad_accum from pretrain:
        Each sample now carries an image tensor (3×224×224 = ~600KB fp32), so
        VRAM per sample is much higher.  batch_size=8 with grad_accum=8 keeps
        the effective batch at 64 while staying within 16GB.

    WARMUP FREEZE STRATEGY (handled in train_ocr.py, not here):
        For the first 1000 steps the vision encoder is frozen; only the connector
        and decoder are trained.  This prevents the pretrained language weights
        from being destroyed by random-init gradients flowing from an untrained
        vision encoder.

    OUTPUT FORMAT — HTML (no bboxes):
        The model generates structured HTML for the entire page:
            <h1>Title</h1><p>Paragraph text.</p><table><tr><td>Cell</td></tr></table>
        No bounding box coordinates — just semantic structure.
        This lets the model handle tables, headers, paragraphs natively
        without needing a separate layout analysis stage.

    CER vs LOSS:
        Perplexity is a poor proxy for OCR quality because it rewards confident
        wrong predictions.  We use Character Error Rate on greedy-decoded outputs
        as the primary validation metric.
    """

    # Data paths — JSONL files with {"image": "path.png", "html": "<p>...</p>"}
    train_path: str = "data/documents/train.jsonl"
    dev_path: str = "data/documents/val.jsonl"

    # Batch / accumulation (lower than pretrain: 784 visual + 2048 text tokens per sample)
    batch_size: int = 2
    grad_accum: int = 16              # effective batch = 2*16 = 32 sequences

    # Training duration
    train_steps: int = 20_000
    eval_steps: int = 250
    eval_iters: int = 25
    warmup_steps: int = 500

    # Optimisation (lower lr than pretrain: fine-tuning preserves existing weights)
    lr: float = 1e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.05
    grad_clip: float = 1.0

    # Checkpoint warm-start from pretrained decoder
    pretrain_ckpt: Optional[str] = None        # path to pretrain.pt; None = train from scratch
    out_ckpt: str = "checkpoints/ocr.pt"
    save_every: int = 500

    # Mixed precision + memory
    use_amp: bool = True
    amp_dtype: Optional[str] = "bf16"
    use_gradient_checkpointing: bool = True

    # Misc
    seed: int = 42
    device: str = "cuda"
    log_dir: str = "logs/ocr"
