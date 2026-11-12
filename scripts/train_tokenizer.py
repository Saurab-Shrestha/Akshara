"""
Train Akshara's own byte-level BPE tokenizer (v2).

WHY OUR OWN TOKENIZER
---------------------
v1 borrowed Qwen3.5's 248k vocab. For a ~110M-param decoder that was the
single biggest waste in the model:
    - embedding table 248k × 768 = 190M params — more than the whole body
    - output head ~380M FLOPs/token — more than all 16 layers combined
A 32k vocab trained on *our* languages (ne/hi/mr/sa + en) cuts both by ~8×
and spends the compute on the layers that actually learn.

DESIGN
------
- Byte-level BPE (GPT-2 / Qwen style): every string is encodable, no <unk>.
- NFC normalisation: Devanagari has multiple byte sequences for the same
  glyph (nukta forms); NFC makes them one.
- Custom pre-tokenizer split. The GPT-2 regex groups ` ?\\p{L}+`, but
  Devanagari vowel signs and virama are \\p{M} (marks), so "नेपाल" would be
  torn into "न", "े", "पाल" before BPE ever sees it. We put \\p{M} and
  ZWJ/ZWNJ (used in Nepali half-forms like र्‍) in the word class.
- Digits grouped 1–3 (LLaMA-3 style) so numbers don't eat vocab.
- Special tokens reserved NOW (chat + image placeholders) so the MLLM
  stages never need to resize the embedding.

USAGE
-----
    PYTHONPATH=. python scripts/train_tokenizer.py \\
        --input data/corpus_v5/train.jsonl --vocab_size 32768 \\
        --out_dir tokenizer/akshara-bpe --compare Qwen/Qwen3.5-0.8B

Prints chars/token per script on held-out docs (higher = better compression)
and asserts decode(encode(x)) == NFC(x).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import unicodedata

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

SPECIAL_TOKENS = [
    "<|bos|>", "<|eos|>", "<|pad|>",
    "<|image|>",                      # placeholder replaced by vision tokens
    "<|im_start|>", "<|im_end|>",     # chat turns (instruction tuning)
] + [f"<|reserved_{i}|>" for i in range(10)]

SPLIT_PATTERN = (
    r" ?[\p{L}\p{M}\x{200C}\x{200D}]+"   # words incl. matras/virama/ZWJ
    r"| ?\p{N}{1,3}"                      # numbers in groups of ≤3
    r"| ?[^\s\p{L}\p{M}\p{N}]+"           # punctuation runs (। ॥ . , …)
    r"|\s+(?!\S)|\s+"                     # whitespace
)


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def iter_texts(path: str, sample_bytes: int, seed: int):
    """Yield a uniform random sample of ~sample_bytes from a JSONL corpus.

    prepare_data.py writes sources one after another (Nepali, then Hindi, …),
    so taking the first N lines would train an all-Nepali tokenizer.
    """
    keep = min(1.0, sample_bytes / max(1, os.path.getsize(path)))
    rng = random.Random(seed)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if rng.random() < keep:
                yield json.loads(line)["text"]


def script_of(text: str) -> str:
    deva = sum("ऀ" <= c <= "ॿ" for c in text)
    latin = sum(c.isascii() and c.isalpha() for c in text)
    return "devanagari" if deva >= latin else "latin"


def compression_report(encode, texts, label):
    chars = {"devanagari": 0, "latin": 0}
    toks = {"devanagari": 0, "latin": 0}
    for t in texts:
        s = script_of(t)
        chars[s] += len(t)
        toks[s] += len(encode(t))
    row = " | ".join(f"{s}: {chars[s] / max(1, toks[s]):.2f}" for s in chars)
    print(f"  {label:<28} chars/token  {row}")


def main():
    ap = argparse.ArgumentParser(description="Train Akshara byte-level BPE tokenizer")
    ap.add_argument("--input", default="data/corpus_v5/train.jsonl")
    ap.add_argument("--eval", default=None, help="held-out JSONL for the report (default: --input sibling val.jsonl)")
    ap.add_argument("--vocab_size", type=int, default=32768)
    ap.add_argument("--sample_bytes", type=float, default=2e9, help="training sample size (bytes of JSONL)")
    ap.add_argument("--out_dir", default="tokenizer/akshara-bpe")
    ap.add_argument("--compare", default=None, help="HF tokenizer to compare against, e.g. Qwen/Qwen3.5-0.8B")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # uint16 token shards (tokenize_corpus.py) need ids < 65536
    assert args.vocab_size <= 65535, "vocab_size must fit uint16 for the packed .bin shards"

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print(f"[tokenizer] training {args.vocab_size:,}-vocab BPE on ~{args.sample_bytes/1e9:.1f} GB of {args.input}")
    tok.train_from_iterator(iter_texts(args.input, int(args.sample_bytes), args.seed), trainer=trainer)

    from transformers import PreTrainedTokenizerFast
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|bos|>", eos_token="<|eos|>", pad_token="<|pad|>",
        additional_special_tokens=SPECIAL_TOKENS[3:],
    )
    hf_tok.save_pretrained(args.out_dir)
    print(f"[tokenizer] saved → {args.out_dir} (len={len(hf_tok):,})")

    # --- held-out report + round-trip check ---
    eval_path = args.eval or os.path.join(os.path.dirname(args.input), "val.jsonl")
    if not os.path.exists(eval_path):
        print(f"[tokenizer] no eval file at {eval_path}; skipping report")
        return
    with open(eval_path, encoding="utf-8") as f:
        texts = [json.loads(l)["text"] for _, l in zip(range(2000), f)]

    for t in texts[:500]:
        ids = hf_tok.encode(t, add_special_tokens=False)
        assert hf_tok.decode(ids) == unicodedata.normalize("NFC", t), "round-trip failed"
    print("[tokenizer] round-trip decode(encode(x)) == NFC(x) ✅")

    print("[tokenizer] compression on held-out docs (higher is better):")
    compression_report(lambda t: hf_tok.encode(t, add_special_tokens=False), texts, f"akshara-bpe {len(hf_tok)//1000}k")
    if args.compare:
        from transformers import AutoTokenizer
        ref = AutoTokenizer.from_pretrained(args.compare)
        compression_report(lambda t: ref.encode(t, add_special_tokens=False), texts, f"{args.compare} {len(ref)//1000}k")


if __name__ == "__main__":
    main()
