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

Prints chars/token per language on held-out docs (higher = better
compression) and asserts decode(encode(x)) == NFC(x). Minority languages of
Nepal (Maithili, Newari, Limbu, …) are oversampled `--boost`× in the
training sample so BPE learns merges for them.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import unicodedata

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from scripts.prepare_data import MINORITY

MINORITY_LANGS = {m[3] for m in MINORITY}

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


def iter_texts(path: str, sample_bytes: int, seed: int, boost: float, counts: dict):
    """Yield a random sample of ~sample_bytes from a JSONL corpus.

    prepare_data.py writes sources one after another (Nepali, then Hindi, …),
    so taking the first N lines would train an all-Nepali tokenizer. Rows
    tagged with a minority language are kept `boost`× more often: BPE only
    learns merges for text it sees, and Maithili/Limbu/… are <1% of the corpus.
    """
    keep = min(1.0, sample_bytes / max(1, os.path.getsize(path)))
    rng = random.Random(seed)
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            lang = row.get("lang") or script_of(row["text"])
            p = keep * boost if lang in MINORITY_LANGS else keep
            if rng.random() < p:
                counts[lang] = counts.get(lang, 0) + len(row["text"])
                yield row["text"]


def script_of(text: str) -> str:
    deva = sum("ऀ" <= c <= "ॿ" for c in text)
    latin = sum(c.isascii() and c.isalpha() for c in text)
    return "devanagari" if deva >= latin else "latin"


def load_eval(path: str, per_lang: int = 300) -> dict[str, list[str]]:
    """Up to `per_lang` held-out docs for every language tag in the file."""
    by_lang: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            lang = row.get("lang") or script_of(row["text"])
            docs = by_lang.setdefault(lang, [])
            if len(docs) < per_lang:
                docs.append(row["text"])
    return by_lang


def chars_per_token(encode, texts) -> float:
    return sum(len(t) for t in texts) / max(1, sum(len(encode(t)) for t in texts))


def main():
    ap = argparse.ArgumentParser(description="Train Akshara byte-level BPE tokenizer")
    ap.add_argument("--input", default="data/corpus_v5/train.jsonl")
    ap.add_argument("--eval", default=None, help="held-out JSONL for the report (default: --input sibling val.jsonl)")
    ap.add_argument("--vocab_size", type=int, default=32768)
    ap.add_argument("--sample_bytes", type=float, default=2e9, help="training sample size (bytes of JSONL)")
    ap.add_argument("--out_dir", default="tokenizer/akshara-bpe")
    ap.add_argument("--compare", default=None, help="HF tokenizer to compare against, e.g. Qwen/Qwen3.5-0.8B")
    ap.add_argument("--boost", type=float, default=5.0, help="oversampling factor for minority languages")
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
    counts: dict[str, int] = {}
    tok.train_from_iterator(
        iter_texts(args.input, int(args.sample_bytes), args.seed, args.boost, counts), trainer=trainer)
    total = sum(counts.values())
    print("[tokenizer] training sample by language: " + ", ".join(
        f"{l} {c/total:.1%}" for l, c in sorted(counts.items(), key=lambda x: -x[1])))

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
    by_lang = load_eval(eval_path)

    for texts in by_lang.values():
        for t in texts[:50]:
            ids = hf_tok.encode(t, add_special_tokens=False)
            assert hf_tok.decode(ids) == unicodedata.normalize("NFC", t), "round-trip failed"
    print("[tokenizer] round-trip decode(encode(x)) == NFC(x) ✅")

    ours = lambda t: hf_tok.encode(t, add_special_tokens=False)
    ref = None
    if args.compare:
        from transformers import AutoTokenizer
        ref_tok = AutoTokenizer.from_pretrained(args.compare)
        ref = lambda t: ref_tok.encode(t, add_special_tokens=False)

    print(f"[tokenizer] chars/token on held-out docs (higher is better)")
    print(f"  {'lang':<6} {'docs':>5}  {'ours ' + str(len(hf_tok)//1024) + 'k':>9}" + (f"  {args.compare:>20}" if ref else ""))
    for lang in sorted(by_lang, key=lambda l: (l in MINORITY_LANGS, l)):
        texts = by_lang[lang]
        line = f"  {lang:<6} {len(texts):>5}  {chars_per_token(ours, texts):>9.2f}"
        if ref:
            line += f"  {chars_per_token(ref, texts):>20.2f}"
        print(line)


if __name__ == "__main__":
    main()
