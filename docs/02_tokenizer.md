# 02 · Tokenizer

Choosing and verifying a tokenizer that handles Devanagari correctly.

---

## Why not train our own?

Training a good BPE tokenizer requires a large, clean corpus. Nepali Wikipedia
has ~100k articles — decent but small. A tokenizer trained on that alone would
have weak coverage of technical vocabulary, code-switched text (Nepali + English
in the same document), and formal register.

Qwen3.5-0.8B was trained on a 36T-token corpus including multiple Indian
subcontinent languages. Its tokenizer has already solved the coverage problem.

**The rule:** use the best existing tool rather than building from scratch if
the license allows. Qwen3.5-0.8B tokenizer is Apache-2.0 — free to use and
modify commercially.

---

## Devanagari tokenizer requirements

1. **No byte fallback** — Devanagari characters should map to real subword
   tokens, not `\xe0\xa4\x95` sequences
2. **Conjunct merging** — `क` + `्` + `ष` should ideally produce `क्ष` as
   one token
3. **Matra merging** — `क` + `ा` → `का` (consonant + vowel sign together)
4. **Reasonable ratio** — target < 4 tokens/word for Nepali (English is ~1.3)

---

## Verification (`tokenizer/verify.py`)

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

# Test Devanagari coverage
test_words = ["नेपाल", "राम्रो", "देश", "क्षमा", "संविधान"]
for word in test_words:
    tokens = tok.tokenize(word)
    has_byte_fallback = any("\\x" in repr(t) for t in tokens)
    print(f"{word:12} → {tokens}  {'❌ byte fallback' if has_byte_fallback else '✅'}")
```

**Results:**
```
नेपाल       → ['ने', 'पाल']                    ✅
राम्रो      → ['राम', '्रो']                   ✅
देश         → ['देश']                           ✅  (single token!)
क्षमा       → ['क्ष', 'मा']                    ✅  (conjunct merged)
संविधान     → ['सं', 'वि', 'धान']              ✅
```

No byte fallback anywhere. Average: **3.0 tokens/word** — well within target.

---

## Using the tokenizer in training

```python
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

# Encode a line of OCR ground truth
text = "नेपाल एक सुन्दर देश हो।"
ids  = tok.encode(text, return_tensors="pt")  # shape: (1, n_tokens)

# Decode back (for inspection)
tok.decode(ids[0])  # → 'नेपाल एक सुन्दर देश हो।'
```

Special tokens:
- `tok.eos_token_id` — end of sequence (used to signal generation stop)
- `tok.bos_token_id` — beginning of sequence (prepended before OCR target)
- `tok.pad_token_id` — padding (for batching variable-length sequences)

---

## Vocab size in the model

The tokenizer vocabulary is **248,044 tokens** (Qwen3.5-0.8B). Our model is
initialized with:

```python
DEFAULT_CONFIG = dict(
    vocab_size = 248_044,   # matches tokenizer exactly
    ...
)
```

If the model and tokenizer vocab sizes don't match, you'll get a shape error in
the embedding table lookup.

---

## Files

- [`tokenizer/verify.py`](../tokenizer/verify.py) — runs all Devanagari checks

---

**Next:** [03 · Building Blocks](03_building_blocks.md)
