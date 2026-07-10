# Tokenization — Splitting Text into Numbers

Before a model can read text, it must convert characters into numbers. That
process is tokenization.

---

## Why not just use characters?

You could give the model one number per character. But Nepali has ~100 distinct
Devanagari characters, plus hundreds of conjunct forms and matras. A
character-level model would need very long sequences to encode even a short
sentence — and transformers slow down quadratically with sequence length.

On the other extreme you could have one number per word — but vocabulary
explodes (millions of Nepali words), and unknown words break everything.

**BPE (Byte Pair Encoding)** finds the sweet spot: frequent subword sequences
get their own token; rare sequences are split into smaller pieces.

---

## BPE in one paragraph

Start with individual bytes. Find the most frequent pair (e.g. `त` + `ा` →
`ता`). Merge it into a single token. Repeat 50,000–250,000 times. The result:
common words get a single token, common morphemes get a single token, and
truly rare character sequences fall back to individual bytes. The vocabulary
size is a hyperparameter set before training.

---

## The Devanagari challenge

Devanagari has three properties that trip up naive tokenizers:

| Feature | What it is | Why it matters |
|---|---|---|
| **Matras** | Vowel signs that attach to consonants | `क` + `ि` → `कि` — should merge |
| **Conjuncts** | Two consonants joined by virama | `क्` + `ष` → `क्ष` — should merge |
| **Shirorekha** | The horizontal bar above characters | Purely visual; Unicode has it implicit |

A tokenizer trained mostly on English will split Devanagari into individual
bytes (fallback), producing tokens like `\xe0\xa4\x95` instead of `क`. This
inflates sequence length 3× and destroys semantic structure.

---

## Why Qwen3.5-0.8B tokenizer

We evaluated four options:

| Tokenizer | Vocab | Devanagari | License |
|---|---|---|---|
| BPE trained from scratch | custom | mediocre (small corpus) | n/a |
| Llama 3 | 128k | byte fallback on Nepali | llama3 |
| GPT-4o tiktoken | 100k | partial | proprietary |
| **Qwen3.5-0.8B** | **248k** | **native, no fallback** | **Apache-2.0** |

Qwen3.5 was trained on a massive multilingual corpus including Hindi and
Nepali. Its 248k vocabulary is large enough to have genuine Devanagari tokens
for common conjuncts.

**Verification result** (`tokenizer/verify.py`):
```
नेपाल → ['ने', 'पाल']        ← 2 tokens (not 15 bytes)
राम्रो → ['राम', '्रो']      ← 2 tokens
```
No `\x` byte tokens appear. Average: **3.0 tokens per Nepali word** — excellent
(English averages 1.3 tokens/word with this tokenizer).

---

## What a token ID looks like

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

ids = tok.encode("नमस्ते")
# → [3610, 248032, 5673, ...]  (actual Devanagari subword IDs)

tok.decode(ids)
# → 'नमस्ते'
```

The model never sees the string `"नमस्ते"` — it sees `[3610, 248032, 5673]`.
The embedding table converts each ID into a 768-dimensional vector.

---

## Vocab size consequence

248,044 tokens × 768 dimensions = **190M parameters** just in the embedding
table. That's the dominant cost of our model. We manage it with weight tying:
the embedding matrix is shared with the output projection (LM head), so we
don't pay for it twice.

---

**Next:** [Building Blocks](../03_building_blocks.md) — RMSNorm, RoPE, SwiGLU
