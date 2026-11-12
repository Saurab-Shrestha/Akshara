# 02 · Tokenizer

Training our own byte-level BPE tokenizer for Devanagari + English (v2).

---

## Why v1 borrowed one, and why v2 trains its own

v1 used the **Qwen3.5-0.8B** tokenizer (248k tokens). The reasoning made sense
at the time: a huge multilingual tokenizer had already solved coverage. Then
we measured what it cost a ~110M-param model:

| | Qwen3.5 248k | **akshara-bpe 32k** |
|---|---|---|
| Devanagari chars/token (held-out) | 1.46 | **3.52** |
| Latin chars/token | 4.22 | 4.05 |
| Embedding params (dim 1024) | 254M | **34M** |
| Output-head FLOPs/token (fwd+bwd) | ~1.5 GFLOP | ~0.2 GFLOP |

Qwen's vocabulary is spread across ~100 languages. Only a small slice of it
is Devanagari, so Nepali is split into short pieces and every sentence costs
2.4× more tokens. Meanwhile the 248k-row embedding and output head took most
of the parameters and most of the compute. A tokenizer trained on *our*
languages fixes both, and building it is a lesson in its own right.

(Numbers measured with a tokenizer trained on a 100 MB sample of the v1 corpus.
Re-check them on the full v2 corpus.)

---

## Design decisions (`scripts/train_tokenizer.py`)

**1. Byte-level BPE.** Start from the 256 byte values and repeatedly merge the
most frequent adjacent pair. Any string is encodable, so there is no `<unk>`.
A Devanagari character is 3 UTF-8 bytes, and the frequent ones merge into
single tokens quickly.

**2. NFC normalisation.** Some Devanagari glyphs have two encodings: `क़` can
be U+0958 or `क` + nukta `़`. NFC maps both to one canonical form, so the
model doesn't have to learn the same glyph twice.

**3. A pre-tokenizer that respects Devanagari.** BPE never merges across
pre-tokenizer boundaries. GPT-2/Qwen split with ` ?\p{L}+`, which means
"letters". Devanagari vowel signs (matras), virama `्` and anusvara are
`\p{M}` (*marks*), so that regex tears `नेपाल` into `न` `े` `पाल` before BPE
ever sees it. Our pattern:

```
 ?[\p{L}\p{M}\x{200C}\x{200D}]+   words incl. marks + ZWNJ/ZWJ (र्‍ half-forms)
| ?\p{N}{1,3}                     numbers in groups of ≤3 (LLaMA-3 style)
| ?[^\s\p{L}\p{M}\p{N}]+          punctuation runs (। ॥ . , …)
|\s+(?!\S)|\s+                    whitespace
```

**4. Special tokens reserved now.**

| Token | id | Used in |
|---|---|---|
| `<|bos|>` / `<|eos|>` / `<|pad|>` | 0 / 1 / 2 | pretraining: every doc is `<|bos|> … <|eos|>` |
| `<|image|>` | 3 | stage 4–5: placeholder replaced by visual tokens |
| `<|im_start|>` / `<|im_end|>` | 4 / 5 | stage 5: chat turns |
| `<|reserved_0..9|>` | 6–15 | future use, no embedding resize needed |

Adding a token after pretraining means resizing the embedding and the new
row starts untrained. Reserving rows now costs nothing.

**5. Vocab ≤ 65,535.** Token ids then fit in `uint16`, which halves the
packed corpus on disk (`scripts/tokenize_corpus.py`).

---

## Train it

```bash
PYTHONPATH=. python scripts/train_tokenizer.py \
    --input data/corpus_v5/train.jsonl --vocab_size 32768 \
    --out_dir tokenizer/akshara-bpe --compare Qwen/Qwen3.5-0.8B
```

- Trains on a uniform random ~2 GB sample. The corpus is written source by
  source, so taking the first N lines would give an all-Nepali tokenizer.
- Asserts `decode(encode(x)) == NFC(x)` on 500 held-out docs.
- Prints chars/token per script. If Latin compression lags, try
  `--vocab_size 49152` or `65535` and compare.
- Commit `tokenizer/akshara-bpe/`. Every checkpoint depends on these exact ids.

Sample output (100 MB training sample):

```
['नेपाल', ' राम्रो', ' छ', '।', ' Hell', 'o', ' 202', '6']
```

Whole words come out as single tokens and the danda is its own token. `Hello`
is split only because the small sample had little English, so check it again
on the full corpus.

---

## Using it

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("tokenizer/akshara-bpe")
ids = tok.encode("नेपाल एक सुन्दर देश हो।")          # no BOS/EOS added
tok.decode(ids)
```

Scripts read the tokenizer path from the config (`tokenizer` field in
`configs/base.json`) or from a checkpoint's saved config. v1 checkpoints have
no such field and fall back to `Qwen/Qwen3.5-0.8B`.
`len(tokenizer)` must be ≤ `vocab_size`, and `pretrain.py` asserts it.

---

## Files

- [`scripts/train_tokenizer.py`](../scripts/train_tokenizer.py): train, round-trip check, compression report
- [`scripts/tokenize_corpus.py`](../scripts/tokenize_corpus.py): JSONL → packed uint16 `.bin`
- [`tokenizer/verify.py`](../tokenizer/verify.py): v1 Qwen Devanagari checks (historical)

---

**Next:** [03 · Building Blocks](03_building_blocks.md)
