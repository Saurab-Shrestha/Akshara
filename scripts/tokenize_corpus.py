"""
Tokenize a JSONL corpus once into a flat uint16 token file (v2).

WHY
---
v1 tokenized on the fly and padded every document to max_seq_len, so short
docs were mostly padding, and the whole corpus had to fit in RAM. At 3B+
tokens neither works. Here every doc becomes  <|bos|> … <|eos|>, all docs
are concatenated into one array on disk, and training reads fixed windows
from a memmap (PackedTokenDataset): zero padding, ~0 RAM, 2 bytes/token.

USAGE
-----
    PYTHONPATH=. python scripts/tokenize_corpus.py \\
        --input data/corpus_v5/train.jsonl --tokenizer tokenizer/akshara-bpe
    # → data/corpus_v5/train.bin  (same for val.jsonl)

REPEATED N-GRAM FILTER
----------------------
A document is dropped if any span of 1–13 tokens repeats --max_repeats (32)
times back to back: scraped menus, "read more" loops, spam, mojibake, PDF
tables of 0.00. Such text is trivially predictable and causes loss spikes
(OLMo 2, via the Smol Training Playbook). Detection is vectorised per batch:
for period n, a run of ≥ (R-1)·n positions where ids[i] == ids[i-n] means some
n-gram repeats ≥ R times.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from collections import Counter

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")  # Rust batch encode uses all cores


MAX_NGRAM = 13


def repeated_docs(ids: list[list[int]], max_repeats: int) -> set[int]:
    """Indices of docs containing an n-gram (n ≤ 13) repeated ≥ max_repeats times in a row."""
    lens = np.fromiter((len(d) + 1 for d in ids), dtype=np.int64, count=len(ids))
    starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
    # -1 between docs never equals a token id, so no run crosses a doc boundary
    flat = np.fromiter((t for d in ids for t in (*d, -1)), dtype=np.int64, count=int(lens.sum()))
    bad: set[int] = set()
    for n in range(1, MAX_NGRAM + 1):
        eq = (flat[n:] == flat[:-n]) & (flat[n:] >= 0)
        edges = np.diff(np.concatenate(([0], eq.view(np.int8), [0])))
        run_s, run_e = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        hit = run_s[(run_e - run_s) >= (max_repeats - 1) * n]
        bad.update(np.searchsorted(starts, hit, side="right") - 1)
    return {int(i) for i in bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/akshara-bpe")
    ap.add_argument("--out", default=None, help="default: input with .bin extension")
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--max_repeats", type=int, default=32, help="repeated n-gram filter; 0 disables")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    assert len(tok) <= 65535, f"vocab {len(tok)} does not fit uint16"
    bos, eos = tok.bos_token_id, tok.eos_token_id
    out_path = args.out or os.path.splitext(args.input)[0] + ".bin"

    n_tokens, n_docs, t0 = 0, 0, time.time()
    seen, dropped = Counter(), Counter()   # per lang

    def flush(texts, langs, out):
        nonlocal n_tokens, n_docs
        ids = tok(texts, add_special_tokens=False)["input_ids"]
        seen.update(langs)
        if args.max_repeats:
            bad = repeated_docs(ids, args.max_repeats)
            dropped.update(langs[i] for i in bad)
            ids = [d for i, d in enumerate(ids) if i not in bad]
        flat = np.fromiter(
            (t for doc in ids for t in (bos, *doc, eos)), dtype=np.uint16,
            count=sum(len(d) + 2 for d in ids),
        )
        flat.tofile(out)
        n_tokens += len(flat)
        n_docs += len(ids)

    with open(args.input, encoding="utf-8") as f, open(out_path, "wb") as out:
        buf, langs = [], []
        for line in f:
            row = json.loads(line)
            buf.append(row["text"])
            langs.append(row.get("lang", "?"))
            if len(buf) == args.batch:
                flush(buf, langs, out)
                buf, langs = [], []
                if sum(seen.values()) % (args.batch * 50) == 0:
                    print(f"  {sum(seen.values()):,} docs | {n_tokens/1e6:,.0f}M tokens | {time.time()-t0:.0f}s")
        if buf:
            flush(buf, langs, out)

    if args.max_repeats:
        tot = sum(dropped.values())
        print(f"[tokenize] repeated n-gram filter dropped {tot:,} / {sum(seen.values()):,} docs: "
              + ", ".join(f"{l} {dropped[l]:,} ({dropped[l]/seen[l]:.1%})" for l, _ in dropped.most_common()))

    print(f"[tokenize] {n_docs:,} docs → {n_tokens:,} tokens → {out_path} "
          f"({os.path.getsize(out_path)/1e9:.2f} GB)")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--selftest"]:
        clean = list(range(100, 400))                       # no repeats
        menu = [5, 6, 7] * 32 + list(range(500, 600))     # 3-gram ×32
        near = [5, 6, 7] * 31 + list(range(500, 600))     # 3-gram ×31: kept
        run1 = [9] * 32                                    # unigram ×32
        assert repeated_docs([clean, menu, near, run1], 32) == {1, 3}
        assert repeated_docs([[1, 2] * 20, [1, 2] * 20], 32) == set()  # no run across docs
        print("tokenize_corpus selftest passed")
    else:
        main()
