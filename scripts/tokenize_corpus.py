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
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")  # Rust batch encode uses all cores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/akshara-bpe")
    ap.add_argument("--out", default=None, help="default: input with .bin extension")
    ap.add_argument("--batch", type=int, default=2000)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    assert len(tok) <= 65535, f"vocab {len(tok)} does not fit uint16"
    bos, eos = tok.bos_token_id, tok.eos_token_id
    out_path = args.out or os.path.splitext(args.input)[0] + ".bin"

    n_tokens, n_docs, t0 = 0, 0, time.time()

    def flush(texts, out):
        nonlocal n_tokens, n_docs
        ids = tok(texts, add_special_tokens=False)["input_ids"]
        flat = np.fromiter(
            (t for doc in ids for t in (bos, *doc, eos)), dtype=np.uint16,
            count=sum(len(d) + 2 for d in ids),
        )
        flat.tofile(out)
        n_tokens += len(flat)
        n_docs += len(texts)

    with open(args.input, encoding="utf-8") as f, open(out_path, "wb") as out:
        buf = []
        for line in f:
            buf.append(json.loads(line)["text"])
            if len(buf) == args.batch:
                flush(buf, out)
                buf = []
                if n_docs % (args.batch * 50) == 0:
                    print(f"  {n_docs:,} docs | {n_tokens/1e6:,.0f}M tokens | {time.time()-t0:.0f}s")
        if buf:
            flush(buf, out)

    print(f"[tokenize] {n_docs:,} docs → {n_tokens:,} tokens → {out_path} "
          f"({os.path.getsize(out_path)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
