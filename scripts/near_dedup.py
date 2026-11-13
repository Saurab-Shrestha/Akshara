"""
Near-duplicate removal with MinHash + LSH (Broder 1997; settings from FineWeb).

Exact dedup (prepare_data.doc_key) misses the same article with a different
header, footer or date. Here two documents count as near-duplicates when the
Jaccard similarity of their word 5-gram sets is above ~0.75:

  1. shingles   every run of 5 whitespace-separated words, hashed to 32 bits
                (whitespace, not \\w: Python's \\w breaks Devanagari words at
                every vowel sign)
  2. MinHash    112 random hash functions; keep each one's minimum over the
                shingles. P(two docs share a minimum) = their Jaccard similarity.
  3. LSH        cut the 112 minima into 14 bands of 8. Docs equal on all 8
                values of any band are duplicates (threshold ≈ (1/14)^(1/8) ≈ 0.72).
  4. clusters   union-find; each cluster keeps its earliest document, so the
                source order of prepare_data (FineWeb-2 first) decides who stays.

Only documents whose "lang" is in --langs are compared; the rest pass through.

    PYTHONPATH=. python scripts/near_dedup.py --input data/corpus_v6/train.jsonl \\
        --drop_list data/corpus_v6/train.neardup.npy --langs npi
    PYTHONPATH=. python scripts/tokenize_corpus.py --input data/corpus_v6/train.jsonl \\
        --skip_lines data/corpus_v6/train.neardup.npy

The drop list (line numbers) avoids writing a second 60 GB JSONL; --out writes
a filtered copy instead.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from multiprocessing import Pool

import numpy as np

NGRAM, NUM_PERM, BANDS, ROWS = 5, 112, 14, 8
MAX_SHINGLES = 2000          # ponytail: long docs sampled evenly to 2000 shingles; raise if recall matters
_P = np.uint64(4_294_967_311)  # prime just above 2^32
_rng = np.random.default_rng(42)
_A = _rng.integers(1, 2**32 - 1, NUM_PERM, dtype=np.uint64)   # a·h + b < 2^64: no overflow
_B = _rng.integers(0, 2**32 - 1, NUM_PERM, dtype=np.uint64)


def signature(text: str) -> np.ndarray | None:
    """(NUM_PERM,) uint32 MinHash signature, or None if the doc has < NGRAM words."""
    words = np.fromiter((zlib.crc32(w.encode()) for w in text.split()), dtype=np.uint64)
    if len(words) < NGRAM:
        return None
    h = np.zeros(len(words) - NGRAM + 1, dtype=np.uint64)
    for i in range(NGRAM):                      # polynomial hash of the 5-gram (wraps mod 2^64)
        h = h * np.uint64(1_000_003) + words[i:len(words) - NGRAM + 1 + i]
    h = np.unique(h & np.uint64(0xFFFFFFFF))
    if len(h) > MAX_SHINGLES:
        h = h[np.linspace(0, len(h) - 1, MAX_SHINGLES).astype(np.int64)]
    mins = ((h[None, :] * _A[:, None] + _B[:, None]) % _P).min(axis=1)
    return np.minimum(mins, 2**32 - 1).astype(np.uint32)


def _batch(texts):
    return [signature(t) for t in texts]


def find_duplicates(sigs: np.ndarray) -> np.ndarray:
    """sigs: (N, NUM_PERM) uint32. Returns a bool mask of docs to drop."""
    n = len(sigs)
    parent = np.arange(n)

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:              # path compression
            parent[i], i = root, parent[i]
        return root

    for b in range(BANDS):
        band = np.ascontiguousarray(sigs[:, b * ROWS:(b + 1) * ROWS]).view(np.dtype((np.void, 4 * ROWS)))[:, 0]
        order = np.argsort(band, kind="stable")
        sorted_band = band[order]
        starts = np.flatnonzero(np.concatenate(([True], sorted_band[1:] != sorted_band[:-1])))
        sizes = np.diff(np.concatenate((starts, [n])))
        for s, size in zip(starts[sizes > 1], sizes[sizes > 1]):
            members = order[s:s + size]
            r0 = find(int(members.min()))
            for m in members:
                rm = find(int(m))
                if rm != r0:
                    lo, hi = min(r0, rm), max(r0, rm)  # the earliest doc stays the root
                    parent[hi] = lo
                    r0 = lo
    return np.array([find(i) != i for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--drop_list", help="save the line numbers to drop (.npy) for tokenize_corpus --skip_lines")
    ap.add_argument("--out", help="write a filtered copy of the JSONL")
    ap.add_argument("--langs", default="npi", help="comma-separated lang tags to dedup")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    langs = set(args.langs.split(","))

    # pass 1: signatures for the selected docs (line numbers remembered)
    t0, lines, texts, sigs = time.time(), [], [], []
    with open(args.input, encoding="utf-8") as f, Pool(args.workers) as pool:
        pending = []
        for i, line in enumerate(f):
            row = json.loads(line)
            if row.get("lang") in langs:
                lines.append(i)
                texts.append(row["text"])
            if len(texts) == 4000:
                pending.append(pool.apply_async(_batch, (texts,)))
                texts = []
        if texts:
            pending.append(pool.apply_async(_batch, (texts,)))
        for p in pending:
            sigs.extend(p.get())
    keep_idx = [k for k, s in enumerate(sigs) if s is not None]   # < 5 words: never a duplicate
    mat = np.stack([sigs[k] for k in keep_idx]) if keep_idx else np.zeros((0, NUM_PERM), np.uint32)
    print(f"[near_dedup] {len(lines):,} {args.langs} docs signed in {time.time() - t0:.0f}s")

    # pass 2: LSH + union-find
    drop_rows = {lines[keep_idx[j]] for j in np.flatnonzero(find_duplicates(mat))}
    print(f"[near_dedup] {len(drop_rows):,} near-duplicates ({len(drop_rows) / max(1, len(lines)):.1%} "
          f"of {args.langs}) in {time.time() - t0:.0f}s")

    if args.drop_list:
        np.save(args.drop_list, np.array(sorted(drop_rows), dtype=np.int64))
        print(f"[near_dedup] drop list → {args.drop_list}")
    if args.out:
        with open(args.input, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
            out.writelines(line for i, line in enumerate(f) if i not in drop_rows)
        print(f"[near_dedup] → {args.out}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--selftest"]:
        base = " ".join(f"शब्द{i}" for i in range(300))
        near = base.replace("शब्द7 ", "अर्को ") + " थप वाक्य"        # small edit: ~0.97 Jaccard
        other = " ".join(f"word{i}" for i in range(300))
        half = " ".join(base.split()[:150] + other.split()[:150])  # ~0.33 Jaccard
        sigs = np.stack([signature(t) for t in [base, other, near, half]])
        assert find_duplicates(sigs).tolist() == [False, False, True, False]
        assert signature("too short") is None
        print("near_dedup selftest passed")
    else:
        main()
