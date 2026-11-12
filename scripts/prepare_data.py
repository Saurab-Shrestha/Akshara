"""
Data preparation for Akshara — streaming, token-budgeted corpus (v2).

WHY v2
------
v1 collected ~965k docs into a Python list (~80–100M tokens) and the
pretrain run cycled over it ~5×. v2 targets 3B+ *unique* tokens, so:
    - budgets are in characters (≈ tokens × 3.5 with our BPE), not docs —
      a Wikipedia article and a forum post are not the same "1 doc"
    - each doc is streamed straight to disk; nothing is held in RAM
    - the val split is a deterministic hash of the text (1 in 200), so it
      spans every source even though sources are written one after another

SOURCES (share of total characters)
-----------------------------------
    FineWeb-2   ne 30% · hi 25% · mr 4% · sa 1%   (length-filtered)
    FineWeb-Edu en 30%                             (EduScore ≥ 3)
    FineMath    3%     (finemath-4plus: digits, equations, reasoning text)
    Wikipedia   ne 3% · hi 2% · en 2%              (curated)

English is 30%+ (up from v1's 20%) because the MLLM instruction data
(LLaVA, Cauldron, Docmatix) is mostly English. If a source runs dry before
its budget (FineWeb-2 Nepali may), the script says so and moves on.

USAGE
-----
    # ~12B chars ≈ 3.4B tokens (~30 GB of JSONL on disk)
    PYTHONPATH=. python scripts/prepare_data.py --out_dir data/corpus_v5

    # tiny smoke corpus
    PYTHONPATH=. python scripts/prepare_data.py --total_chars 2e6 --out_dir /tmp/corpus_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib

QUALITY_THRESHOLD = 3

# (label, loader, loader-arg, share of total chars)
MIX = [
    ("fineweb2/ne",  "fineweb2", "npi_Deva", 0.30),
    ("fineweb2/hi",  "fineweb2", "hin_Deva", 0.25),
    ("fineweb2/mr",  "fineweb2", "mar_Deva", 0.04),
    ("fineweb2/sa",  "fineweb2", "san_Deva", 0.01),
    ("fineweb-edu",  "edu",      None,       0.30),
    ("finemath",     "math",     None,       0.03),
    ("wiki/ne",      "wiki",     "20231101.ne", 0.03),
    ("wiki/hi",      "wiki",     "20231101.hi", 0.02),
    ("wiki/en",      "wiki",     "20231101.en", 0.02),
]
assert abs(sum(m[3] for m in MIX) - 1.0) < 1e-9


def stream_texts(kind: str, arg: str | None, quality: int):
    """Yield cleaned document texts (≥50 chars) from one HF source, streaming."""
    from datasets import load_dataset

    if kind == "fineweb2":
        ds = load_dataset("HuggingFaceFW/fineweb-2", arg, split="train", streaming=True)
    elif kind == "edu":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-100BT", split="train", streaming=True)
    elif kind == "math":
        ds = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", streaming=True)
    elif kind == "wiki":
        ds = load_dataset("wikimedia/wikipedia", arg, split="train", streaming=True)
    else:
        raise ValueError(kind)

    it = iter(ds)
    while True:
        try:
            sample = next(it)
        except StopIteration:
            return
        except Exception as e:
            # FineWeb-2 shards occasionally have schema drift; skip the bad row
            if "Cast" in type(e).__name__ or "Schema" in type(e).__name__:
                continue
            raise
        if kind == "edu" and (sample.get("score") or 0) < quality:
            continue
        text = (sample.get("text") or "").strip()
        if len(text) >= 50:
            yield text


def is_val(text: str) -> bool:
    return zlib.crc32(text.encode("utf-8")) % 200 == 0


def prepare_corpus(total_chars: float, out_dir: str, quality: int):
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    with open(os.path.join(out_dir, "train.jsonl"), "w", encoding="utf-8") as ftr, \
         open(os.path.join(out_dir, "val.jsonl"), "w", encoding="utf-8") as fva:
        for label, kind, arg, share in MIX:
            budget = int(total_chars * share)
            got, docs, t0 = 0, 0, time.time()
            print(f"[{label}] target {budget/1e6:,.0f}M chars")
            try:
                for text in stream_texts(kind, arg, quality):
                    (fva if is_val(text) else ftr).write(
                        json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    got += len(text)
                    docs += 1
                    if docs % 50_000 == 0:
                        print(f"    {docs:,} docs | {got/1e6:,.0f}M chars | {time.time()-t0:.0f}s")
                    if got >= budget:
                        break
            except Exception as e:
                print(f"    [warn] {label} failed after {docs:,} docs: {e}")
            if got < budget:
                print(f"    [warn] {label} exhausted at {got/1e6:,.0f}M / {budget/1e6:,.0f}M chars")
            stats[label] = (docs, got)

    total = sum(c for _, c in stats.values())
    print(f"\n[data] {total/1e9:.2f}B chars (≈{total/3.5/1e9:.2f}B tokens) → {out_dir}/")
    for label, (docs, chars) in stats.items():
        print(f"  {label:<14} {docs:>10,} docs  {chars/1e6:>9,.0f}M chars  {chars/max(1,total):6.1%}")


def main():
    ap = argparse.ArgumentParser(description="Akshara corpus preparation (streaming, char-budgeted)")
    ap.add_argument("--total_chars", type=float, default=12e9, help="total character budget (default 12e9 ≈ 3.4B tokens)")
    ap.add_argument("--out_dir", default="data/corpus_v5")
    ap.add_argument("--quality", type=int, default=QUALITY_THRESHOLD, help="FineWeb-Edu min EduScore")
    args = ap.parse_args()
    prepare_corpus(args.total_chars, args.out_dir, args.quality)


if __name__ == "__main__":
    main()
