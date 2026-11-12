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

LANGUAGES OF NEPAL (on top of the budget)
-----------------------------------------
Maithili, Newari, Bhojpuri, Awadhi, Tharu, Tamang, Sherpa, Doteli, Sunwar,
Limbu (Devanagari + Limbu script), Santali (Ol Chiki): FineWeb-2 + their
Wikipedias. Together they're ~0.1 GB, too small for a % budget, so ALL of it
is taken and repeated 4×. Every row carries a "lang" tag so the tokenizer
can oversample these languages and report per-language compression.

Hand-collected sources (Rai, Magar, Limbu, … from Bible translations, OPUS,
Leipzig, MADLAD/HPLT, …) go in data/extra_langs/*.jsonl as
{"text", "lang", "source"} rows. They join the same pool, exact-deduped
against everything above. Bible-sourced text is kept 1× instead of 4×.

USAGE
-----
    # ~12B chars ≈ 3.4B tokens (~30 GB of JSONL on disk)
    PYTHONPATH=. python scripts/prepare_data.py --out_dir data/corpus_v5

    # tiny smoke corpus
    PYTHONPATH=. python scripts/prepare_data.py --total_chars 2e6 --out_dir /tmp/corpus_smoke
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import time
import zlib

QUALITY_THRESHOLD = 3

# (label, loader, loader-arg, share of total chars, lang tag)
MIX = [
    ("fineweb2/ne",  "fineweb2", "npi_Deva", 0.30, "npi"),
    ("fineweb2/hi",  "fineweb2", "hin_Deva", 0.25, "hin"),
    ("fineweb2/mr",  "fineweb2", "mar_Deva", 0.04, "mar"),
    ("fineweb2/sa",  "fineweb2", "san_Deva", 0.01, "san"),
    ("fineweb-edu",  "edu",      None,       0.30, "eng"),
    ("finemath",     "math",     None,       0.03, "eng"),
    ("wiki/ne",      "wiki",     "20231101.ne", 0.03, "npi"),
    ("wiki/hi",      "wiki",     "20231101.hi", 0.02, "hin"),
    ("wiki/en",      "wiki",     "20231101.en", 0.02, "eng"),
]
assert abs(sum(m[3] for m in MIX) - 1.0) < 1e-9

# Languages of Nepal with little web text. Too small for a % budget: take ALL
# of it and repeat it MINORITY_REPEATS times (Muennighoff et al. 2023: up to
# ~4 epochs of repeated data is nearly as good as fresh data). Rai languages
# and Magar are not in FineWeb-2 — they need hand-collected sources.
# (label, loader, loader-arg, lang tag)
MINORITY = [
    ("fineweb2/mai",      "fineweb2", "mai_Deva", "mai"),   # Maithili
    ("fineweb2/new",      "fineweb2", "new_Deva", "new"),   # Newari / Nepal Bhasa
    ("fineweb2/bho",      "fineweb2", "bho_Deva", "bho"),   # Bhojpuri
    ("fineweb2/awa",      "fineweb2", "awa_Deva", "awa"),   # Awadhi
    ("fineweb2/thl",      "fineweb2", "thl_Deva", "thl"),   # Dangaura Tharu
    ("fineweb2/taj",      "fineweb2", "taj_Deva", "taj"),   # Eastern Tamang
    ("fineweb2/xsr",      "fineweb2", "xsr_Deva", "xsr"),   # Sherpa
    ("fineweb2/dty",      "fineweb2", "dty_Deva", "dty"),   # Doteli
    ("fineweb2/suz",      "fineweb2", "suz_Deva", "suz"),   # Sunwar
    ("fineweb2/lif-deva", "fineweb2", "lif_Deva", "lif"),   # Limbu (Devanagari)
    ("fineweb2/lif-limb", "fineweb2", "lif_Limb", "lif"),   # Limbu (Limbu script)
    ("fineweb2/sat-olck", "fineweb2", "sat_Olck", "sat"),   # Santali (Ol Chiki)
    ("wiki/new",          "wiki",     "20231101.new", "new"),
    ("wiki/mai",          "wiki",     "20231101.mai", "mai"),
    ("wiki/bh",           "wiki",     "20231101.bh",  "bho"),
    ("wiki/awa",          "wiki",     "20231101.awa", "awa"),
    ("wiki/dty",          "wiki",     "20231101.dty", "dty"),
    ("wiki/sat",          "wiki",     "20231101.sat", "sat"),
]
MINORITY_REPEATS = 4
BIBLE_SOURCES = ("bible", "door43")


def stream_texts(kind: str, arg: str | None, quality: int):
    """Yield cleaned document texts (≥50 chars) from one HF source, streaming."""
    from datasets import load_dataset

    if kind == "fineweb2":
        # Read the parquet files directly: some configs (e.g. new_Deva) carry
        # extra columns, and casting them to the repo's declared schema fails.
        ds = load_dataset("parquet", split="train", streaming=True,
                          data_files=f"hf://datasets/HuggingFaceFW/fineweb-2/data/{arg}/train/*.parquet")
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
            # A schema error ends the stream — never swallow it silently
            print(f"    [warn] {kind}/{arg} stream error, stopping source: {type(e).__name__}: {str(e)[:120]}")
            return
        if kind == "edu" and (sample.get("score") or 0) < quality:
            continue
        text = (sample.get("text") or "").strip()
        if len(text) >= 50:
            yield text


SCRIPT_RANGES = {"Deva": ("\u0900", "\u097f"), "Limb": ("\u1900", "\u194f"), "Olck": ("\u1c50", "\u1c7f")}


def expected_script(arg: str) -> str:
    """FineWeb-2 configs end in the script (mai_Deva, lif_Limb); Santali Wikipedia is Ol Chiki."""
    return arg.split("_")[-1] if "_" in arg else ("Olck" if arg.endswith(".sat") else "Deva")


def script_ok(text: str, script: str, min_ratio: float = 0.5) -> bool:
    """Keep a minority-language doc only if most of its letters are in the expected
    script — low-resource web subsets are often English or boilerplate
    (Kreutzer et al. 2022)."""
    lo, hi = SCRIPT_RANGES[script]
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(lo <= c <= hi for c in letters) / len(letters) >= min_ratio


def is_val(text: str) -> bool:
    return zlib.crc32(text.encode("utf-8")) % 200 == 0


def iter_extra(extra_dir: str):
    """Hand-collected minority-language JSONL ({"text","lang","source"}),
    one file per source+language — see data/extra_langs/MANIFEST_*.json."""
    if not extra_dir or not os.path.isdir(extra_dir):
        return
    licenses = {}
    for mf in glob.glob(os.path.join(extra_dir, "MANIFEST_*.json")):
        with open(mf, encoding="utf-8") as f:
            for e in json.load(f):
                if e.get("file"):
                    licenses[os.path.basename(e["file"])] = e.get("license") or ""
    for path in sorted(glob.glob(os.path.join(extra_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            first = json.loads(f.readline())
        lic = licenses.get(os.path.basename(path))
        def texts(path=path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    t = json.loads(line)["text"].strip()
                    if len(t) >= 20:
                        yield t
        yield path, first["lang"], first.get("source", ""), train_only(lic), texts()


def train_only(license: str | None) -> bool:
    """Train on it, never publish it: non-commercial / no-derivatives / not
    redistributable licenses, and anything without a known license."""
    if not license:
        return True
    lic = license.upper()
    return any(k in lic for k in ("NC", "NON-COMMERCIAL", "REDISTRIBUTABLE=FALSE", "UNCLEAR", "NONE STATED"))


def _row(text: str, lang: str, train_only: bool = False) -> str:
    row = {"text": text, "lang": lang}
    if train_only:
        row["train_only"] = True   # excluded from any public release of the corpus
    return json.dumps(row, ensure_ascii=False) + "\n"


def prepare_corpus(total_chars: float, out_dir: str, quality: int,
                   minority: bool = True, seed: int = 42,
                   extra_dir: str = "data/extra_langs"):
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    with open(os.path.join(out_dir, "train.jsonl"), "w", encoding="utf-8") as ftr, \
         open(os.path.join(out_dir, "val.jsonl"), "w", encoding="utf-8") as fva:
        for label, kind, arg, share, lang in MIX:
            budget = int(total_chars * share)
            got, docs, t0 = 0, 0, time.time()
            print(f"[{label}] target {budget/1e6:,.0f}M chars")
            try:
                for text in stream_texts(kind, arg, quality):
                    (fva if is_val(text) else ftr).write(_row(text, lang))
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

        # Minority languages: take everything, write val once, then write the
        # train docs MINORITY_REPEATS times — each pass reshuffled, so copies of
        # one doc sit ~a whole pass apart and never share a training window.
        if minority:
            pool: list[tuple[str, str, int, bool]] = []   # (text, lang, repeats, train_only)
            seen: set[bytes] = set()                 # exact dedupe across sources

            def add(label, texts, lang, repeats, nc=False):
                got = docs = 0
                try:
                    for text in texts:
                        h = hashlib.sha1(text.strip().encode("utf-8")).digest()
                        if h in seen:
                            continue
                        seen.add(h)
                        if is_val(text):
                            fva.write(_row(text, lang, nc))
                        else:
                            pool.append((text, lang, repeats, nc))
                        got += len(text)
                        docs += 1
                except Exception as e:
                    print(f"    [warn] {label} failed after {docs:,} docs: {e}")
                print(f"[{label}] took all: {docs:,} docs, {got/1e6:,.1f}M chars (×{repeats})"
                      + (" [train-only]" if nc else ""))
                stats[label] = (docs * repeats, got * repeats)

            for label, kind, arg, lang in MINORITY:
                script = expected_script(arg)
                texts = (t for t in stream_texts(kind, arg, quality) if script_ok(t, script))
                add(label, texts, lang, MINORITY_REPEATS)
            for path, lang, source, nc, texts in iter_extra(extra_dir):
                # Bible-derived text (eBible, door43 Open Bible Stories) is one
                # narrow register — don't let it dominate
                repeats = 1 if any(b in source.lower() for b in BIBLE_SOURCES) else MINORITY_REPEATS
                add(f"extra/{os.path.basename(path)}", texts, lang, repeats, nc)

            rng = random.Random(seed)
            for k in range(MINORITY_REPEATS):
                rng.shuffle(pool)
                ftr.writelines(_row(t, l, nc) for t, l, r, nc in pool if k < r)

    total = sum(c for _, c in stats.values())
    print(f"\n[data] {total/1e9:.2f}B chars (≈{total/3.5/1e9:.2f}B tokens) → {out_dir}/")
    for label, (docs, chars) in stats.items():
        print(f"  {label:<18} {docs:>10,} docs  {chars/1e6:>9,.1f}M chars  {chars/max(1,total):6.2%}")


def main():
    ap = argparse.ArgumentParser(description="Akshara corpus preparation (streaming, char-budgeted)")
    ap.add_argument("--total_chars", type=float, default=12e9, help="total character budget (default 12e9 ≈ 3.4B tokens)")
    ap.add_argument("--out_dir", default="data/corpus_v5")
    ap.add_argument("--quality", type=int, default=QUALITY_THRESHOLD, help="FineWeb-Edu min EduScore")
    ap.add_argument("--no_minority", action="store_true", help="skip the minority-language sources")
    ap.add_argument("--extra_dir", default="data/extra_langs",
                    help="hand-collected minority-language JSONL ({text, lang, source})")
    args = ap.parse_args()
    prepare_corpus(args.total_chars, args.out_dir, args.quality,
                   minority=not args.no_minority, extra_dir=args.extra_dir)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--selftest"]:
        assert script_ok("नेपाल भाषा", "Deva") and not script_ok("The Limbu Language Kit", "Limb")
        assert script_ok("ᤛᤡᤖᤡᤈᤨᤅ", "Limb") and expected_script("lif_Limb") == "Limb"
        assert expected_script("20231101.sat") == "Olck" and expected_script("20231101.new") == "Deva"
        assert train_only("CC BY-NC-ND 3.0") and train_only("CC BY-NC 4.0") and train_only(None)
        assert not train_only("CC BY-SA 4.0") and not train_only("CC0") and not train_only("CC BY 2.0 FR")
        print("prepare_data selftest passed")
    else:
        main()
