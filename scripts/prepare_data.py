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
    Nepali extra  --nepali_share (default 15%, carved out of FineWeb-2 ne)

NEPALI SOURCES
--------------
scripts/collect_nepali.py downloads HPLT 2.0, IndicCorpV2, FinePDFs, GlotCC
and DCAD-2000 for Nepali into data/nepali/*.jsonl. They join MIX as ordinary
budgeted sources (not through extra_langs, which is held in RAM and repeated
up to 4x for minority languages). Their weights inside --nepali_share are in
NEPALI below; files that are absent are skipped and the rest renormalized.
Rationale: they are independent crawl/filter pipelines, so they add document
diversity that FineWeb-2 alone cannot.

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
# Optional 6th field = train_only (license doesn't allow republishing).
MIX = [
    ("fineweb2/ne",  "fineweb2", "npi_Deva", 0.30, "npi"),
    ("fineweb2/hi",  "fineweb2", "hin_Deva", 0.19, "hin"),   # bridge language for ne/mai/bho/awa
    ("fineweb2/mr",  "fineweb2", "mar_Deva", 0.03, "mar"),
    ("fineweb2/sa",  "fineweb2", "san_Deva", 0.01, "san"),
    ("fineweb-edu",  "edu",      None,       0.24, "eng"),
    ("code",         "code",     None,       0.07, "code"),  # code improves NL reasoning (Aryabumi et al. 2024)
    ("finemath",     "math",     None,       0.04, "eng"),
    ("cosmopedia",   "cosmo",    None,       0.03, "eng"),   # synthetic textbooks / step-by-step explanations
    # Parallel text (~1% total; each source is capped by how much exists).
    # Tags like "eng-npi" keep pairs out of per-language stats and tokenizer floors.
    # BPCC (ai4bharat): "subdir+subdir/lang". Human = seed + daily (CC-BY-4.0/CC0);
    # mined = NLLB/Samanantar/comparable (CC0, noisier, NLLB-mai is mostly Bible).
    ("parallel/bpcc-ne",  "bpcc", "bpcc-seed-latest+daily/npi_Deva", 0.003,  "eng-npi"),
    ("parallel/bpcc-mai", "bpcc", "bpcc-seed-latest+daily/mai_Deva", 0.0015, "eng-mai"),
    ("parallel/bpcc-sat", "bpcc", "bpcc-seed-latest+daily/sat_Olck", 0.0015, "eng-sat"),
    ("parallel/bpcc-mined-ne",  "bpcc", "nllb_filtered+samanantar_v2+comparable/npi_Deva", 0.0095, "eng-npi"),
    ("parallel/bpcc-mined-mai", "bpcc", "nllb_filtered/mai_Deva", 0.0005, "eng-mai"),
    ("parallel/opus-ne",  "opus", "en-ne",    0.004,  "eng-npi", True),  # OPUS-100, license unknown
    ("wiki/ne",      "wiki",     "20231101.ne", 0.03, "npi"),
    ("wiki/hi",      "wiki",     "20231101.hi", 0.02, "hin"),
    ("wiki/en",      "wiki",     "20231101.en", 0.02, "eng"),
]

# codeparrot-clean ships file contents + a per-file license; keep permissive only.
PERMISSIVE_CODE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "cc0-1.0", "mpl-2.0"}
PARALLEL_PAIRS_PER_DOC = 16
LANG_NAMES = {"ne": "नेपाली", "npi": "नेपाली", "mai": "मैथिली", "sat": "ᱥᱟᱱᱛᱟᱲᱤ"}


def parallel_docs(pairs, name_a: str, name_b: str):
    """Group (a, b) translation pairs into documents of PARALLEL_PAIRS_PER_DOC.
    Direction alternates per document so the model links concepts both ways."""
    buf, flip = [], False
    for a, b in pairs:
        a, b = (a or "").strip(), (b or "").strip()
        if len(a) < 10 or len(b) < 10:
            continue
        buf.append((a, b))
        if len(buf) == PARALLEL_PAIRS_PER_DOC:
            yield "\n\n".join(f"{name_b}: {y}\n{name_a}: {x}" if flip else f"{name_a}: {x}\n{name_b}: {y}"
                               for x, y in buf)
            buf, flip = [], not flip
assert abs(sum(m[3] for m in MIX) - 1.0) < 1e-9

# Nepali sources produced by scripts/collect_nepali.py. Weights are relative
# *inside* --nepali_share; the budget is taken out of fineweb2/ne so MIX still
# sums to 1.0. (label, loader, filename, weight, lang tag, train_only)
NEPALI = [
    ("nepali/hplt2",     "hfjsonl", "hf_hplt2_npi.jsonl",      0.36, "npi", False),
    ("nepali/indiccorp", "hfjsonl", "hf_indiccorp_npi.jsonl",  0.30, "npi", False),
    ("nepali/finepdfs",  "hfjsonl", "hf_finepdfs_npi.jsonl",   0.18, "npi", False),
    ("nepali/glotcc",    "hfjsonl", "hf_glotcc_npi.jsonl",     0.08, "npi", False),
    ("nepali/dcad2000",  "hfjsonl", "hf_dcad2000_npi.jsonl",   0.08, "npi", True),
]
DEFAULT_NEPALI_SHARE = 0.15

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
MINORITY_REPEATS = 4            # max repeats for the smallest languages
MINORITY_TARGET_CHARS = 200e6   # repeat a language until ~this many chars, capped above
BIBLE_SOURCES = ("bible", "door43")


def lang_repeats(per_lang_chars: dict[str, int]) -> dict[str, int]:
    """Repeat small languages up to MINORITY_REPEATS×, big ones (Maithili, Newari,
    Bhojpuri: 300M+ chars) not at all — otherwise 4× repeats would make the
    minority pool ~30% of the corpus."""
    return {l: max(1, min(MINORITY_REPEATS, round(MINORITY_TARGET_CHARS / max(1, c))))
            for l, c in per_lang_chars.items()}


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
    elif kind == "cosmo":
        ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    elif kind == "code":
        # SmolLM's python-edu / stack-edu hold only Software Heritage blob ids
        # (contents need AWS S3); codeparrot-clean includes the file text.
        for r in load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True):
            if (r.get("license") or "").lower() not in PERMISSIVE_CODE or r.get("autogenerated"):
                continue
            if (r.get("alpha_frac") or 0) < 0.25 or (r.get("line_max") or 0) > 1000:
                continue
            t = (r.get("content") or "").strip()
            if len(t) >= 200:
                yield t
        return
    elif kind == "opus":
        src, tgt = arg.split("-")
        ds = load_dataset("parquet", split="train", streaming=True,
                          data_files=f"hf://datasets/Helsinki-NLP/opus-100/{arg}/train-*.parquet")
        pairs = (((r.get("translation") or {}).get(src, ""), (r.get("translation") or {}).get(tgt, ""))
                 for r in ds)
        yield from parallel_docs(pairs, "English", LANG_NAMES[tgt])
        return
    elif kind == "bpcc":
        # arg "dir1+dir2/lang". Column order differs per subdir, so read the header.
        from huggingface_hub import hf_hub_download
        dirs, lang = arg.split("/")
        unq = lambda t: t[1:-1].replace('""', '"') if len(t) > 1 and t[0] == t[-1] == '"' else t
        def rows():
            for d in dirs.split("+"):
                path = hf_hub_download("ai4bharat/BPCC", f"{d}/{lang}.tsv", repo_type="dataset")
                with open(path, encoding="utf-8") as f:
                    head = f.readline().rstrip("\n").split("\t")
                    si, ti = head.index("src"), head.index("tgt")
                    for line in f:
                        cols = line.rstrip("\n").split("\t")
                        if len(cols) > max(si, ti):
                            yield unq(cols[si]), unq(cols[ti])
        yield from parallel_docs(rows(), "English", LANG_NAMES[lang.split("_")[0]])
        return
    elif kind == "hfjsonl":
        # Local corpus from scripts/collect_nepali.py: one {"text",...} per line.
        with open(arg, encoding="utf-8") as f:
            for line in f:
                t = (json.loads(line).get("text") or "").strip()
                if len(t) >= 50:
                    yield t
        return
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
    # "other" / "per-subsource" = aggregates whose parts carry their own licenses
    return any(k in lic for k in ("NC", "NON-COMMERCIAL", "REDISTRIBUTABLE=FALSE", "UNCLEAR",
                                  "NONE STATED", "OTHER", "PER-SUBSOURCE", "RESERVED",
                                  "NO PART", "PERMISSION", "UNKNOWN"))


def _row(text: str, lang: str, train_only: bool = False) -> str:
    row = {"text": text, "lang": lang}
    if train_only:
        row["train_only"] = True   # excluded from any public release of the corpus
    return json.dumps(row, ensure_ascii=False) + "\n"


def nepali_jobs(nepali_dir: str, share: float) -> list[tuple]:
    """Resolve the NEPALI registry to concrete (label, kind, path, share, lang,
    train_only) jobs. Missing files are skipped and the remaining weights are
    renormalized, so a partial download still produces a full 1.0 mix."""
    if share <= 0 or not os.path.isdir(nepali_dir):
        return []
    present = [n for n in NEPALI if os.path.exists(os.path.join(nepali_dir, n[2]))]
    if not present:
        return []
    w = sum(n[3] for n in present)
    jobs = [(label, kind, os.path.join(nepali_dir, fn), share * weight / w, lang, nc)
            for label, kind, fn, weight, lang, nc in present]
    print(f"[nepali] {len(present)}/{len(NEPALI)} source(s) on disk: "
          + ", ".join(f"{j[0].split('/')[-1]} {j[3]:.1%}" for j in jobs))
    return jobs


def prepare_corpus(total_chars: float, out_dir: str, quality: int,
                   minority: bool = True, seed: int = 42,
                   extra_dir: str = "data/extra_langs",
                   nepali_share: float = DEFAULT_NEPALI_SHARE,
                   nepali_dir: str = "data/nepali"):
    os.makedirs(out_dir, exist_ok=True)

    # Nepali extra sources are paid for out of fineweb2/ne, so MIX still sums
    # to 1.0 and the total character budget is unchanged.
    take = 0.0
    extra = nepali_jobs(nepali_dir, nepali_share)
    if extra:
        fw = next(m[3] for m in MIX if m[0] == "fineweb2/ne")
        take = min(nepali_share, fw)
        if take < nepali_share:
            print(f"[nepali] --nepali_share {nepali_share:.0%} > fineweb2/ne {fw:.0%}; "
                  f"using {take:.0%}")
            extra = nepali_jobs(nepali_dir, take)
    jobs = []
    for m in MIX:
        label, kind, arg, share, lang = m[:5]
        jobs.append((label, kind, arg, share - take if label == "fineweb2/ne" else share,
                     lang, m[5] if len(m) > 5 else False))
    jobs += extra
    assert abs(sum(j[3] for j in jobs) - 1.0) < 1e-9

    stats = {}
    with open(os.path.join(out_dir, "train.jsonl"), "w", encoding="utf-8") as ftr, \
         open(os.path.join(out_dir, "val.jsonl"), "w", encoding="utf-8") as fva:
        for label, kind, arg, share, lang, nc in jobs:
            budget = int(total_chars * share)
            got, docs, t0 = 0, 0, time.time()
            print(f"[{label}] target {budget/1e6:,.0f}M chars")
            try:
                for text in stream_texts(kind, arg, quality):
                    (fva if is_val(text) else ftr).write(_row(text, lang, nc))
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
        # train docs in reshuffled passes, so copies of one doc sit ~a whole
        # pass apart and never share a training window. Repeats are sized per
        # language (lang_repeats): small languages up to 4×, big ones 1×.
        # ponytail: whole minority pool held in RAM (~2B chars ≈ 5 GB); switch to
        # an on-disk shuffle if it outgrows the box.
        if minority:
            pool: list[tuple[str, str, bool, bool]] = []   # (text, lang, bible, train_only)
            seen: set[bytes] = set()                       # exact dedupe across sources
            sources: dict[str, tuple[str, int, int]] = {}  # label -> (lang, docs, chars)

            def add(label, texts, lang, bible=False, nc=False):
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
                            pool.append((text, lang, bible, nc))
                        got += len(text)
                        docs += 1
                except Exception as e:
                    print(f"    [warn] {label} failed after {docs:,} docs: {e}")
                print(f"[{label}] {lang}: {docs:,} docs, {got/1e6:,.1f}M chars"
                      + (" [bible 1×]" if bible else "") + (" [train-only]" if nc else ""))
                sources[label] = (lang, docs, got, bible)

            for label, kind, arg, lang in MINORITY:
                script = expected_script(arg)
                texts = (t for t in stream_texts(kind, arg, quality) if script_ok(t, script))
                add(label, texts, lang)
            for path, lang, source, nc, texts in iter_extra(extra_dir):
                # Bible-derived text (eBible, door43) is one narrow register: kept 1×
                bible = any(b in source.lower() for b in BIBLE_SOURCES)
                add(f"extra/{os.path.basename(path)}", texts, lang, bible, nc)

            per_lang: dict[str, int] = {}
            for lang, _, chars, _ in sources.values():
                per_lang[lang] = per_lang.get(lang, 0) + chars
            reps = lang_repeats(per_lang)
            print("[minority] repeats per language: " + ", ".join(
                f"{l} {per_lang[l]/1e6:.0f}M×{reps[l]}" for l in sorted(per_lang, key=lambda l: -per_lang[l])))
            for label, (lang, docs, chars, bible) in sources.items():
                r = 1 if bible else reps[lang]
                stats[label] = (docs * r, chars * r)

            rng = random.Random(seed)
            for k in range(MINORITY_REPEATS):
                rng.shuffle(pool)
                ftr.writelines(_row(t, l, nc) for t, l, b, nc in pool
                               if k < (1 if b else reps[l]))

    total = sum(c for _, c in stats.values())
    fmt = lambda n: f"{n/1e9:.2f}B" if n >= 1e9 else f"{n/1e6:,.1f}M"
    print(f"\n[data] {fmt(total)} chars (≈{total/3.5/1e9:.2f}B tokens) → {out_dir}/")
    for label, (docs, chars) in stats.items():
        print(f"  {label:<18} {docs:>10,} docs  {fmt(chars):>9} chars  {chars/max(1,total):6.2%}")


def main():
    ap = argparse.ArgumentParser(description="Akshara corpus preparation (streaming, char-budgeted)")
    ap.add_argument("--total_chars", type=float, default=12e9, help="total character budget (default 12e9 ≈ 3.4B tokens)")
    ap.add_argument("--out_dir", default="data/corpus_v5")
    ap.add_argument("--quality", type=int, default=QUALITY_THRESHOLD, help="FineWeb-Edu min EduScore")
    ap.add_argument("--no_minority", action="store_true", help="skip the minority-language sources")
    ap.add_argument("--extra_dir", default="data/extra_langs",
                    help="hand-collected minority-language JSONL ({text, lang, source})")
    ap.add_argument("--nepali_share", type=float, default=DEFAULT_NEPALI_SHARE,
                    help="share of total chars from data/nepali/ (carved out of fineweb2/ne); 0 disables")
    ap.add_argument("--nepali_dir", default="data/nepali",
                    help="output of scripts/collect_nepali.py")
    args = ap.parse_args()
    prepare_corpus(args.total_chars, args.out_dir, args.quality,
                   minority=not args.no_minority, extra_dir=args.extra_dir,
                   nepali_share=args.nepali_share, nepali_dir=args.nepali_dir)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--selftest"]:
        assert script_ok("नेपाल भाषा", "Deva") and not script_ok("The Limbu Language Kit", "Limb")
        assert script_ok("ᤛᤡᤖᤡᤈᤨᤅ", "Limb") and expected_script("lif_Limb") == "Limb"
        assert expected_script("20231101.sat") == "Olck" and expected_script("20231101.new") == "Deva"
        assert train_only("CC BY-NC-ND 3.0") and train_only("CC BY-NC 4.0") and train_only(None)
        assert not train_only("CC BY-SA 4.0") and not train_only("CC0") and not train_only("CC BY 2.0 FR")
        assert train_only("other (per-subsource licenses)") and train_only("odc-by (per-subsource licenses)")
        assert not train_only("odc-by") and not train_only("cc0-1.0 (CC ToU)")
        docs = list(parallel_docs([(f"english sentence {i}", f"नेपाली वाक्य {i}") for i in range(32)], "English", "नेपाली"))
        assert len(docs) == 2 and docs[0].startswith("English:") and docs[1].startswith("नेपाली:")
        assert train_only("All rights reserved") and train_only("no part may be reproduced")
        assert lang_repeats({"mai": 400e6, "sat": 130e6, "awa": 65e6, "lif": 3e6}) == {"mai": 1, "sat": 2, "awa": 3, "lif": 4}
        print("prepare_data selftest passed")
    else:
        main()
