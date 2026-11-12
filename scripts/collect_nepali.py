"""
Collect Nepali (npi) pretraining text from openly licensed corpora.

WHY THIS EXISTS
---------------
prepare_data.py takes Nepali from FineWeb-2 (30%) + Wikipedia (3%). Both are
good, but they are one pipeline with one set of filters. For a low-resource
language, *independent pipelines* matter more than more of the same text: a
document that survives FineWeb-2's filter and the same document that survives
HPLT's are not independent samples, but a PDF-derived text and a web-crawl text
usually are. These sources were crawled and filtered differently:

    key        license            Nepali volume      what it is
    hplt2      CC0-1.0            ~7.4 GB parquet    2023 web crawl, per-doc
                                                     robotstxt + lang prob
    indiccorp  CC0-1.0            ~14.8 GB text      AI4Bharat, one sentence
                                                     per line, news/web mixed
    finepdfs   ODC-BY             ~3.2 GB parquet    PDFs -> text (reports,
                                                     gov docs, books)
    glotcc     CC0-1.0            ~1.8 GB parquet    curated CommonCrawl
    dcad2000   other/per-source   ~2.2 GB parquet    aggregated + deduped;
                                                     train-only (not publishable)

WHY NOT extra_langs/
--------------------
data/extra_langs/ is for minority languages: prepare_data.py holds that whole
pool in RAM and repeats small languages up to 4x. Nepali is the main language
and these files are multi-GB, so they are written to data/nepali/ and read as
first-class MIX sources (see --nepali_share in prepare_data.py).

OUTPUT
------
    data/nepali/<prefix>_<key>_npi.jsonl   one JSON object per doc:
        {"text": ..., "lang": "npi", "source": ..., "url": ...}
    data/nepali/MANIFEST.json              license / volume / filter stats,
                                           merged across runs

Filter order (first match wins, all counted in the manifest):
    short    < --min_chars after normalization
    offscript  < 50% Devanagari letters
    junk     line-repetition or punctuation/digit swamp (nav, tables, cookies)
    dup      exact duplicate of an earlier doc (normalized, blake2b-64)

USAGE
-----
    ./venv/bin/python scripts/collect_nepali.py --list
    ./venv/bin/python scripts/collect_nepali.py                     # all sources
    ./venv/bin/python scripts/collect_nepali.py --sources hplt2,indiccorp
    ./venv/bin/python scripts/collect_nepali.py --sources hplt2 --max_chars 1e9
    ./venv/bin/python scripts/collect_nepali.py --sources glotcc --restart

Re-running a source appends and skips docs already on disk (--resume, default
on). --restart truncates that source's file first.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

UA = "akshara-corpus-builder/1.0 (+local research; contact: repository owner)"
HF = "https://huggingface.co"
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

# key -> how to fetch it. `dir` is the path inside the HF repo (parquet), or
# `config` for a load_dataset config, or `text_url` for a raw one-line-per-sentence
# file. `train_only` marks licenses that forbid redistribution of the text.
SOURCES = {
    "hplt2": dict(
        kind="parquet", repo="HPLT/HPLT2.0_cleaned", config="npi_Deva",
        license="cc0-1.0",
        notes="HPLT 2.0 cleaned. Filtered to robotstxt=allowed and filter=keep.",
    ),
    "indiccorp": dict(
        kind="lines", repo="ai4bharat/IndicCorpV2", path="data/ne.txt",
        license="cc0-1.0",
        notes="IndicCorpV2 (AI4Bharat), sentence-per-line. Lines are grouped "
              "into ~1.2k-char pseudo-documents so they aren't 1-sentence docs.",
    ),
    "finepdfs": dict(
        kind="parquet", repo="HuggingFaceFW/finepdfs", config="npi_Deva",
        license="odc-by",
        notes="FinePDFs: PDF -> text. Different register from web crawl "
              "(government reports, books, forms).",
    ),
    "glotcc": dict(
        kind="parquet", repo="cis-lmu/GlotCC-V1", config="npi-Deva",
        license="cc0-1.0",
        notes="GlotCC-V1, curated CommonCrawl subset.",
    ),
    "dcad2000": dict(
        kind="parquet", repo="openbmb/DCAD-2000", config="npi_Deva",
        license="other (per-subsource licenses)", train_only=True, min_chars=100,
        notes="DCAD-2000 aggregate; sub-sources carry their own licenses, so "
              "train_only=True keeps it out of any public corpus release.",
    ),
}

FILE_PREFIX = {"parquet": "hf", "lines": "hf"}


# ---------------------------------------------------------------- normalization

_CTRL = {c: None for c in range(32) if c not in (9, 10)}
_CTRL[127] = None


def normalize(text: str) -> str:
    """NFC, unify whitespace, drop control chars but keep newlines.

    Devanagari ZWNJ (200C) / ZWJ (200D) are preserved: they change conjunct
    rendering and are part of the orthography. Unicode composition matters
    because Nepali text mixes precomposed and decomposed matras, and dedupe
    should see them as the same document.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_CTRL).replace("\u00a0", " ").replace("\u200b", "")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    joined = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", joined)


def deva_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum("\u0900" <= c <= "\u097f" for c in letters) / len(letters)


def is_junk(text: str) -> bool:
    """Nav bars, cookie tables, price lists. Two cheap signals:
    - a handful of lines repeated over and over (menus, footers)
    - punctuation/digits crowding out words (tables, OCR noise, listings)
    """
    lines = [ln for ln in text.split("\n") if ln]
    if len(lines) >= 8:
        top = collections.Counter(lines).most_common(1)[0][1]
        if top >= 5 and top / len(lines) > 0.30:
            return True
    letters = sum(c.isalpha() for c in text)
    return letters < 0.40 * len(text)


class Filter:
    """Normalize -> length -> script -> junk -> exact dedupe, with counts."""

    def __init__(self, min_chars: int, min_deva: float, dedupe: bool, min_prob: float = 0.8):
        self.min_chars = min_chars
        self.min_deva = min_deva
        self.min_prob = min_prob
        self.dedupe = dedupe
        self.seen: set[int] = set()
        self.stats = collections.Counter()
        self.chars = 0

    def load_existing(self, path: str) -> int:
        """Seed the dedupe set from a file we are appending to."""
        n = 0
        if not self.dedupe or not os.path.exists(path):
            return 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    t = json.loads(line)["text"]
                except (json.JSONDecodeError, KeyError):
                    continue
                self.seen.add(self._h(t))
                n += 1
        return n

    @staticmethod
    def _h(text: str) -> int:
        return int.from_bytes(
            hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big"
        )

    def __call__(self, raw: str, lang_prob: float | None = None) -> str | None:
        self.stats["raw"] += 1
        if lang_prob is not None and lang_prob < self.min_prob:
            self.stats["offlang"] += 1
            return None
        text = normalize(raw)
        if len(text) < self.min_chars:
            self.stats["short"] += 1
            return None
        if deva_ratio(text) < self.min_deva:
            self.stats["offscript"] += 1
            return None
        if is_junk(text):
            self.stats["junk"] += 1
            return None
        if self.dedupe:
            h = self._h(text)
            if h in self.seen:
                self.stats["dup"] += 1
                return None
            self.seen.add(h)
        self.stats["kept"] += 1
        self.chars += len(text)
        return text


# ------------------------------------------------------------------- fetching

def _get(url: str, timeout: int = 120):
    headers = {"User-Agent": UA}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


def hf_parquet_urls(repo: str, config: str) -> list[str]:
    """Shard URLs for one config of a HF dataset.

    Prefer the datasets-server index: finepdfs, GlotCC and DCAD only publish
    parquet on the auto-converted `refs/convert/parquet` branch, which is not
    in the repo tree. Fall back to shards committed to main (HPLT).
    """
    index = f"https://datasets-server.huggingface.co/parquet?dataset={urllib.parse.quote(repo)}"
    try:
        with _get(index) as r:
            files = json.load(r).get("parquet_files", [])
        urls = sorted(
            f["url"] for f in files
            if f.get("config") == config and f.get("split") == "train"
        )
        if urls:
            return urls
    except Exception:
        pass

    try:
        with _get(f"{HF}/api/datasets/{repo}/tree/main/{config}") as r:
            entries = json.load(r)
        urls = sorted(
            f"{HF}/datasets/{repo}/resolve/main/{e['path']}"
            for e in entries if e.get("path", "").endswith(".parquet")
        )
        if urls:
            return urls
    except Exception:
        pass

    sys.exit(f"{repo}: no parquet shards found for config {config!r} (check --list)")


def iter_parquet(urls: list[str], cache_dir: str):
    """Stream row dicts from remote parquet shards.

    Each shard is downloaded to a local cache file and read with pyarrow.
    HTTP range reads look appealing but are far slower in practice: the
    auto-converted parquet branch serves ~0.6 s/doc through fsspec, versus
    ~30 s per 200 MB shard as one sequential stream. Only the columns we use
    are decoded, and the shard file is deleted after reading.

    pyarrow (not the `datasets` streaming Dataset) because the latter adds
    per-row Python overhead — same data, ~13x slower.
    """
    import pyarrow.parquet as pq
    import shutil

    wanted = ("text", "content", "url", "u", "warc-target-uri", "robotstxt",
              "filter", "prob", "identification-prob")
    os.makedirs(cache_dir, exist_ok=True)
    for i, url in enumerate(urls, 1):
        # Unique per URL: concurrent collectors (or two sources with the same
        # shard names) must not delete each other's download.
        tag = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
        local = os.path.join(cache_dir, f"{tag}-{i:05d}.parquet")
        try:
            t0 = time.time()
            with _get(url, timeout=1800) as r, open(local + ".part", "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            os.replace(local + ".part", local)
            pf = pq.ParquetFile(local)
            cols = [c for c in wanted if c in pf.schema_arrow.names]
            print(f"    shard {i}/{len(urls)}: {pf.metadata.num_rows:,} rows "
                  f"(downloaded in {time.time()-t0:.0f}s)", flush=True)
            for batch in pf.iter_batches(batch_size=1000, columns=cols):
                yield from batch.to_pylist()
        except Exception as e:
            print(f"    [warn] shard {i}/{len(urls)} failed: {type(e).__name__}: {str(e)[:120]}")
        finally:
            for p in (local, local + ".part"):
                if os.path.exists(p):
                    os.remove(p)


def iter_lines(url: str):
    with _get(url, timeout=180) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                yield line


def iter_docs(src: dict, max_chars: float, cache_dir: str):
    """Yield (text, url, lang_prob) from one source, honouring its fetch kind."""
    if src["kind"] == "parquet":
        urls = hf_parquet_urls(src["repo"], src["config"])
        print(f"    {len(urls)} parquet shard(s): {src['repo']}/{src['config']}")
        total = 0
        for row in iter_parquet(urls, cache_dir):
            # HPLT carries crawl-level filters; respect them instead of
            # re-deriving them from text.
            if row.get("robotstxt") not in (None, "allowed"):
                continue
            if row.get("filter") not in (None, "keep"):
                continue
            text = row.get("text") or row.get("content") or ""
            if not text:
                continue
            # Language confidence: HPLT `prob` (float or per-segment list),
            # GlotCC `identification-prob`. Sources without it yield None.
            prob = row.get("prob")
            if prob is None:
                prob = row.get("identification-prob")
            if isinstance(prob, list):
                vals = [p for p in prob if isinstance(p, (int, float))]
                prob = max(vals) if vals else None
            if prob is not None:
                try:
                    prob = float(prob)
                except (TypeError, ValueError):
                    prob = None
            yield text, row.get("url") or row.get("u") or row.get("warc-target-uri"), prob
            total += len(text)
            if max_chars and total >= max_chars:
                return

    elif src["kind"] == "lines":
        url = f"{HF}/datasets/{src['repo']}/resolve/main/{src['path']}"
        # One sentence per line: buffer into pseudo-docs so the model sees
        # paragraph context and dedupe compares whole passages, not fragments.
        buf: list[str] = []
        size = 0
        total = 0
        for line in iter_lines(url):
            buf.append(line)
            size += len(line) + 1
            if size >= 1200:
                doc = "\n".join(buf)
                buf, size = [], 0
                total += len(doc)
                yield doc, None, None
                if max_chars and total >= max_chars:
                    return
        if buf:
            yield "\n".join(buf), None, None
    else:
        raise ValueError(src["kind"])


# -------------------------------------------------------------------- driver

def out_path(out_dir: str, key: str) -> str:
    return os.path.join(out_dir, f"{FILE_PREFIX[SOURCES[key]['kind']]}_{key}_npi.jsonl")


def load_manifest(out_dir: str) -> dict[str, dict]:
    path = os.path.join(out_dir, "MANIFEST.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {e["file"]: e for e in json.load(f)}


def save_manifest(out_dir: str, entries: dict[str, dict]):
    path = os.path.join(out_dir, "MANIFEST.json")
    ordered = [entries[k] for k in sorted(entries)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=1)
        f.write("\n")


def collect(key: str, out_dir: str, args) -> dict:
    src = SOURCES[key]
    path = out_path(out_dir, key)
    resume = os.path.exists(path) and not args.restart
    min_chars = src.get("min_chars", args.min_chars)
    flt = Filter(min_chars, args.min_deva, not args.no_dedupe, args.min_prob)
    seeded = flt.load_existing(path) if resume else 0
    if args.restart and os.path.exists(path):
        os.remove(path)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[{key}] {src['license']}  {'-> ' + path if not seeded else f'resume, {seeded:,} docs on disk'}")
    t0 = time.time()
    docs = 0
    mode = "a" if resume else "w"
    with open(path, mode, encoding="utf-8") as f:
        try:
            for text, url, prob in iter_docs(src, args.max_chars, args.cache_dir):
                clean = flt(text, prob)
                if clean is None:
                    continue
                row = {"text": clean, "lang": "npi", "source": src["repo"]}
                if url:
                    row["url"] = url
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                docs += 1
                if docs % 10_000 == 0:
                    f.flush()
                    print(
                        f"    {docs:,} kept | {flt.chars/1e6:,.0f}M chars | "
                        f"{time.time()-t0:.0f}s",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\n    interrupted — keeping what is on disk")
        except Exception as e:
            # One flaky shard should not lose an hour of crawling; keep the
            # partial file and let --resume pick up where this left off.
            print(f"    [warn] {key} stopped early: {type(e).__name__}: {str(e)[:160]}")

    s = flt.stats
    print(
        f"[{key}] kept {docs:,} (+{seeded:,} existing) | {flt.chars/1e6:,.1f}M chars | "
        f"raw={s['raw']:,} offlang={s['offlang']:,} short={s['short']:,} "
        f"offscript={s['offscript']:,} junk={s['junk']:,} dup={s['dup']:,} | {time.time()-t0:.0f}s"
    )
    return {
        "file": os.path.join(out_dir, os.path.basename(path)),
        "lang": "npi",
        "source": src["repo"],
        "url": f"{HF}/datasets/{src['repo']}",
        "license": src["license"],
        "train_only": src.get("train_only", False),
        "docs": docs + seeded,
        "chars": flt.chars,
        "notes": (
            f"{src['notes']} filters: min_chars={min_chars} "
            f"min_deva={args.min_deva} min_prob={args.min_prob} "
            f"dedupe={not args.no_dedupe}; run raw={s['raw']} offlang={s['offlang']} "
            f"short={s['short']} offscript={s['offscript']} junk={s['junk']} dup={s['dup']}"
        ),
    }


def main():
    ap = argparse.ArgumentParser(description="Collect permissively licensed Nepali text")
    ap.add_argument("--sources", default="all", help="comma-separated keys, or 'all'")
    ap.add_argument("--out_dir", default="data/nepali")
    ap.add_argument("--min_chars", type=int, default=200, help="drop docs shorter than this")
    ap.add_argument("--min_deva", type=float, default=0.5, help="min Devanagari letter ratio")
    ap.add_argument("--min_prob", type=float, default=0.8,
                    help="min crawl language-confidence, when the source carries one")
    ap.add_argument("--cache_dir", default="~/.cache/akshara_nepali",
                    help="one parquet shard is downloaded here at a time, then deleted")
    ap.add_argument("--max_chars", type=float, default=0, help="per-source char budget (0 = all)")
    ap.add_argument("--no_dedupe", action="store_true", help="skip exact dedupe (faster, smaller RAM)")
    ap.add_argument("--restart", action="store_true", help="truncate the source's file instead of appending")
    ap.add_argument("--list", action="store_true", help="print the source registry and exit")
    args = ap.parse_args()
    args.cache_dir = os.path.expanduser(args.cache_dir)

    if args.list:
        for k, s in SOURCES.items():
            print(f"{k:<11} {s['license']:<32} {s.get('config') or s.get('path')}  {s['repo']}")
        return

    keys = list(SOURCES) if args.sources == "all" else [k.strip() for k in args.sources.split(",")]
    unknown = [k for k in keys if k not in SOURCES]
    if unknown:
        sys.exit(f"unknown source(s): {', '.join(unknown)} — try --list")

    entries = load_manifest(args.out_dir)
    for key in keys:
        entries[out_path(args.out_dir, key)] = collect(key, args.out_dir, args)
        save_manifest(args.out_dir, entries)
    print(f"\n[manifest] {os.path.join(args.out_dir, 'MANIFEST.json')}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        assert normalize("  नेपाली\u00a0 भाषा \n\n\n छ  ") == "नेपाली भाषा\n\nछ"
        assert "\u200d" in normalize("देखापर्‍यो")          # ZWJ survives
        assert deva_ratio("नेपाली भाषा") == 1.0 and deva_ratio("English") == 0.0
        assert not is_junk("नेपाली भाषाको पहिलो वाक्य हो ।\nदोस्रो वाक्य फरक छ ।")
        assert is_junk("\n".join(["गृहपृष्ठ"] * 8))          # repeated menu
        assert is_junk("123 456 789 012 345 678 901 234")     # digit swamp
        f = Filter(10, 0.5, True)
        assert f("छोटो") is None and f.stats["short"] == 1
        assert f("This is an English sentence only.") is None and f.stats["offscript"] == 1
        ok = "नेपाली भाषामा लेखिएको एउटा लामो वाक्य हो, जुन परीक्षणका लागि प्रयोग गरिन्छ ।"
        assert f(ok) is not None and f(ok) is None and f.stats["dup"] == 1
        assert f(ok, 0.1) is None and f.stats["offlang"] == 1
        assert out_path("/tmp/x", "glotcc").endswith("hf_glotcc_npi.jsonl")
        print("collect_nepali selftest passed")
        sys.exit(0)

    main()
