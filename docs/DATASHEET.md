# Datasheet — Akshara Corpus (working name)

A text corpus for Nepali, English and the languages of Nepal, built to train
Akshara's tokenizer and language model. Format follows *Datasheets for
Datasets* (Gebru et al., 2021). This file is the source for the dataset card
and the dataset paper — **update it as the data is built**, not afterwards.

Status: **draft, 2026-11-13.** Full build `corpus_v5` done: 12.58B chars, 3.19B train tokens (48k tokenizer), private HF repo `Saurab0/akshara-corpus-v5`. Numbers marked TODO come from the full build
(`scripts/prepare_data.py`) and from the native-speaker audit (§3.6).

---

## 1. Motivation

**Why was it created?** Open corpora barely cover the languages of Nepal.
Nepali appears in large web crawls, but Maithili, Newari, Bhojpuri, Tamang,
Tharu, Limbu and the Rai languages have little or no clean, documented text.
Large multilingual tokenizers reflect this. Measured on held-out text, the
Qwen3.5 tokenizer averages 1.46 chars/token on Devanagari, while a tokenizer
trained on this corpus averages 3.52.

**What task?** Primarily pretraining a tokenizer and a language model for a
multimodal LLM. Also usable for language ID, low-resource LM research and
tokenizer studies.

**Who created it / funded it?** TODO: names, affiliation, compute credits.

---

## 2. Composition

**What do instances represent?** One document per JSONL row:
`{"text": "...", "lang": "<ISO 639-3>"}` (plus `"source"` for hand-collected
subsets). Training also uses a packed uint16 token version (`*.bin`).

### 2.1 Main mix (character budget, default 12B chars ≈ 3.4B tokens)

| Source | Lang | Share | License |
|---|---|---|---|
| FineWeb-2 `npi_Deva` | npi | 30% | ODC-By 1.0 |
| FineWeb-2 `hin_Deva` | hin | 13% | ODC-By 1.0 |
| FineWeb-2 `mar_Deva` / `san_Deva` | mar / san | 3% / 1% | ODC-By 1.0 |
| FineWeb-Edu (EduScore ≥ 3) | eng | 24% | ODC-By 1.0 |
| codeparrot-clean (Python; MIT/Apache/BSD/ISC/MPL/CC0/Unlicense files only; quality-filtered) | code | 9% | per-file permissive |
| The Stack dedup v1: JSON / YAML / Markdown (all repo licenses permissive; 200 B–50 KB; line ≤ 1000) | code | 0.4% / 0.3% / 0.3% | per-file permissive |
| FineMath `finemath-4plus` | eng | 6% | ODC-By 1.0 |
| Cosmopedia v2 (smollm-corpus) | eng | 4% | ODC-By 1.0 |
| BPCC seed + daily en↔npi / mai / sat (human-translated) | eng-npi / eng-mai / eng-sat | 0.3% / 0.15% / 0.15% | CC0 / CC-BY-4.0 |
| BPCC mined en↔npi (NLLB, Samanantar, comparable) / en↔mai (NLLB) | eng-npi / eng-mai | 0.95% / 0.05% | CC0 (packaging) |
| OPUS-100 en↔ne | eng-npi | 0.4% | unknown → train-only |
| Nepali extras from `scripts/collect_nepali.py` (HPLT 2.0, IndicCorpV2, FinePDFs, GlotCC, DCAD) | npi | `--nepali_share` (default 15%, carved out of FineWeb-2 ne) | per source |
| Wikipedia ne / hi / en (20231101) | npi / hin / eng | 3% / 2% / 2% | CC BY-SA 4.0 |

### 2.2 Languages of Nepal (all available text, repeated per language)

Full build `corpus_v5` (2026-11-13). "Unique chars" is after the script filter
and dedup, summed over FineWeb-2, Wikipedia and the hand-collected sources.
Languages with ≥ ~100M chars are taken once and smaller ones are repeated up to 4×
(Bible text 1×), see §3.

| Language | ISO | Script | Unique chars | Repeats |
|---|---|---|---|---|
| Maithili | mai | Devanagari | 350M | 1× |
| Newari / Nepal Bhasa | new | Devanagari | 220M | 1× |
| Bhojpuri | bho | Devanagari | 140M | 1× |
| Santali | sat | Ol Chiki | 46M | 4× |
| Awadhi | awa | Devanagari | 41M | 4× |
| Sunwar | suz | Devanagari | 16M | 4× |
| Tamang (Eastern) | taj | Devanagari | 8M | 4× |
| Limbu | lif | Devanagari + Limbu | 5M | 4× |
| Doteli | dty | Devanagari | 3M | 4× |
| Tharu (Dangaura) | thl | Devanagari | 2M | 4× |
| Sherpa / Tharu (Rana) / Rawat | xsr / thr / rav | Devanagari | ~1M each | 4× |
| Magar, Bantawa, Dhimal, Kulung, Thakali, … | mgp / ybh / dhi / klr / the / … | — | < 0.5M each | 4× |
| Other Rai languages, Gurung, Chepang | various | — | none | — |

### 2.3 Hand-collected sources

Two collection passes (HF multilingual corpora; non-HF sources such as eBible,
Leipzig, OPUS, Wikisource, Common Voice text) write to `data/extra_langs/`
with manifests `MANIFEST_hf.json` and `MANIFEST_web.json`.
TODO: summarize per language and license here once the manifests are final.
List the "needs user" sources (registration or unclear license) separately.

**Polite crawl (2026-11-12/13).** Train-only unless openly licensed:

| Source | Lang | Docs | Chars | License |
|---|---|---|---|---|
| NepalMandal (1,175 of 2,244 articles; stopped when the server began dropping connections) | new | 1,168 | 2.37M | all rights reserved |
| Kirat Yakthung Chumlung (Limbu-script paragraphs only) | lif | 9 | 5.5K | none stated |
| Pangloss | klr / taj | 8 / 1 | 51K / 1.5K | CC BY-NC-ND / NC-SA 2.5 |
| Yakkha grammar (langsci/66) | ybh | 23 | 61K | CC BY 4.0 (publishable) |
| Rising Voices | the | 1 | 7.6K | CC BY 3.0 (publishable) |
| CulturaX | new / bho / mai | 3,995 / 208 / 48 | 3.6M total | mC4 + OSCAR terms |

**Sikkim Herald language editions:** 126 PDFs are archived in
`data/extra_langs/raw/sikkim_herald/` (lif 39, xsr 29, suz 26, mgp 17, gvr 7,
bap 4, taj 4; there is no Newar edition). None has a Unicode text layer. They use
legacy ASCII fonts (Limbu "Srijonga", Sunwar "kirat1"/"Shipmoo", Tibetan-script
fonts for Tamang/Sherpa, a "Magar" font), and Gurung is image-only. Unlocking
them needs per-font Unicode maps (ask the Sikkim IPR Department) or OCR.
The Rai edition appears to use the Kirat Rai script (Unicode U+16D40, added 2024).

**Is anything confidential or personal?** The sources are public web text,
Wikipedia and published translations. Web text can contain names and
contact details of private individuals. TODO: decide on PII filtering
(emails and phone numbers at minimum) before release.

**Offensive content?** Web subsets are filtered by their upstream pipelines
(FineWeb-2 / FineWeb-Edu), but not by us. TODO: report a toxicity sample rate.

---

## 3. Collection and preprocessing

1. **Streaming download** (`stream_texts`). Documents under 50 characters are dropped.
   FineWeb-Edu needs EduScore ≥ 3.
2. **Character budgets** per main source; minority sources are taken whole.
3. **Script filter (minority sources).** A document is kept only if ≥ 50% of
   its letters are in the expected script (Devanagari U+0900–097F, Limbu
   U+1900–194F, Ol Chiki U+1C50–1C7F). This was added after finding English
   pages labelled as Limbu script.
4. **Exact dedup** (SHA-1 of the stripped text) across all minority sources,
   including hand-collected ones. Main sources rely on upstream dedup
   (FineWeb-2 MinHash).
5. **Validation split** is 1 in 200 documents by CRC32 of the text, so it is
   deterministic and spans every source.
6. **Repetition.** Minority train docs are written 4× in reshuffled passes;
   Bible-derived text is kept 1×.
7. **Tokenization** uses our 48k byte-level BPE (`scripts/train_tokenizer.py`), packed to uint16.

**Known pitfalls found so far**
- FineWeb-2 `new_Deva` has an extra column, so casting it to the declared schema
  failed and the source silently returned 0 docs. Fixed by reading the parquet directly.
- `lif_Limb` (Limbu script) contains English pages, which the script filter now drops.

### 3.6 Quality audit — TODO (required before publishing)
Native speakers rate 100 random lines per language as *correct language / wrong
language / not running text*, following Kreutzer et al. (2022). Results are
reported per language, and subsets under the agreed threshold are dropped or flagged.

| Language | Raters | Lines | Correct % | Wrong lang % | Junk % |
|---|---|---|---|---|---|
| TODO | | | | | |

---

## 4. Uses

**Intended:** tokenizer and LM pretraining; research on low-resource and
Indic languages; OCR and multimodal models for Nepal's languages.

**Not intended:** identifying or profiling individuals; claims about a
community's language use (web text skews urban, formal and news-heavy);
being treated as a standard reference for any language's orthography.

**Composition caveats:** minority subsets are tiny and repeated 4×. Several
are dominated by one domain (news, Wikipedia stubs, Bible). Newari Wikipedia
contains many bot-generated stub articles.

---

## 5. Distribution

**Train-only data (decided 2026-11-12).** Rows from sources with
non-commercial, no-derivatives, not-redistributable or unknown licenses carry
`"train_only": true`. They're used for training and **never published** in the
corpus. Today that covers eBible Limbu, Sunwar and Tamang (CC BY-NC-ND 3.0),
NepTam Tamang (CC BY-NC 4.0), and any file missing from the manifests.
The release script must drop these rows. "All rights reserved" sources crawled for training
(e.g. NepalMandal) are also train-only.

**Crawling policy.** We crawl only sites that haven't opted out. We respect
robots.txt and crawl-delay, use a self-identifying research user agent,
and never bypass logins or anti-bot measures. Sites that reserve rights against
AI training (e.g. Nepalbhasa Times `Content-Signal: ai-train=no`, Gorkhapatra
blocking AI crawlers) are **excluded unless the publisher gives written permission**. Note: training on NC data may
also restrict commercial use of the resulting model. Revisit before any
commercial use.

- **Where:** Hugging Face dataset repo (currently private `Saurab0/akshara-corpus-v5`, packed tokens + tokenizer only), plus the rebuild scripts in this repo.
- **License:** follows the most restrictive license in each subset: ODC-By for the FineWeb
  family, CC BY-SA 4.0 for Wikipedia. Subsets that can't be redistributed
  (some Bible translations, archives) ship as **rebuild scripts only**.
- **Communities:** for Indigenous languages (Rai, Limbu, Tamang, …), follow the
  CARE principles. TODO: contact the Language Commission of Nepal (भाषा आयोग) and
  community language organizations before release.

---

## 6. Maintenance

- **Maintainer:** TODO.
- **Versioning:** `corpus_v5` is the first public candidate. Changes are recorded below.
- **Contributions:** new minority-language text goes in as `data/extra_langs/*.jsonl` with a manifest entry.

| Version | Date | Change |
|---|---|---|
| v5-draft | 2026-11-12 | Main mix + FineWeb-2/Wikipedia minority languages, script filter, Newari schema fix |
