"""
Step 2 — Tokenizer verification.
Checks whether Qwen3.5-0.8B tokenizer handles Nepali/Devanagari
as proper subword units or falls back to raw UTF-8 bytes.

Run:
    python tokenizer/verify.py
"""

from transformers import AutoTokenizer


def check(tok, text):
    tokens = tok.encode(text)
    decoded = [tok.decode([t]) for t in tokens]
    # byte fallback tokens look like '\xe0\xa4...' — raw hex
    has_byte_fallback = any("\\x" in repr(t) for t in decoded)
    status = "❌ BYTE FALLBACK" if has_byte_fallback else "✅ OK"
    print(f"{status}  {text!r:25} → {len(tokens):2d} tokens → {decoded}")


def main():
    print("Loading Qwen3.5-0.8B tokenizer...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    print(f"Vocab size: {tok.vocab_size:,}\n")

    print("=== Nepali (Devanagari) ===")
    check(tok, "नमस्ते")          # namaste
    check(tok, "नेपाल")           # Nepal
    check(tok, "सम्झना")          # conjunct म्
    check(tok, "क्ष")             # classic conjunct
    check(tok, "राम्रो छ")        # good (virama)
    check(tok, "धन्यवाद")         # thank you
    check(tok, "काठमाडौं")        # Kathmandu

    print("\n=== Mixed Nepali + English ===")
    check(tok, "hello नेपाल")
    check(tok, "OCR प्रणाली")     # OCR system
    check(tok, "2024 सालमा")      # in year 2024

    print("\n=== English (baseline) ===")
    check(tok, "hello world")
    check(tok, "document recognition")

    print("\n=== Devanagari digits ===")
    check(tok, "०१२३४५६७८९")     # Devanagari 0-9

    # Summary: tokens per character ratio
    # Good tokenizer: ~1.5-3 tokens per Nepali word
    # Bad tokenizer:  ~9+ tokens per Nepali word (byte fallback, 3 bytes each)
    sample = "नेपाल राम्रो देश हो"
    t = tok.encode(sample)
    words = sample.split()
    print(f"\nEfficiency check: '{sample}'")
    print(f"  {len(words)} words → {len(t)} tokens ({len(t)/len(words):.1f} tokens/word)")
    print(f"  (Good: <4 t/word. Bad: >9 t/word means byte fallback)")


if __name__ == "__main__":
    main()
