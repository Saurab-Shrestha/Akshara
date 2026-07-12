"""Thorough language-understanding eval for the pretrained decoder.

Two kinds of test:

1. GENERATION (greedy / argmax, deterministic) across categories that matter
   for the OCR use case: general Nepali, formal/document Nepali, code-switched
   Nepali+English+numbers, general English, English factual, and numerals/dates.
   Greedy (not sampled) so the output reflects the model's most-confident
   continuation — the clearest read on what it has actually learned.

2. PERPLEXITY minimal pairs: for a correct sentence and a word-scrambled
   version of the SAME words, a model that understands language must assign
   lower perplexity (higher probability) to the correct ordering. This is a
   quantitative check that it learned grammar/structure, not just vocabulary.

Usage:
    PYTHONPATH=. python scripts/lm_eval.py --ckpt checkpoints/pretrain_a100_v4.pt
"""
import argparse
import math

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.models.hybrid_decoder import HybridDecoder

GEN_PROMPTS = {
    "Nepali — general": [
        "नेपाल एक सुन्दर देश हो जहाँ",
        "मानिसहरू बिहान सबेरै उठेर",
        "विज्ञान र प्रविधिको विकासले",
    ],
    "Nepali — formal / document": [
        "यस सूचनाद्वारा सम्पूर्ण कर्मचारीहरूलाई जानकारी गराइन्छ कि",
        "नेपाल सरकार, शिक्षा मन्त्रालयले",
    ],
    "Code-switch (Nepali + English + numbers)": [
        "मेरो फोन नम्बर 9841 हो र मेरो नाम",
        "काठमाडौंस्थित ABC Company Ltd. ले",
    ],
    "English — general": [
        "The weather today is",
        "Once upon a time there was a",
    ],
    "English — factual": [
        "The capital of Nepal is",
        "The largest ocean on Earth is the",
    ],
    "Numerals / dates": [
        "आजको मिति २०८१ साल",
        "The meeting is scheduled for January",
    ],
}

# (label, correct, scrambled-same-words). Correct should get LOWER perplexity.
PPL_PAIRS = [
    ("Nepali",
     "नेपाल दक्षिण एसियामा अवस्थित एक भूपरिवेष्टित देश हो ।",
     "अवस्थित देश एसियामा हो नेपाल एक दक्षिण भूपरिवेष्टित ।"),
    ("English",
     "The sun rises in the east and sets in the west .",
     "east rises the in sun sets the and west the in ."),
]


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ck["config"]
    model = HybridDecoder(
        vocab_size=c["vocab_size"], n_embed=c["n_embed"], n_heads=c["n_heads"],
        n_kv_heads=c["n_kv_heads"], n_layers=c["n_layers"],
        max_seq_len=c["max_seq_len"], attn_every=c["attn_every"],
        use_fla=c.get("use_fla", False),
    )
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    return model, c["max_seq_len"], ck.get("step")


@torch.no_grad()
def greedy(model, tok, prompt, device, max_seq_len, max_new=40):
    eos = tok.eos_token_id
    ids = [eos] + tok.encode(prompt)
    x = torch.tensor([ids], device=device)
    for _ in range(max_new):
        logits, _ = model(x[:, -max_seq_len:])
        nxt = int(logits[0, -1].argmax())
        if nxt == eos:
            break
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return tok.decode(x[0, 1:].tolist())


@torch.no_grad()
def perplexity(model, tok, text, device, max_seq_len):
    eos = tok.eos_token_id
    ids = ([eos] + tok.encode(text))[: max_seq_len + 1]
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    logits, _ = model(x)
    loss = F.cross_entropy(logits[0].float(), y[0])
    return math.exp(min(loss.item(), 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_new", type=int, default=40)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    model, max_seq_len, step = load(a.ckpt, a.device)
    print(f"\n{'='*70}\n LANGUAGE EVAL — checkpoint step {step}\n{'='*70}\n")

    print("── 1. GREEDY GENERATION (deterministic best continuation) ──\n")
    for cat, prompts in GEN_PROMPTS.items():
        print(f"[{cat}]")
        for p in prompts:
            out = greedy(model, tok, p, a.device, max_seq_len, a.max_new)
            print(f"  · {p}")
            print(f"    → {out}\n")

    print("── 2. PERPLEXITY MINIMAL PAIRS (correct should be LOWER) ──\n")
    for label, correct, scrambled in PPL_PAIRS:
        pc = perplexity(model, tok, correct, a.device, max_seq_len)
        ps = perplexity(model, tok, scrambled, a.device, max_seq_len)
        verdict = "PASS ✓" if pc < ps else "FAIL ✗"
        print(f"[{label}]  correct ppl={pc:6.2f}   scrambled ppl={ps:6.2f}   {verdict}")
    print()


if __name__ == "__main__":
    main()
