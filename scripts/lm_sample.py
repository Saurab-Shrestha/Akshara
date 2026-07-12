"""Quick language sanity check for the pretrained decoder.

Loads a pretrain checkpoint into HybridDecoder and generates continuations for
a few Nepali + English prompts. This tells us whether Stage-1 pretraining gave
the model a real language prior (coherent script, words, grammar) before we
attach the vision encoder for OCR.

Usage:
    PYTHONPATH=. python scripts/lm_sample.py --ckpt checkpoints/pretrain_a100_v4.pt
"""
import argparse

import torch
from transformers import AutoTokenizer

from src.models.hybrid_decoder import HybridDecoder

PROMPTS = [
    "नेपालको राजधानी",
    "हिमालय संसारको सबैभन्दा",
    "विद्यालयमा विद्यार्थीहरूले",
    "The capital of France is",
    "Water is made of hydrogen and",
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
def generate(model, tok, prompt, device, max_seq_len, max_new=50, temperature=0.7, top_k=40):
    eos = tok.eos_token_id           # training used EOS as the BOS sentinel
    ids = [eos] + tok.encode(prompt)
    x = torch.tensor([ids], device=device)
    for _ in range(max_new):
        logits, _ = model(x[:, -max_seq_len:])
        logits = logits[0, -1].float() / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1).item()
        if nxt == eos:
            break
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return tok.decode(x[0, 1:].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_new", type=int, default=50)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    model, max_seq_len, step = load(a.ckpt, a.device)
    print(f"[lm_sample] loaded {a.ckpt} (step {step}) | device={a.device}\n")
    for p in PROMPTS:
        out = generate(model, tok, p, a.device, max_seq_len,
                       max_new=a.max_new, temperature=a.temperature)
        print(f"PROMPT : {p}")
        print(f"OUTPUT : {out}\n")


if __name__ == "__main__":
    main()
