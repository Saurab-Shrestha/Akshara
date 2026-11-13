# Ablations before the 443M run

Following the [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook):
change **one** thing per run, keep parameter counts matched, train enough tokens,
and decay every run so final losses are comparable. If a change hurts at this
scale, drop it. If it helps, keep it for the 443M run.

The proxy is 148M parameters (768 wide, 16 layers), trained on 1B tokens of
`corpus_v6` with the full v2 recipe (`base.json`). Each run takes about 2.5–7
A100-hours. Checkpoints are backed up to the private `Saurab0/akshara-ablations` repo.

| Run | Changes | Question |
|---|---|---|
| a1 | nothing (baseline) | reference curve |
| a2 / a3 | lr 8e-4 / 1.6e-3 | is 4e-4 too low? does Muon diverge higher? |
| a4 | all-attention, 18 layers | does the 3:1 Gated DeltaNet hybrid earn its complexity at 1024 context? |
| a5 | AdamW instead of Muon | is Muon worth its LR sensitivity? |

**Order:** run a1–a3 first. Then run a4 and a5 at the best LR from those:

```bash
PYTHONPATH=. python scripts/pretrain.py --config configs/ablations/a1_hybrid_muon_4e-4.json
PYTHONPATH=. python scripts/pretrain.py --config configs/ablations/a4_attention_only.json --lr 8e-4 --min_lr 8e-5
# a session died? add: --resume checkpoints/ablations/<run>.pt   (download it from the HF repo first)
```

**How to read the results.** All runs share the tokenizer and `val.bin`, so their
dev losses can be compared directly. Look at the final dev loss after the decay,
and at whether the curve has spikes. Differences under ~0.01 are noise: to measure
the noise floor, rerun a1 with `--seed 7`. If a4 is within noise of a1, choose
pure attention, which is simpler and has more proven recipes. If a5 is within
noise of the best Muon run, AdamW is the safer choice.
