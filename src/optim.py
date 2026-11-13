"""
Optimizer construction shared by pretrain.py and train_ocr.py.

WHY MUON (v2)
-------------
AdamW scales each weight's update independently. Muon treats a weight
*matrix* as a matrix: it takes the momentum-averaged gradient and
orthogonalises it (Newton–Schulz iterations, ~5 matmuls), so every
direction in the matrix gets an update of similar size instead of a few
dominant directions eating the whole step. Result in the Moonshot / Keller
Jordan experiments: ~1.3–2× fewer tokens to reach the same loss.

Muon only makes sense for 2-D hidden weight matrices. Everything else
stays on AdamW:
    - token embedding / tied lm_head  (a lookup table, not a linear map)
    - norms, biases, GDN gates/conv    (1-D or 3-D)
    - vision encoder + connector       (DINOv2 was pretrained with AdamW;
                                        switching optimizer at fine-tune
                                        time gives no clear gain)

adjust_lr_fn="match_rms_adamw" rescales Muon's update to AdamW's RMS, so
one `lr` and one `weight_decay` drive both optimizers and the cosine
schedule can set them together.
"""

from __future__ import annotations

import torch


class MultiOptimizer:
    """Steps several optimizers as one (Muon rejects non-2-D params, so we need two)."""

    def __init__(self, *opts):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return [o.state_dict() for o in self.opts]

    def load_state_dict(self, sd):
        for o, s in zip(self.opts, sd):
            o.load_state_dict(s)


def _is_muon_param(name: str, p: torch.Tensor) -> bool:
    # ".layers." = decoder blocks only (HybridDecoder: "layers.N…", VLM:
    # "decoder.layers.N…"). DINOv2 uses "layer.N" (singular) → stays AdamW.
    return p.ndim == 2 and ".layers." in f".{name}"


def build_optimizer(model: torch.nn.Module, cfg, device: str):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    muon = [p for n, p in named if cfg.optimizer == "muon" and _is_muon_param(n, p)]
    muon_ids = {id(p) for p in muon}
    rest = [(n, p) for n, p in named if id(p) not in muon_ids]
    # v2: no weight decay on the (tied) token embedding, as in OLMo 2 / SmolLM3.
    # Off by default so v1 optimizer checkpoints keep their param-group layout.
    no_wd_embed = getattr(cfg, "no_wd_embed", False)
    decay = lambda n, p: p.ndim >= 2 and not (no_wd_embed and "embed" in n)

    adamw = torch.optim.AdamW([
        {"params": [p for n, p in rest if decay(n, p)],     "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in rest if not decay(n, p)], "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.95),
       fused=device.startswith("cuda") and torch.cuda.is_available())

    if not muon:
        return adamw  # plain AdamW keeps v1 optimizer checkpoints resumable

    print(f"[optim] Muon: {sum(p.numel() for p in muon)/1e6:.1f}M params in "
          f"{len(muon)} matrices | AdamW: {sum(p.numel() for _, p in rest)/1e6:.1f}M")
    return MultiOptimizer(
        torch.optim.Muon(muon, lr=cfg.lr, weight_decay=cfg.weight_decay,
                         adjust_lr_fn="match_rms_adamw"),
        adamw,
    )


if __name__ == "__main__":
    from types import SimpleNamespace
    from src.models.hybrid_decoder import HybridDecoder

    torch.manual_seed(0)
    m = HybridDecoder(vocab_size=256, n_embed=64, n_heads=4, n_kv_heads=2,
                      n_layers=4, max_seq_len=32, qk_norm=True)
    cfg = SimpleNamespace(optimizer="muon", lr=3e-3, weight_decay=0.1)
    opt = build_optimizer(m, cfg, "cpu")
    assert isinstance(opt, MultiOptimizer)
    # embedding must not be on Muon
    assert all(p is not m.token_embed.weight for p in opt.opts[0].param_groups[0]["params"])

    ids = torch.randint(0, 256, (8, 32))
    first = None
    for _ in range(30):
        opt.zero_grad()
        _, loss = m(ids[:, :-1], ids[:, 1:])
        loss.backward()
        opt.step()
        first = first or loss.item()
    assert loss.item() < first, f"loss did not drop: {first:.3f} → {loss.item():.3f}"

    opt.load_state_dict(opt.state_dict())  # round-trip
    print(f"optim self-check passed: loss {first:.3f} → {loss.item():.3f}")
