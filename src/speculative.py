"""
Speculative decoding (Leviathan et al. 2023; Chen et al. 2023).

A small draft model proposes k tokens one by one (cheap). The big target model
then scores all k in ONE forward pass and keeps the longest prefix it agrees
with, plus one token of its own. The output has exactly the target model's
distribution (greedy: exactly the target's greedy text), but the target runs
about once per accepted run instead of once per token.

Acceptance rule for proposal x with target prob p(x) and draft prob q(x):
    accept with probability min(1, p(x) / q(x))
    on rejection sample from normalize(max(0, p − q))
    if all k are accepted, sample one more token from the target.

Draft and target must share the tokenizer. Akshara's 148M ablation proxy
(configs/ablations) and the 443M model do.

    from src.speculative import speculative_generate
    out, stats = speculative_generate(target, draft, ids, max_new_tokens=200, k=4)
"""

from __future__ import annotations

import torch

from src.models.hybrid_decoder import probs_from_logits


@torch.no_grad()
def speculative_generate(target, draft, token_ids: torch.Tensor, max_new_tokens: int, k: int = 4,
                         temperature: float = 0.0, top_k: int | None = None,
                         top_p: float | None = None, eos_token_id: int | None = None):
    """token_ids: (1, T) prompt. Returns ((1, T + n) ids, {"accepted", "proposed", "target_calls"})."""
    assert token_ids.shape[0] == 1, "speculative decoding here is batch size 1"
    dist = lambda logits: probs_from_logits(logits, temperature, top_k, top_p)[0]
    pick = lambda p: torch.multinomial(p, 1)  # one-hot p (greedy) always picks its argmax

    tc, dc = target.new_cache(), draft.new_cache()
    p_next = dist(target.forward_cached(token_ids, tc)[:, -1])   # target's view of the next token
    d_logits = draft.forward_cached(token_ids, dc)[:, -1]
    out, stats = token_ids, {"accepted": 0, "proposed": 0, "target_calls": 1}

    while out.shape[1] - token_ids.shape[1] < max_new_tokens:
        # 1. draft proposes k tokens. Draft cache tensors are never modified in
        #    place, so a shallow copy is a snapshot to restore afterwards.
        snap = {"pos": dc["pos"], "layers": [dict(c) for c in dc["layers"]]}
        props, qs = [], []
        for _ in range(k):
            q = dist(d_logits)
            x = pick(q)
            props.append(x)
            qs.append(q)
            d_logits = draft.forward_cached(x.view(1, 1), dc)[:, -1]

        # 2. target scores all proposals in one pass (keep_states for rollback)
        v_logits = target.forward_cached(torch.cat(props).view(1, k), tc, keep_states=True)
        ps = [p_next] + [dist(v_logits[:, i]) for i in range(k)]
        stats["target_calls"] += 1

        # 3. accept the longest agreeing prefix
        n = 0
        for i in range(k):
            x = props[i].item()
            if torch.rand(()) * qs[i][x] >= ps[i][x]:    # reject with prob 1 - min(1, p/q)
                break
            n += 1
        if n < k:
            residual = (ps[n] - qs[n]).clamp(min=0)
            fix = pick(residual / residual.sum() if residual.sum() > 0 else ps[n])
        else:
            fix = pick(ps[k])
        new = torch.cat(props[:n] + [fix]).view(1, -1)
        stats["accepted"] += n
        stats["proposed"] += k

        # 4. roll both models back to the accepted tokens and feed the correction.
        # ponytail: the correction costs the target one extra 1-token pass per
        # round; fold it into the next verify pass if decoding speed matters.
        target.rollback(tc, n)
        p_next = dist(target.forward_cached(fix.view(1, 1), tc)[:, -1])
        dc.update(snap)
        d_logits = draft.forward_cached(new, dc)[:, -1]

        out = torch.cat([out, new], dim=1)
        if eos_token_id is not None and (new == eos_token_id).any():
            first = (out[0, token_ids.shape[1]:] == eos_token_id).nonzero()[0, 0]
            out = out[:, : token_ids.shape[1] + first + 1]
            break

    return out[:, : token_ids.shape[1] + max_new_tokens], stats
