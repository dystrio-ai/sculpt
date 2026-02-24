"""Eval: token-level NLL perplexity."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


@torch.no_grad()
def eval_perplexity(
    model, tokenizer, texts: Sequence[str], max_len: int, device: str, max_eval_tokens: int,
) -> float:
    model.eval()
    nll_sum = 0.0
    tok_count = 0

    for t in texts:
        inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)
        inp = {k: v.to(device) for k, v in inp.items()}
        out = model(**inp, use_cache=False)
        logits = out.logits

        ids = inp["input_ids"][0]
        if ids.shape[0] < 2:
            continue

        target = ids[1:]
        pred = logits[0, :-1, :]
        logp = torch.log_softmax(pred, dim=-1)
        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)

        nll_sum += float(nll.sum().detach().cpu())
        tok_count += int(target.shape[0])

        if tok_count >= max_eval_tokens:
            break

    return float(np.exp(nll_sum / max(1, tok_count)))
