"""Dataset loading for calibration, training, and evaluation."""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

from datasets import load_dataset

_log = logging.getLogger(__name__)


def _collect_texts(ds, n: int, field: str = "text") -> List[str]:
    out: List[str] = []
    for i in range(len(ds)):
        t = ds[i].get(field, "")
        if t and t.strip():
            out.append(t)
            if len(out) >= n:
                break
    return out


def load_text_sets(n_cal: int, n_train: int, n_eval: int) -> Dict[str, List[str]]:
    w2_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    w2_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    w103_val = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    return {
        "cal": _collect_texts(w2_train, n_cal),
        "train": _collect_texts(w2_train, n_train),
        "eval_w2": _collect_texts(w2_test, n_eval),
        "eval_w103": _collect_texts(w103_val, n_eval),
    }


def deterministic_subset(
    texts: Sequence[str], n: int, seed: int = 0,
) -> List[str]:
    """Select a fixed-size subset of *texts* using a seeded shuffle.

    Stable across runs for the same seed — suitable for cheap eval during
    repair and early stopping.  If n >= len(texts) returns a copy of the
    full list.
    """
    import random

    if n >= len(texts):
        return list(texts)
    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    chosen = sorted(indices[:n])
    return [texts[i] for i in chosen]
