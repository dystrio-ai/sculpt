"""Dataset utilities: load wikitext splits for calibration, training, and eval."""

from __future__ import annotations

import logging
from typing import Dict, List

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
    """Canonical split (matches what we ran):

    - wikitext-2 train: calibration + repair training
    - wikitext-2 test: eval (w2-test)
    - wikitext-103 validation: eval (w103-valid)
    """
    w2_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    w2_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    w103_val = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

    return {
        "cal": _collect_texts(w2_train, n_cal),
        "train": _collect_texts(w2_train, n_train),
        "eval_w2": _collect_texts(w2_test, n_eval),
        "eval_w103": _collect_texts(w103_val, n_eval),
    }


def load_ood_texts(n_texts: int) -> List[str]:
    """Load out-of-distribution eval texts (OpenWebText -> C4 fallback).

    Uses streaming to avoid downloading entire datasets.
    Returns deterministic first *n_texts* non-empty texts.
    """
    candidates = [
        ("Skylion007/openwebtext", None, "train"),
        ("allenai/c4", "en", "validation"),
    ]
    for dataset_id, config, split in candidates:
        try:
            args = (dataset_id,) if config is None else (dataset_id, config)
            ds = load_dataset(
                *args, split=split, streaming=True, trust_remote_code=True,
            )
            texts: List[str] = []
            for example in ds:
                t = example.get("text", "")
                if t and t.strip():
                    texts.append(t)
                    if len(texts) >= n_texts:
                        break
            if texts:
                _log.info(
                    "OOD dataset loaded: %s (%d texts)", dataset_id, len(texts),
                )
                return texts
        except Exception as exc:
            _log.debug("OOD candidate %s failed: %s", dataset_id, exc)
            continue

    _log.warning("No OOD dataset could be loaded — OOD eval will be skipped")
    return []
