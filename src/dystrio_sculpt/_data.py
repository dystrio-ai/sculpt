"""Dataset loading for calibration, training, and evaluation."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from datasets import load_dataset

_log = logging.getLogger(__name__)

# Defaults that reproduce the original hardcoded behavior
DEFAULT_CALIB_DATASET = "wikitext"
DEFAULT_CALIB_CONFIG = "wikitext-2-raw-v1"
DEFAULT_CALIB_SPLIT = "train"
DEFAULT_CALIB_TEXT_FIELD = "text"
DEFAULT_EVAL_DATASET = "wikitext"
DEFAULT_EVAL_CONFIG = "wikitext-103-raw-v1"
DEFAULT_EVAL_SPLIT = "validation"


# Workload presets: map workload name -> CalibConfig kwargs.
# Users can always override individual fields via --calib-* flags.
WORKLOAD_PRESETS: Dict[str, Dict[str, str]] = {
    "general": {
        "dataset": DEFAULT_CALIB_DATASET,
        "config": DEFAULT_CALIB_CONFIG,
        "split": DEFAULT_CALIB_SPLIT,
        "text_field": DEFAULT_CALIB_TEXT_FIELD,
    },
    "code": {
        "dataset": "bigcode/the-stack-smol",
        "config": "default",
        "split": "train",
        "text_field": "content",
    },
    "chat": {
        "dataset": "tatsu-lab/alpaca",
        "config": "default",
        "split": "train",
        "text_field": "text",
    },
}


def calib_config_for_workload(workload: str) -> "CalibConfig":
    """Return a CalibConfig for a named workload preset."""
    if workload not in WORKLOAD_PRESETS:
        raise ValueError(
            f"Unknown workload {workload!r}. "
            f"Available: {list(WORKLOAD_PRESETS.keys())}"
        )
    return CalibConfig(**WORKLOAD_PRESETS[workload])


@dataclass
class CalibConfig:
    """Calibration corpus configuration — fully describes where text comes from."""

    dataset: str = DEFAULT_CALIB_DATASET
    config: str = DEFAULT_CALIB_CONFIG
    split: str = DEFAULT_CALIB_SPLIT
    text_field: str = DEFAULT_CALIB_TEXT_FIELD
    num_samples: Optional[int] = None
    seq_len: Optional[int] = None
    seed: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "calib_dataset": self.dataset,
            "calib_config": self.config,
            "calib_split": self.split,
            "calib_text_field": self.text_field,
            "calib_num_samples": self.num_samples,
            "calib_seq_len": self.seq_len,
            "calib_seed": self.seed,
        }


def _collect_texts(ds, n: int, field: str = "text") -> List[str]:
    out: List[str] = []
    for i in range(len(ds)):
        t = ds[i].get(field, "")
        if t and t.strip():
            out.append(t)
            if len(out) >= n:
                break
    return out


def _deterministic_sample(texts: List[str], n: int, seed: int) -> List[str]:
    """Deterministically pick *n* texts using a seeded shuffle."""
    if n >= len(texts):
        return texts
    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    chosen = sorted(indices[:n])
    return [texts[i] for i in chosen]


def load_calibration_corpus(
    calib: CalibConfig,
    n_cal: int,
    n_train: int,
) -> Dict[str, List[str]]:
    """Load calibration + training texts from an arbitrary HF dataset.

    When *calib* matches the defaults, behaviour is identical to the original
    ``load_text_sets`` (wikitext-2-raw-v1 train).
    """
    _log.info(
        "loading calibration corpus: %s / %s / %s (field=%s, seed=%d)",
        calib.dataset, calib.config, calib.split, calib.text_field, calib.seed,
    )
    ds = load_dataset(calib.dataset, calib.config, split=calib.split)
    pool = _collect_texts(ds, max(n_cal, n_train) * 2, field=calib.text_field)

    if calib.num_samples is not None and calib.num_samples < len(pool):
        pool = _deterministic_sample(pool, calib.num_samples, calib.seed)

    cal = _deterministic_sample(pool, n_cal, calib.seed)
    train = _deterministic_sample(pool, n_train, calib.seed + 1)
    return {"cal": cal, "train": train}


def load_text_sets(
    n_cal: int,
    n_train: int,
    n_eval: int,
    calib: Optional[CalibConfig] = None,
) -> Dict[str, List[str]]:
    """Load cal/train/eval text sets.

    When *calib* is None or uses default settings, this produces the same
    wikitext-based splits as the original hardcoded implementation.

    When a non-default corpus is used, an additional ``eval_workload`` key
    is returned containing held-out texts from the workload corpus.  The
    engine uses this for repair early stopping so the optimization signal
    matches the target distribution.  WikiText eval sets are always
    included for cross-run comparability.
    """
    is_default = (
        calib is None
        or (calib.dataset == DEFAULT_CALIB_DATASET and calib.config == DEFAULT_CALIB_CONFIG)
    )

    if is_default:
        w2_train = load_dataset(
            DEFAULT_CALIB_DATASET, DEFAULT_CALIB_CONFIG, split=DEFAULT_CALIB_SPLIT,
        )
        w2_test = load_dataset(
            DEFAULT_CALIB_DATASET, DEFAULT_CALIB_CONFIG, split="test",
        )
        w103_val = load_dataset(
            DEFAULT_EVAL_DATASET, DEFAULT_EVAL_CONFIG, split=DEFAULT_EVAL_SPLIT,
        )
        result = {
            "cal": _collect_texts(w2_train, n_cal),
            "train": _collect_texts(w2_train, n_train),
            "eval_w2": _collect_texts(w2_test, n_eval),
            "eval_w103": _collect_texts(w103_val, n_eval),
        }
    else:
        assert calib is not None
        corpus = load_calibration_corpus(calib, n_cal, n_train)
        w103_val = load_dataset(
            DEFAULT_EVAL_DATASET, DEFAULT_EVAL_CONFIG, split=DEFAULT_EVAL_SPLIT,
        )
        w2_test = load_dataset(
            DEFAULT_CALIB_DATASET, DEFAULT_CALIB_CONFIG, split="test",
        )
        result = {
            "cal": corpus["cal"],
            "train": corpus["train"],
            "eval_w2": _collect_texts(w2_test, n_eval),
            "eval_w103": _collect_texts(w103_val, n_eval),
        }

        # Build held-out eval set from the workload corpus so repair
        # optimizes for the right distribution, not WikiText.
        try:
            _log.info(
                "loading workload eval split from %s / %s",
                calib.dataset, calib.config,
            )
            eval_split = "validation" if calib.split == "train" else "test"
            try:
                eval_ds = load_dataset(calib.dataset, calib.config, split=eval_split)
            except (ValueError, KeyError):
                eval_ds = load_dataset(calib.dataset, calib.config, split=calib.split)
            workload_eval = _collect_texts(eval_ds, n_eval, field=calib.text_field)
            if len(workload_eval) >= 20:
                result["eval_workload"] = workload_eval
                _log.info("workload eval loaded: %d texts", len(workload_eval))
            else:
                _log.warning(
                    "workload eval too small (%d texts), falling back to train holdout",
                    len(workload_eval),
                )
                holdout = _deterministic_sample(corpus["train"], n_eval, calib.seed + 99)
                result["eval_workload"] = holdout
        except Exception as exc:
            _log.warning("failed to load workload eval: %s — using train holdout", exc)
            holdout = _deterministic_sample(corpus["train"], n_eval, calib.seed + 99)
            result["eval_workload"] = holdout

    parts = ["cal=%d", "train=%d", "eval_w2=%d", "eval_w103=%d"]
    vals = [len(result["cal"]), len(result["train"]),
            len(result["eval_w2"]), len(result["eval_w103"])]
    if "eval_workload" in result:
        parts.append("eval_workload=%d")
        vals.append(len(result["eval_workload"]))
    _log.info("texts loaded: " + " ".join(parts), *vals)
    return result


def deterministic_subset(
    texts: Sequence[str], n: int, seed: int = 0,
) -> List[str]:
    """Select a fixed-size subset of *texts* using a seeded shuffle.

    Stable across runs for the same seed — suitable for cheap eval during
    repair and early stopping.  If n >= len(texts) returns a copy of the
    full list.
    """
    if n >= len(texts):
        return list(texts)
    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    chosen = sorted(indices[:n])
    return [texts[i] for i in chosen]
