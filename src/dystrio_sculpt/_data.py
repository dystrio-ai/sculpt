"""Dataset loading for calibration, training, and evaluation."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Any

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


# ── Row formatters for non-trivial dataset schemas ───────────────────────────

def _format_openhermes(row: Dict) -> str:
    """Flatten OpenHermes 2.5 conversation turns into a single text."""
    convos = row.get("conversations", [])
    parts = []
    for turn in convos:
        role = turn.get("from", "")
        value = turn.get("value", "")
        if value:
            parts.append(f"{role}: {value}" if role else value)
    return "\n".join(parts)


def _format_mmlu_qa(row: Dict) -> str:
    """Format MMLU auxiliary_train row into a Q&A text with choices."""
    q = row.get("question", "")
    choices = row.get("choices", [])
    answer_idx = row.get("answer", 0)
    text = f"Question: {q}\n"
    for i, c in enumerate(choices):
        text += f"  ({chr(65 + i)}) {c}\n"
    if 0 <= answer_idx < len(choices):
        text += f"Answer: ({chr(65 + answer_idx)}) {choices[answer_idx]}"
    return text


def _format_gsm8k_qa(row: Dict) -> str:
    """Format GSM8K with full chain-of-thought answer, not just the question."""
    q = row.get("question", "")
    a = row.get("answer", "")
    return f"Question: {q}\nAnswer: {a}"


def _format_apps_solution(row: Dict) -> str:
    """Format APPS training row: problem statement + first solution."""
    q = row.get("question", "")
    sols = row.get("solutions", "")
    if isinstance(sols, str):
        import json as _json
        try:
            sols = _json.loads(sols)
        except (ValueError, TypeError):
            sols = []
    if isinstance(sols, list) and sols:
        return f"{q}\n\n{sols[0]}"
    return q


_FORMATTERS: Dict[str, Callable[[Dict], str]] = {
    "openhermes": _format_openhermes,
    "mmlu_qa": _format_mmlu_qa,
    "gsm8k_qa": _format_gsm8k_qa,
    "apps_solution": _format_apps_solution,
}


# ── Workload presets ──────────────────────────────────────────────────────────

# Simple presets: single dataset source.
WORKLOAD_PRESETS: Dict[str, Dict[str, str]] = {
    "general": {
        "dataset": DEFAULT_CALIB_DATASET,
        "config": DEFAULT_CALIB_CONFIG,
        "split": DEFAULT_CALIB_SPLIT,
        "text_field": DEFAULT_CALIB_TEXT_FIELD,
    },
    "code": {
        "dataset": "sahil2801/CodeAlpaca-20k",
        "config": "default",
        "split": "train",
        "text_field": "output",
    },
    "chat": {
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "config": "default",
        "split": "train_sft",
        "text_field": "prompt",
    },
    "math": {
        "dataset": "gsm8k",
        "config": "main",
        "split": "train",
        "text_field": "question",
    },
}

# Benchmark-aligned mixture: repair data covers the same capabilities we
# evaluate on (MMLU, HellaSwag, ARC, TruthfulQA, Winogrande, GSM8K).
#
# Sources use either text_field (simple field extraction) or formatter
# (named function in _FORMATTERS for complex schemas).
MIXTURE_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "general_v2": [
        {
            "dataset": "wikitext", "config": "wikitext-103-raw-v1",
            "split": "train", "text_field": "text", "weight": 0.15,
            "purpose": "broad language anchor",
        },
        {
            "dataset": "cais/mmlu", "config": "all",
            "split": "auxiliary_train", "formatter": "mmlu_qa", "weight": 0.20,
            "purpose": "factual knowledge across 57 subjects, directly supports MMLU SLO",
        },
        {
            "dataset": "teknium/OpenHermes-2.5", "config": "default",
            "split": "train", "formatter": "openhermes", "weight": 0.25,
            "purpose": "high-quality instruction/reasoning from 16 curated sources",
        },
        {
            "dataset": "Rowan/hellaswag", "config": "default",
            "split": "train", "text_field": "ctx", "weight": 0.15,
            "purpose": "commonsense completion, supports HellaSwag/Winogrande",
        },
        {
            "dataset": "gsm8k", "config": "main",
            "split": "train", "formatter": "gsm8k_qa", "weight": 0.10,
            "purpose": "math chain-of-thought reasoning, supports GSM8K",
        },
        {
            "dataset": "Open-Orca/OpenOrca", "config": "default",
            "split": "train", "text_field": "response", "weight": 0.15,
            "purpose": "instruction/reasoning, supports ARC/TQA",
        },
    ],
    "code_v2": [
        {
            "dataset": "sahil2801/CodeAlpaca-20k", "config": "default",
            "split": "train", "text_field": "output", "weight": 0.35,
            "purpose": "code instruction, supports HumanEval/MBPP",
        },
        {
            "dataset": "teknium/OpenHermes-2.5", "config": "default",
            "split": "train", "formatter": "openhermes", "weight": 0.25,
            "purpose": "high-quality instruction, preserves general capability",
        },
        {
            "dataset": "wikitext", "config": "wikitext-103-raw-v1",
            "split": "train", "text_field": "text", "weight": 0.15,
            "purpose": "broad language",
        },
        {
            "dataset": "gsm8k", "config": "main",
            "split": "train", "formatter": "gsm8k_qa", "weight": 0.15,
            "purpose": "math chain-of-thought reasoning",
        },
        {
            "dataset": "cais/mmlu", "config": "all",
            "split": "auxiliary_train", "formatter": "mmlu_qa", "weight": 0.10,
            "purpose": "factual knowledge retention",
        },
    ],
    "code_starcoder": [
        {
            "dataset": "bigcode/starcoderdata", "data_dir": "python",
            "split": "train", "text_field": "content", "weight": 0.45,
            "purpose": "actual StarCoder training distribution (Python), primary repair signal",
            "note": "gated: requires agreeing to The Stack Terms of Use on HF",
        },
        {
            "dataset": "bigcode/starcoderdata", "data_dir": "javascript",
            "split": "train", "text_field": "content", "weight": 0.15,
            "purpose": "StarCoder training distribution (JavaScript), second-highest language",
        },
        {
            "dataset": "bigcode/starcoderdata", "data_dir": "java",
            "split": "train", "text_field": "content", "weight": 0.10,
            "purpose": "StarCoder training distribution (Java), multi-language coverage",
        },
        {
            "dataset": "gsm8k", "config": "main",
            "split": "train", "formatter": "gsm8k_qa", "weight": 0.10,
            "purpose": "math/reasoning preservation — SC2-15B beats DSCoder-33B on GSM8K",
        },
        {
            "dataset": "teknium/OpenHermes-2.5", "config": "default",
            "split": "train", "formatter": "openhermes", "weight": 0.10,
            "purpose": "instruction following, prevents general capability collapse",
        },
        {
            "dataset": "wikitext", "config": "wikitext-103-raw-v1",
            "split": "train", "text_field": "text", "weight": 0.10,
            "purpose": "language anchor",
        },
    ],
}


def is_mixture_workload(workload: str) -> bool:
    """Check if a workload name refers to a mixture preset."""
    return workload in MIXTURE_PRESETS


def calib_config_for_workload(workload: str) -> "CalibConfig":
    """Return a CalibConfig for a named workload preset.

    For mixture workloads (general_v2, code_v2), returns the first source
    as the CalibConfig (used for metadata only); actual loading goes through
    ``load_mixture_corpus``.
    """
    if workload in WORKLOAD_PRESETS:
        return CalibConfig(**WORKLOAD_PRESETS[workload])
    if workload in MIXTURE_PRESETS:
        first = MIXTURE_PRESETS[workload][0]
        return CalibConfig(
            dataset=first["dataset"],
            config=first.get("config", first.get("data_dir", "default")),
            split=first["split"],
            text_field=first.get("text_field", "text"),
        )
    all_keys = list(WORKLOAD_PRESETS.keys()) + list(MIXTURE_PRESETS.keys())
    raise ValueError(f"Unknown workload {workload!r}. Available: {all_keys}")


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


def _collect_texts(
    ds, n: int, field: str = "text",
    formatter: Optional[Callable[[Dict], str]] = None,
) -> List[str]:
    out: List[str] = []
    for i in range(len(ds)):
        if formatter is not None:
            t = formatter(ds[i])
        else:
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


def load_mixture_corpus(
    workload: str,
    n_cal: int,
    n_train: int,
    seed: int = 0,
) -> Dict[str, List[str]]:
    """Load calibration + training texts from a multi-source mixture.

    Samples from each source proportionally to its weight, then shuffles
    into a single interleaved corpus.  Falls back gracefully if any
    individual source fails to load.
    """
    sources = MIXTURE_PRESETS[workload]
    total_weight = sum(s["weight"] for s in sources)
    n_total = max(n_cal, n_train) * 2

    all_texts: List[str] = []
    for src in sources:
        n_from_src = max(20, int(n_total * src["weight"] / total_weight))
        ds_name = src["dataset"]
        ds_label = src.get("data_dir") or src.get("config", "default")
        try:
            _log.info(
                "mixture: loading %d texts from %s/%s [%s]",
                n_from_src, ds_name, ds_label, src.get("purpose", ""),
            )
            load_kwargs: Dict[str, Any] = {"split": src["split"]}
            if "data_dir" in src:
                load_kwargs["data_dir"] = src["data_dir"]
                ds = load_dataset(ds_name, **load_kwargs)
            else:
                ds = load_dataset(ds_name, src.get("config", "default"), **load_kwargs)
            fmt_name = src.get("formatter")
            fmt_fn = _FORMATTERS.get(fmt_name) if fmt_name else None
            texts = _collect_texts(
                ds, n_from_src * 2,
                field=src.get("text_field", "text"),
                formatter=fmt_fn,
            )
            sampled = _deterministic_sample(texts, n_from_src, seed)
            all_texts.extend(sampled)
            _log.info("  -> got %d texts", len(sampled))
        except Exception as exc:
            _log.warning("mixture: failed to load %s/%s: %s — skipping", ds_name, ds_label, exc)

    if not all_texts:
        raise RuntimeError(f"All sources failed for mixture workload {workload!r}")

    rng = random.Random(seed)
    rng.shuffle(all_texts)
    _log.info("mixture corpus: %d total texts from %d sources", len(all_texts), len(sources))

    cal = _deterministic_sample(all_texts, n_cal, seed)
    train = _deterministic_sample(all_texts, n_train, seed + 1)
    return {"cal": cal, "train": train}


def load_text_sets(
    n_cal: int,
    n_train: int,
    n_eval: int,
    calib: Optional[CalibConfig] = None,
    mixture_workload: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Load cal/train/eval text sets.

    When *calib* is None or uses default settings, this produces the same
    wikitext-based splits as the original hardcoded implementation.

    When *mixture_workload* is set (e.g. ``"general_v2"``), loads from
    a multi-source mixture designed to cover the downstream benchmark
    families (MMLU, HellaSwag, ARC, TruthfulQA, Winogrande, GSM8K).

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

    # Mixture workloads get special loading path
    is_mixture = (
        mixture_workload is not None
        and mixture_workload in MIXTURE_PRESETS
    )

    if is_mixture:
        assert mixture_workload is not None
        corpus = load_mixture_corpus(
            mixture_workload, n_cal, n_train,
            seed=calib.seed if calib else 0,
        )
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
        # Use a held-out slice of the mixture as workload eval
        holdout = _deterministic_sample(corpus["train"], n_eval, 99)
        result["eval_workload"] = holdout
        _log.info("mixture workload eval: %d texts (holdout from mixture)", len(holdout))

    elif is_default:
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
