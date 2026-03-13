"""Backfill: ingest existing benchmark CSVs into the Dystrio Efficiency Dataset.

Reads benchmark CSVs from data/bench_{family}/ directories and creates
dataset records for models that were compiled before the factory existed.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schema import DatasetRecord, TierRecord, BaselineInfo, EnvironmentInfo
from .logger import DatasetLogger

_log = logging.getLogger(__name__)

KNOWN_MODELS = [
    {
        "base_model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "family": "mistral",
        "bench_dir": "bench_mistral",
        "compile_dir": "mistral-7b-instruct-f5",
    },
    {
        "base_model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "family": "llama",
        "bench_dir": "bench_llama",
        "compile_dir": "llama-3.1-8b-instruct-f4",
    },
    {
        "base_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "family": "qwen",
        "bench_dir": "bench_qwen",
        "compile_dir": "qwen2.5-7b-instruct-f3",
    },
]


def _load_bench(bench_csv: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load benchmarks.csv into {model_id: {workload: row}}."""
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    with open(bench_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["model_id"]
            wl = row["workload"]
            if mid not in results:
                results[mid] = {}
            results[mid][wl] = row
    return results


def _extract_tier_name(model_path: str) -> str:
    """Extract tier name from model path like '/data/zoo/.../frontier_0_default/model'."""
    m = re.search(r"frontier_\d+_(\w+)", model_path)
    if m:
        return m.group(1)
    return "unknown"


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v) if v else default
    except (ValueError, TypeError):
        return default


def _safe_int(v, default=0) -> int:
    try:
        return int(v) if v else default
    except (ValueError, TypeError):
        return default


def _build_record_from_bench(
    base_model_id: str,
    family: str,
    bench: Dict[str, Dict[str, Dict[str, Any]]],
) -> Optional[DatasetRecord]:
    """Build a DatasetRecord from benchmark data for one model family."""

    # Extract baseline
    base_wiki = bench.get(base_model_id, {}).get("wikitext", {})
    base_chat = bench.get(base_model_id, {}).get("chat", {})
    if not base_wiki and not base_chat:
        _log.warning("no baseline data for %s", base_model_id)
        return None

    base_rag = bench.get(base_model_id, {}).get("rag", {})
    baseline = BaselineInfo(
        ppl_wikitext=_safe_float(base_wiki.get("ppl_wikitext")),
        prefill_tps=_safe_float(base_chat.get("prefill_tokens_per_sec")),
        decode_tps=_safe_float(base_chat.get("decode_tokens_per_sec")),
        ttft_p95_ms=_safe_float(base_rag.get("ttft_ms_p95")),
        weights_gb=_safe_float(base_wiki.get("weights_gb") or base_chat.get("weights_gb")),
        num_params=_safe_int(base_wiki.get("num_params") or base_chat.get("num_params")),
    )

    # Extract tiers (non-baseline model paths)
    tier_records: List[TierRecord] = []
    for model_path, workloads in bench.items():
        if model_path == base_model_id:
            continue

        tier_name = _extract_tier_name(model_path)
        wiki = workloads.get("wikitext", {})
        chat = workloads.get("chat", {})
        rag = workloads.get("rag", {})

        ppl_ratio = _safe_float(wiki.get("ppl_ratio") or chat.get("ppl_ratio"))

        # Estimate keep_frac from weight ratio
        tier_weights = _safe_float(wiki.get("weights_gb") or chat.get("weights_gb"))
        base_weights = baseline.weights_gb
        keep_frac = round(tier_weights / base_weights, 3) if base_weights > 0 else 0.0

        hf_repo = f"dystrio/{base_model_id.split('/')[-1]}-sculpt-{tier_name}"

        tier_prefill = _safe_float(chat.get("prefill_tokens_per_sec"))
        tier_decode = _safe_float(chat.get("decode_tokens_per_sec"))
        prefill_speedup = round(tier_prefill / baseline.prefill_tps, 3) if baseline.prefill_tps > 0 else 0.0
        decode_speedup = round(tier_decode / baseline.decode_tps, 3) if baseline.decode_tps > 0 else 0.0

        tier_records.append(TierRecord(
            name=tier_name,
            keep_frac=keep_frac,
            ppl_ratio=ppl_ratio,
            prefill_tps=tier_prefill,
            ttft_p95_ms=_safe_float(rag.get("ttft_ms_p95")),
            decode_tps=tier_decode,
            weights_gb=tier_weights,
            num_params=_safe_int(wiki.get("num_params") or chat.get("num_params")),
            prefill_speedup=prefill_speedup,
            decode_speedup=decode_speedup,
            artifact_url=f"https://huggingface.co/{hf_repo}",
        ))

    if not tier_records:
        _log.warning("no tier data for %s", base_model_id)
        return None

    # Try to fingerprint for architecture info
    arch_dict = {"family": family}
    try:
        from ..architectures import fingerprint
        desc = fingerprint(base_model_id)
        arch_dict = desc.to_dict()
    except Exception:
        pass

    record = DatasetRecord(
        model_id=base_model_id,
        source="backfill",
        source_repo=base_model_id,
        architecture=arch_dict,
        optimization_config={
            "frontier": len(tier_records),
            "deterministic": True,
            "selector": "structural",
            "policy": "auto",
        },
        environment=EnvironmentInfo(
            gpu="NVIDIA A100-SXM4-80GB",
            dtype="bf16",
            torch_version="2.10.0+cu128",
            transformers_version="5.3.0",
            dystrio_version="1.0.0",
        ),
        tiers=tier_records,
        baseline=baseline,
        decision_trace="backfilled from existing benchmark CSVs",
    )
    return record


def run_backfill(
    data_dir: str = "data",
    dataset_path: Optional[str] = None,
) -> int:
    """Backfill the dataset from existing benchmark CSVs.

    Returns number of records added.
    """
    data = Path(data_dir)
    logger = DatasetLogger(dataset_path)
    count = 0

    for model_info in KNOWN_MODELS:
        bench_csv = data / model_info["bench_dir"] / "benchmarks.csv"
        if not bench_csv.exists():
            _log.warning("skip %s: %s not found", model_info["base_model_id"], bench_csv)
            continue

        _log.info("backfilling %s from %s", model_info["base_model_id"], bench_csv)
        bench = _load_bench(bench_csv)

        record = _build_record_from_bench(
            model_info["base_model_id"],
            model_info["family"],
            bench,
        )
        if record is not None:
            logger.log(record)
            count += 1
            _log.info(
                "  added %s: %d tiers",
                model_info["base_model_id"], len(record.tiers),
            )

    _log.info("backfill complete: %d records added", count)
    return count
