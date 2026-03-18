"""Dystrio Efficiency Dataset: the canonical record of every optimization run.

Every sculpt run, benchmark, and downstream eval produces a structured record
that flows into this dataset.  The schema follows the product guidance:

    1. Model fingerprint   — architecture, sizes, layer count, modality
    2. Optimization actions — keep_frac, policy, distillation, selector, per-layer decisions
    3. Environment          — GPU, runtime, torch version, dtype, batch sizes
    4. Outcomes             — PPL, benchmarks, latency, throughput, VRAM, cost
    5. Decision trace       — why this config was chosen, alternatives rejected

The dataset is append-only and pushed to HuggingFace after each sculpt run.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

HF_DATASET_REPO = "dystrio/efficiency-dataset"
LOCAL_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "efficiency_dataset.jsonl"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_record(
    *,
    # 1. Model fingerprint
    model_id: str,
    model_family: Optional[str] = None,
    architecture: Optional[str] = None,
    num_hidden_layers: Optional[int] = None,
    hidden_size: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    num_key_value_heads: Optional[int] = None,
    vocab_size: Optional[int] = None,
    max_position_embeddings: Optional[int] = None,
    num_params_original: Optional[int] = None,
    modality: str = "text",

    # 2. Optimization actions
    method: str = "sculpt",
    tier: Optional[str] = None,
    keep_frac: Optional[float] = None,
    policy_name: Optional[str] = None,
    policy_config: Optional[Dict[str, Any]] = None,
    selector: str = "structural",
    distillation_enabled: bool = False,
    distill_alpha: float = 0.0,
    distill_temp: float = 2.0,
    calib_dataset: str = "wikitext",
    calib_config: str = "wikitext-2-raw-v1",
    calib_split: str = "train",
    repair_steps_total: Optional[int] = None,
    num_stages: Optional[int] = None,
    stage_stats: Optional[List[Dict]] = None,
    intermediate_size_compressed: Optional[int] = None,
    num_params_compressed: Optional[int] = None,
    layers_compressed: Optional[int] = None,

    # 3. Environment
    gpu_name: Optional[str] = None,
    gpu_memory_gb: Optional[float] = None,
    torch_version: Optional[str] = None,
    transformers_version: Optional[str] = None,
    cuda_version: Optional[str] = None,
    dtype: str = "bf16",
    dystrio_version: Optional[str] = None,
    git_sha: Optional[str] = None,

    # 4. Outcomes
    ppl_w2: Optional[float] = None,
    ppl_w103: Optional[float] = None,
    ppl_ratio: Optional[float] = None,
    baseline_ppl_w103: Optional[float] = None,
    prefill_speedup: Optional[float] = None,
    decode_speedup: Optional[float] = None,
    prefill_tps: Optional[float] = None,
    decode_tps: Optional[float] = None,
    prefill_ms_p50: Optional[float] = None,
    prefill_ms_p95: Optional[float] = None,
    prefill_ms_p99: Optional[float] = None,
    decode_ms_per_tok_p50: Optional[float] = None,
    decode_ms_per_tok_p95: Optional[float] = None,
    decode_ms_per_tok_p99: Optional[float] = None,
    baseline_prefill_ms_p95: Optional[float] = None,
    baseline_decode_ms_per_tok_p95: Optional[float] = None,
    e2e_speedup_chat: Optional[float] = None,
    e2e_speedup_rag: Optional[float] = None,
    e2e_speedup_batch: Optional[float] = None,
    weights_gb: Optional[float] = None,
    baseline_weights_gb: Optional[float] = None,
    weights_memory_reduction_pct: Optional[float] = None,
    peak_compile_alloc_gb: Optional[float] = None,
    peak_bench_alloc_gb: Optional[float] = None,
    steady_state_alloc_gb: Optional[float] = None,
    compile_time_s: Optional[float] = None,
    risk_score: Optional[float] = None,
    # Downstream benchmarks (lm-eval)
    arc_challenge_acc_norm: Optional[float] = None,
    hellaswag_acc_norm: Optional[float] = None,
    mmlu_acc: Optional[float] = None,
    truthfulqa_mc2_acc: Optional[float] = None,
    eval_engine: Optional[str] = None,
    eval_date: Optional[str] = None,

    # 5. Decision trace
    search_candidates: Optional[List[float]] = None,
    search_ceiling: Optional[float] = None,
    search_risk_score: Optional[float] = None,
    escalation_applied: bool = False,
    guardrail_failed: bool = False,
    failure_reason: Optional[str] = None,
    pilot_report: Optional[Dict[str, Any]] = None,

    # Meta
    record_type: str = "sculpt_run",
    run_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a single canonical dataset record.

    All fields are optional except model_id. Unknown fields passed via
    **extra are preserved under an 'extra' key.
    """
    record: Dict[str, Any] = {
        "record_type": record_type,
        "timestamp": timestamp or _now_utc(),
        "run_id": run_id,

        "model_id": model_id,
        "model_family": model_family,
        "architecture": architecture,
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_position_embeddings,
        "num_params_original": num_params_original,
        "modality": modality,

        "method": method,
        "tier": tier,
        "keep_frac": keep_frac,
        "policy_name": policy_name,
        "policy_config": policy_config,
        "selector": selector,
        "distillation_enabled": distillation_enabled,
        "distill_alpha": distill_alpha,
        "distill_temp": distill_temp,
        "calib_dataset": calib_dataset,
        "calib_config": calib_config,
        "calib_split": calib_split,
        "repair_steps_total": repair_steps_total,
        "num_stages": num_stages,
        "stage_stats": stage_stats,
        "intermediate_size_compressed": intermediate_size_compressed,
        "num_params_compressed": num_params_compressed,
        "layers_compressed": layers_compressed,

        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "cuda_version": cuda_version,
        "dtype": dtype,
        "dystrio_version": dystrio_version,
        "git_sha": git_sha,

        "ppl_w2": ppl_w2,
        "ppl_w103": ppl_w103,
        "ppl_ratio": ppl_ratio,
        "baseline_ppl_w103": baseline_ppl_w103,
        "prefill_speedup": prefill_speedup,
        "decode_speedup": decode_speedup,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "prefill_ms_p50": prefill_ms_p50,
        "prefill_ms_p95": prefill_ms_p95,
        "prefill_ms_p99": prefill_ms_p99,
        "decode_ms_per_tok_p50": decode_ms_per_tok_p50,
        "decode_ms_per_tok_p95": decode_ms_per_tok_p95,
        "decode_ms_per_tok_p99": decode_ms_per_tok_p99,
        "baseline_prefill_ms_p95": baseline_prefill_ms_p95,
        "baseline_decode_ms_per_tok_p95": baseline_decode_ms_per_tok_p95,
        "e2e_speedup_chat": e2e_speedup_chat,
        "e2e_speedup_rag": e2e_speedup_rag,
        "e2e_speedup_batch": e2e_speedup_batch,
        "weights_gb": weights_gb,
        "baseline_weights_gb": baseline_weights_gb,
        "weights_memory_reduction_pct": weights_memory_reduction_pct,
        "peak_compile_alloc_gb": peak_compile_alloc_gb,
        "peak_bench_alloc_gb": peak_bench_alloc_gb,
        "steady_state_alloc_gb": steady_state_alloc_gb,
        "compile_time_s": compile_time_s,
        "risk_score": risk_score,
        "arc_challenge_acc_norm": arc_challenge_acc_norm,
        "hellaswag_acc_norm": hellaswag_acc_norm,
        "mmlu_acc": mmlu_acc,
        "truthfulqa_mc2_acc": truthfulqa_mc2_acc,
        "eval_engine": eval_engine,
        "eval_date": eval_date,

        "search_candidates": search_candidates,
        "search_ceiling": search_ceiling,
        "search_risk_score": search_risk_score,
        "escalation_applied": escalation_applied,
        "guardrail_failed": guardrail_failed,
        "failure_reason": failure_reason,
        "pilot_report": pilot_report,
    }

    if extra:
        record["extra"] = extra

    return record


def record_from_frontier_point(
    frontier_point,
    compile_result,
    baseline_metrics: Dict[str, float],
    search_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a dataset record from a FrontierSearch result.

    Called automatically by the emit pipeline after each frontier point.
    """
    import torch
    from . import __version__

    cr = compile_result
    cfg = cr.config if cr else {}
    policy_cfg = cfg.get("policy", {})
    distill_cfg = cfg.get("distillation", {})
    metrics = cr.metrics_post if cr else {}

    model_config = None
    if cr and cr.model is not None:
        try:
            from ._model import get_text_config
            model_config = get_text_config(cr.model)
        except Exception:
            pass

    gpu_name = None
    gpu_mem = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name()
            gpu_mem = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
        except Exception:
            pass

    base_ppl = baseline_metrics.get("ppl_w103_valid", 1.0)
    base_prefill = baseline_metrics.get("prefill_tokens_per_sec", 1.0)
    base_decode = baseline_metrics.get("decode_tokens_per_sec", 1.0)

    return build_record(
        record_type="sculpt_run",
        model_id=cfg.get("model_id", ""),
        architecture=type(cr.model).__name__ if cr and cr.model else None,
        num_hidden_layers=model_config.num_hidden_layers if model_config else cfg.get("num_layers"),
        hidden_size=getattr(model_config, "hidden_size", None) if model_config else None,
        intermediate_size=getattr(model_config, "intermediate_size", None) if model_config else None,
        num_attention_heads=getattr(model_config, "num_attention_heads", None) if model_config else None,
        num_key_value_heads=getattr(model_config, "num_key_value_heads", None) if model_config else None,
        vocab_size=getattr(model_config, "vocab_size", None) if model_config else None,
        num_params_original=cr.baseline_num_params if cr else None,

        tier=frontier_point.label,
        keep_frac=frontier_point.keep_frac,
        policy_name=policy_cfg.get("name"),
        policy_config=policy_cfg,
        selector=cfg.get("selector", "structural"),
        distillation_enabled=distill_cfg.get("enabled", False),
        distill_alpha=distill_cfg.get("alpha", 0.0),
        calib_dataset=cfg.get("calib_dataset", "wikitext"),
        calib_config=cfg.get("calib_config", "wikitext-2-raw-v1"),
        repair_steps_total=cfg.get("total_repair_steps"),
        num_stages=len(cfg.get("stage_stats", [])),
        stage_stats=cfg.get("stage_stats"),
        num_params_compressed=cr.num_params if cr else None,
        layers_compressed=cfg.get("layers_compressed"),

        gpu_name=gpu_name,
        gpu_memory_gb=gpu_mem,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
        dtype=cfg.get("dtype", "bf16"),
        dystrio_version=__version__,

        ppl_w2=metrics.get("ppl_w2_test"),
        ppl_w103=metrics.get("ppl_w103_valid"),
        ppl_ratio=metrics.get("ppl_w103_valid", 0) / max(1e-9, base_ppl),
        baseline_ppl_w103=base_ppl,
        prefill_speedup=frontier_point.prefill_speedup,
        decode_speedup=frontier_point.decode_speedup,
        prefill_tps=metrics.get("prefill_tokens_per_sec"),
        decode_tps=metrics.get("decode_tokens_per_sec"),
        prefill_ms_p50=metrics.get("prefill_latency_ms_p50"),
        prefill_ms_p95=metrics.get("prefill_latency_ms_p95"),
        prefill_ms_p99=metrics.get("prefill_latency_ms_p99"),
        decode_ms_per_tok_p50=metrics.get("decode_ms_per_token_p50"),
        decode_ms_per_tok_p95=metrics.get("decode_ms_per_token_p95"),
        decode_ms_per_tok_p99=metrics.get("decode_ms_per_token_p99"),
        baseline_prefill_ms_p95=baseline_metrics.get("prefill_latency_ms_p95"),
        baseline_decode_ms_per_tok_p95=baseline_metrics.get("decode_ms_per_token_p95"),
        weights_gb=round(cr.weights_bytes / (1024**3), 3) if cr and cr.weights_bytes else None,
        baseline_weights_gb=round(cr.baseline_weights_bytes / (1024**3), 3) if cr and cr.baseline_weights_bytes else None,
        peak_compile_alloc_gb=round(cr.peak_cuda_allocated_compile_bytes / (1024**3), 2) if cr and cr.peak_cuda_allocated_compile_bytes else None,
        peak_bench_alloc_gb=round(cr.peak_cuda_allocated_bench_bytes / (1024**3), 2) if cr and cr.peak_cuda_allocated_bench_bytes else None,
        steady_state_alloc_gb=round(cr.cuda_allocated_end_bytes / (1024**3), 2) if cr and cr.cuda_allocated_end_bytes else None,
        compile_time_s=cr.wall_time_s if cr else None,
        risk_score=frontier_point.risk_score,

        search_candidates=search_meta.get("candidates") if search_meta else None,
        search_ceiling=search_meta.get("ceiling") if search_meta else None,
        search_risk_score=search_meta.get("risk_score") if search_meta else None,
        escalation_applied=cr.escalation_applied if cr else False,
        guardrail_failed=cr.guardrail_failed if cr else False,
        failure_reason=cr.failure.get("reason") if cr and cr.failure else None,
        pilot_report=cr.pilot_report if cr else None,
    )


# ── Local cache ───────────────────────────────────────────────────────────────

def append_local(record: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Append a record to the local JSONL cache."""
    p = path or LOCAL_CACHE
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    _log.debug("appended record to %s", p)
    return p


def load_local(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load all records from the local JSONL cache."""
    p = path or LOCAL_CACHE
    if not p.exists():
        return []
    records = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── HuggingFace push ──────────────────────────────────────────────────────────

def push_to_hub(
    records: Optional[List[Dict[str, Any]]] = None,
    repo_id: str = HF_DATASET_REPO,
    private: bool = True,
    token: Optional[str] = None,
) -> str:
    """Push records to HuggingFace as a dataset.

    If *records* is None, loads from the local cache.
    Returns the dataset URL.
    """
    from datasets import Dataset, DatasetDict

    if records is None:
        records = load_local()

    if not records:
        raise ValueError("No records to push")

    token = token or os.environ.get("HF_TOKEN")

    ds = Dataset.from_list(records)
    ds_dict = DatasetDict({"optimization_runs": ds})
    ds_dict.push_to_hub(repo_id, private=private, token=token)

    url = f"https://huggingface.co/datasets/{repo_id}"
    _log.info("pushed %d records to %s", len(records), url)
    return url


def push_record(
    record: Dict[str, Any],
    repo_id: str = HF_DATASET_REPO,
    token: Optional[str] = None,
) -> None:
    """Append a single record locally and push the full dataset.

    This is the function called from emit.py after each frontier point.
    """
    append_local(record)
    try:
        push_to_hub(repo_id=repo_id, token=token)
    except Exception as exc:
        _log.warning("HF push failed (record saved locally): %s", exc)
