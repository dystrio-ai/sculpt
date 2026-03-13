"""Log stage: consolidates compile + bench results into a rich dataset record.

This is the most important stage for building the Dystrio Efficiency Dataset.
It extracts per-layer structural insights, risk profiles, repair dynamics,
and policy decisions — data nobody else captures — and appends them as a
single JSONL record.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


def _detect_environment() -> Dict[str, str]:
    env: Dict[str, str] = {"dtype": "bf16"}
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_version"] = str(torch.version.cuda) if torch.cuda.is_available() else ""
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import transformers
        env["transformers_version"] = transformers.__version__
    except Exception:
        pass
    try:
        from .. import __version__
        env["dystrio_version"] = __version__
    except Exception:
        env["dystrio_version"] = "1.0.0"
    try:
        import subprocess
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        env["git_sha"] = sha
    except Exception:
        pass
    return env


def _load_bench_for_model(bench_csv: Path, model_path: str) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    if not bench_csv.exists():
        return results
    with open(bench_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model_id") == model_path:
                results[row["workload"]] = row
    return results


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


def _extract_layer_insights(compile_report: Dict[str, Any]) -> list:
    """Extract per-layer structural insights from compile_report.

    The compile_report has keys like "0", "1", ... for each layer,
    each containing kept_blocks, original_ffn, ffn_kept, keep_frac.
    """
    from ..dataset.schema import LayerInsight

    insights = []
    for key, data in compile_report.items():
        if not key.isdigit():
            continue
        li = LayerInsight(
            layer_idx=int(key),
            original_ffn=data.get("original_ffn", 0),
            ffn_kept=data.get("ffn_kept", 0),
            keep_frac=data.get("keep_frac", 0.0),
            kept_blocks=data.get("kept_blocks", 0),
        )
        insights.append(li)
    insights.sort(key=lambda x: x.layer_idx)
    return insights


def _extract_risk_profile(compile_result) -> "RiskProfile":
    """Extract model-level risk profile from the CompileStageResult.

    risk_detail comes from search.risk_detail (stored on the
    CompileStageResult directly) — it contains per-layer risk scores
    from the prescan/risk analysis phase.
    """
    from ..dataset.schema import RiskProfile

    profile = RiskProfile()
    profile.aggregate_risk = getattr(compile_result, "risk_score", 0.0)

    risk_detail = getattr(compile_result, "risk_detail", {}) or {}
    if risk_detail:
        layer_risks = []
        high = []
        low = []
        for k, v in risk_detail.items():
            if k.isdigit() and isinstance(v, dict):
                score = _safe_float(v.get("risk_score", v.get("score", 0.0)))
                layer_risks.append(score)
                if score > 0.6:
                    high.append(int(k))
                elif score < 0.3:
                    low.append(int(k))
        if layer_risks:
            profile.layer_risks = layer_risks
            profile.high_risk_layers = sorted(high)
            profile.low_risk_layers = sorted(low)

    # Layer ordering (compressibility order) comes from risk detail or
    # from the first tier's config.
    if risk_detail.get("layer_order"):
        profile.compressibility_order = risk_detail["layer_order"]
    elif hasattr(compile_result, "tiers") and compile_result.tiers:
        cfg = getattr(compile_result.tiers[0], "config", {}) or {}
        if cfg.get("layer_order"):
            profile.compressibility_order = cfg["layer_order"]

    return profile


def _extract_policy_trace(tier) -> "PolicyTrace":
    """Extract policy selection and escalation trace from a tier.

    The engine config dict (tier.config) contains the policy dict,
    escalation details, and stage_stats. The policy_name is a direct
    field on TierResult.
    """
    from ..dataset.schema import PolicyTrace

    trace = PolicyTrace()

    cfg = getattr(tier, "config", {}) or {}
    if isinstance(cfg, dict):
        esc = cfg.get("escalation", {})
        if isinstance(esc, dict):
            trace.escalation_applied = bool(esc.get("applied", False))
            trace.escalation_details = str(esc.get("details", ""))

        policy_dict = cfg.get("policy", {})
        if isinstance(policy_dict, dict):
            trace.initial_policy = policy_dict.get("name", "")
            trace.final_policy = policy_dict.get("name", "")
            trace.total_repair_steps = _safe_int(cfg.get("total_repair_steps"))
            trace.stage_size = _safe_int(cfg.get("stage_size"))

    if hasattr(tier, "policy_name") and tier.policy_name:
        trace.final_policy = tier.policy_name

    return trace


def run_log_stage(
    compile_result,
    descriptor,
    *,
    run_id: Optional[str] = None,
    benchmark_csv: Optional[Path] = None,
    dataset_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build and log a rich dataset record from compile + bench results.

    Extracts per-layer insights, risk profiles, policy decisions, and
    repair dynamics. This is the data that makes the Dystrio Efficiency
    Dataset a proprietary asset.
    """
    from ..dataset.schema import (
        DatasetRecord, TierRecord, BaselineInfo, EnvironmentInfo,
        LayerInsight, RiskProfile, PolicyTrace,
    )
    from ..dataset.logger import DatasetLogger

    env_info = _detect_environment()

    kwargs = {}
    if run_id:
        kwargs["run_id"] = run_id

    record = DatasetRecord(
        model_id=compile_result.model_id,
        source="factory",
        source_repo=compile_result.model_id,
        architecture=descriptor.to_dict() if hasattr(descriptor, "to_dict") else {},
        **kwargs,
        optimization_config={
            "frontier": len(compile_result.tiers),
            "deterministic": True,
            "selector": "structural",
            "policy": "auto",
        },
        environment=EnvironmentInfo(**{
            k: v for k, v in env_info.items()
            if k in EnvironmentInfo.__dataclass_fields__
        }),
    )

    # ── Baseline ──────────────────────────────────────────────────────────
    bm = compile_result.baseline_metrics or {}
    if compile_result.tiers:
        first = compile_result.tiers[0]
        record.baseline = BaselineInfo(
            ppl_wikitext=bm.get("ppl_w103_valid", 0.0),
            prefill_tps=bm.get("prefill_tokens_per_sec", 0.0),
            decode_tps=bm.get("decode_tokens_per_sec", 0.0),
            ttft_p95_ms=bm.get("prefill_latency_ms_p95", 0.0),
            weights_gb=round((first.baseline_weights_bytes or 0) / 1e9, 2),
            num_params=first.baseline_num_params or 0,
        )

    # ── Risk profile (model-level) ────────────────────────────────────────
    if compile_result.tiers:
        rp = _extract_risk_profile(compile_result)
        if isinstance(rp, RiskProfile):
            record.risk_profile = rp
        elif isinstance(rp, dict):
            record.risk_profile = RiskProfile(**{
                k: v for k, v in rp.items()
                if k in RiskProfile.__dataclass_fields__
            })

    # ── Policy trace ──────────────────────────────────────────────────────
    if compile_result.tiers:
        pt = _extract_policy_trace(compile_result.tiers[0])
        if isinstance(pt, PolicyTrace):
            record.policy_trace = pt
        elif isinstance(pt, dict):
            record.policy_trace = PolicyTrace(**{
                k: v for k, v in pt.items()
                if k in PolicyTrace.__dataclass_fields__
            })

    # ── Per-tier records with per-layer insights ──────────────────────────
    for tier in compile_result.tiers:
        bench_data: Dict[str, Dict[str, Any]] = {}
        if benchmark_csv is not None:
            bench_data = _load_bench_for_model(
                Path(benchmark_csv), str(tier.model_dir),
            )

        chat = bench_data.get("chat", {})
        rag = bench_data.get("rag", {})
        wiki = bench_data.get("wikitext", {})

        # Per-layer insights from compile report
        layer_insights = []
        if tier.compile_report:
            layer_insights = _extract_layer_insights(tier.compile_report)

        tier_cfg = getattr(tier, "config", {}) or {}
        stage_stats = tier_cfg.get("stage_stats", []) if isinstance(tier_cfg, dict) else []
        total_repair = sum(s.get("repair_steps", 0) for s in stage_stats if isinstance(s, dict))

        prefill_tps = _safe_float(chat.get("prefill_tokens_per_sec"))
        decode_tps = _safe_float(chat.get("decode_tokens_per_sec"))
        base_prefill = _safe_float(bm.get("prefill_tokens_per_sec"))
        base_decode = _safe_float(bm.get("decode_tokens_per_sec"))

        tier_record = TierRecord(
            name=tier.label.split("_")[-1] if "_" in tier.label else tier.label,
            keep_frac=tier.keep_frac,
            ppl_ratio=tier.ppl_ratio,
            prefill_tps=prefill_tps,
            ttft_p95_ms=_safe_float(rag.get("ttft_ms_p95")),
            decode_tps=decode_tps,
            weights_gb=_safe_float(wiki.get("weights_gb") or chat.get("weights_gb")),
            num_params=_safe_int(wiki.get("num_params") or chat.get("num_params")),
            wall_time_s=tier.wall_time_s,
            total_repair_steps=total_repair,
            layers_compressed=len(layer_insights),
            risk_score=tier.risk_score,
            layer_insights=layer_insights,
            artifact_url="",
            prefill_speedup=round(prefill_tps / base_prefill, 3) if base_prefill > 0 else 0.0,
            decode_speedup=round(decode_tps / base_decode, 3) if base_decode > 0 else 0.0,
            early_stopped=getattr(tier, "early_stopped", False),
            guardrail_triggered=getattr(tier, "guardrail_failed", False),
        )

        record.tiers.append(tier_record)

    # ── Decision trace (full compile reports + configs for deep analysis) ──
    traces = []
    for tier in compile_result.tiers:
        trace_obj = {}
        if tier.compile_report:
            trace_obj["compile_report"] = tier.compile_report
        tier_cfg = getattr(tier, "config", {})
        if tier_cfg:
            trace_obj["config"] = tier_cfg
        if trace_obj:
            traces.append(json.dumps(trace_obj, default=str))
    if traces:
        record.decision_trace = "\n---\n".join(traces)

    # ── Log it ────────────────────────────────────────────────────────────
    logger = DatasetLogger(dataset_path)
    logger.log(record)

    _log.info(
        "logged dataset record: %s — %d tiers, %d total layer insights",
        compile_result.model_id,
        len(record.tiers),
        sum(len(t.layer_insights) for t in record.tiers),
    )

    return record.to_dict()
