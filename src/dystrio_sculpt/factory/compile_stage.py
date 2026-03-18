"""Compile stage: wraps FrontierSearch + emit into a single callable.

Takes a model_id and architecture adapter, runs the full Sculpt pipeline,
and writes tier outputs to disk.  Returns per-tier metadata for downstream
stages (bench, publish, log).
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


@dataclass
class TierResult:
    label: str
    keep_frac: float
    ppl_ratio: float
    point_dir: Path
    model_dir: Path
    metrics: Dict[str, float] = field(default_factory=dict)
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    compile_report: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    num_params: Optional[int] = None
    weights_bytes: Optional[int] = None
    baseline_num_params: Optional[int] = None
    baseline_weights_bytes: Optional[int] = None
    wall_time_s: float = 0.0
    risk_score: float = 0.0
    policy_name: str = ""
    early_stopped: bool = False
    guardrail_failed: bool = False


@dataclass
class CompileStageResult:
    model_id: str
    outdir: Path
    tiers: List[TierResult]
    risk_score: float = 0.0
    risk_detail: Dict[str, Any] = field(default_factory=dict)
    baseline_metrics: Dict[str, float] = field(default_factory=dict)


def run_compile_stage(
    model_id: str,
    outdir: Path,
    *,
    frontier: int = 4,
    deterministic: bool = True,
    device: str = "cuda",
    dtype_str: str = "bf16",
    adapter=None,
    max_ppl_multiplier: Optional[float] = None,
    speed_profile: Optional[str] = None,
    use_risk_schedule: bool = False,
    protection_threshold: Optional[float] = None,
    dry_run: bool = False,
) -> CompileStageResult:
    """Run the full sculpt compile pipeline for a model.

    When *dry_run* is True, only fingerprints the model and returns an
    empty CompileStageResult without running compilation.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        _log.info("[dry-run] compile stage for %s — skipping actual compilation", model_id)
        return CompileStageResult(model_id=model_id, outdir=outdir, tiers=[])

    from ..search import FrontierSearch
    from ..emit import emit_frontier_point, emit_run_metadata
    from ..validate import validate_saved_model

    import torch

    emit_run_metadata(outdir, {
        "deterministic": deterministic,
        "seed": 0,
        "dtype": dtype_str,
    })

    search = FrontierSearch(
        model_id=model_id,
        n_frontier=frontier,
        max_ppl_multiplier=max_ppl_multiplier,
        deterministic=deterministic,
        device=device,
        dtype_str=dtype_str,
        outdir=outdir,
        speed_profile=speed_profile,
        use_risk_schedule=use_risk_schedule,
        protection_threshold=protection_threshold,
        adapter=adapter,
    )

    selected = search.run()
    if not selected:
        _log.warning("no viable frontier points for %s", model_id)
        return CompileStageResult(
            model_id=model_id, outdir=outdir, tiers=[],
            risk_score=getattr(search, "risk_score", 0.0),
            risk_detail=getattr(search, "risk_detail", None) or {},
            baseline_metrics=search.baseline_metrics or {},
        )

    tiers: List[TierResult] = []
    for pt in selected:
        cr = pt.compile_result
        if cr is None or cr.model is None:
            _log.warning("frontier point %s has no model — skipping", pt.label)
            continue

        point_dir = emit_frontier_point(
            model=cr.model,
            tokenizer=cr.tokenizer,
            outdir=outdir,
            label=pt.label,
            keep_frac=pt.keep_frac,
            metrics=cr.metrics_post,
            baseline_metrics=cr.baseline_metrics,
            compile_report=cr.compile_report,
            config=cr.config,
            wall_time_s=cr.wall_time_s,
            pilot_report=cr.pilot_report,
            risk_score=pt.risk_score,
            peak_cuda_allocated_compile_bytes=cr.peak_cuda_allocated_compile_bytes,
            peak_cuda_reserved_compile_bytes=cr.peak_cuda_reserved_compile_bytes,
            peak_cuda_allocated_bench_bytes=cr.peak_cuda_allocated_bench_bytes,
            peak_cuda_reserved_bench_bytes=cr.peak_cuda_reserved_bench_bytes,
            cuda_allocated_end_bytes=cr.cuda_allocated_end_bytes,
            cuda_reserved_end_bytes=cr.cuda_reserved_end_bytes,
            num_params=cr.num_params,
            weights_bytes=cr.weights_bytes,
            baseline_num_params=cr.baseline_num_params,
            baseline_weights_bytes=cr.baseline_weights_bytes,
        )

        model_dir = point_dir / "model"
        ok = validate_saved_model(model_dir, device=device)
        if not ok:
            _log.error("validation FAILED for %s — skipping tier", pt.label)
            continue

        base_ppl = (cr.baseline_metrics or {}).get("ppl_w103_valid", 1.0)
        ppl_ratio = pt.ppl_w103 / max(1e-9, base_ppl)

        tiers.append(TierResult(
            label=pt.label,
            keep_frac=pt.keep_frac,
            ppl_ratio=ppl_ratio,
            point_dir=point_dir,
            model_dir=model_dir,
            metrics=cr.metrics_post,
            baseline_metrics=cr.baseline_metrics,
            compile_report=cr.compile_report,
            config=cr.config,
            num_params=cr.num_params,
            weights_bytes=cr.weights_bytes,
            baseline_num_params=cr.baseline_num_params,
            baseline_weights_bytes=cr.baseline_weights_bytes,
            wall_time_s=cr.wall_time_s,
            risk_score=pt.risk_score,
            policy_name=cr.policy_name or "",
            early_stopped=cr.early_stopped,
            guardrail_failed=cr.guardrail_failed,
        ))

        del cr.model
        cr.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = CompileStageResult(
        model_id=model_id,
        outdir=outdir,
        tiers=tiers,
        risk_score=getattr(search, "risk_score", 0.0),
        risk_detail=getattr(search, "risk_detail", None) or {},
        baseline_metrics=search.baseline_metrics or {},
    )
    _log.info("compile stage complete: %d tiers for %s", len(tiers), model_id)
    return result
