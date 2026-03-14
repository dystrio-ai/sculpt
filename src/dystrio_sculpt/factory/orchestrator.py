"""Factory orchestrator: chains fingerprint → compile → bench → publish → log.

This is the central pipeline runner. Each stage is optional and controlled
via flags so the orchestrator can be used for dry runs, partial reruns, etc.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


@dataclass
class FactoryConfig:
    model_id: str
    org: str = "dystrio"
    zoo_dir: str = "zoo"
    outdir: Optional[str] = None
    frontier: int = 4
    deterministic: bool = True
    device: str = "cuda"
    dtype_str: str = "bf16"
    dry_run: bool = False
    skip_bench: bool = False
    skip_publish: bool = False
    skip_log: bool = False
    max_ppl_multiplier: Optional[float] = None
    speed_profile: Optional[str] = None
    use_risk_schedule: bool = False
    protection_threshold: Optional[float] = None
    workloads: List[str] = field(default_factory=lambda: ["wikitext", "chat", "rag", "code"])
    hf_token: Optional[str] = None
    dataset_path: Optional[str] = None


@dataclass
class FactoryResult:
    model_id: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    descriptor: Optional[Any] = None
    adapter: Optional[Any] = None
    compile_result: Optional[Any] = None
    benchmark_csv: Optional[Path] = None
    published_repos: List[str] = field(default_factory=list)
    dataset_record: Optional[Dict[str, Any]] = None
    outdir: Optional[Path] = None
    completeness: Optional[Dict[str, float]] = None
    wall_time_s: float = 0.0
    error: Optional[str] = None
    error_category: Optional[str] = None


def _resolve_outdir(cfg: FactoryConfig) -> Path:
    """Compute the output directory for a factory run."""
    if cfg.outdir:
        return Path(cfg.outdir)
    safe_name = cfg.model_id.replace("/", "__")
    return Path(cfg.zoo_dir) / safe_name


def _write_run_manifest(
    result: FactoryResult,
    cfg: FactoryConfig,
    descriptor,
    outdir: Path,
) -> None:
    """Write run_manifest.json at the outdir root."""
    from ..dataset.schema import SCHEMA_VERSION

    try:
        import subprocess
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_sha = ""

    try:
        from .. import __version__ as dystrio_version
    except Exception:
        dystrio_version = "1.0.0"

    tier_names = []
    if result.compile_result and result.compile_result.tiers:
        for t in result.compile_result.tiers:
            label = t.label if hasattr(t, "label") else t.name if hasattr(t, "name") else "?"
            tier_names.append(label)

    manifest = {
        "run_id": result.run_id,
        "model_id": cfg.model_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema_version": SCHEMA_VERSION,
        "adapter": type(result.adapter).__name__ if result.adapter else None,
        "support_state": getattr(descriptor, "support_state", None),
        "git_sha": git_sha,
        "dystrio_version": dystrio_version,
        "config": {
            "frontier": cfg.frontier,
            "deterministic": cfg.deterministic,
            "device": cfg.device,
            "dtype": cfg.dtype_str,
            "workloads": cfg.workloads,
            "speed_profile": cfg.speed_profile,
            "max_ppl_multiplier": cfg.max_ppl_multiplier,
        },
        "tiers_produced": tier_names,
        "output_dir": str(outdir),
        "dataset_path": cfg.dataset_path,
        "benchmark_csv": str(result.benchmark_csv) if result.benchmark_csv else None,
        "completeness": result.completeness,
        "wall_time_s": round(result.wall_time_s, 1),
        "error": result.error,
        "error_category": result.error_category,
    }

    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    _log.info("run manifest: %s", manifest_path)


def run_factory(cfg: FactoryConfig) -> FactoryResult:
    """Execute the full factory pipeline for a single model.

    Stages:
      1. Fingerprint → ArchitectureDescriptor + adapter lookup
      2. Compile → FrontierSearch producing tier outputs
      3. Benchmark → bench_runner across workloads
      4. Publish → push model cards + weights to HuggingFace
      5. Log → append to JSONL dataset
    """
    t0 = time.time()
    result = FactoryResult(model_id=cfg.model_id)

    # ── Stage 1: Fingerprint ──────────────────────────────────────────────
    _log.info("=" * 60)
    _log.info("FACTORY: %s  [run_id=%s]", cfg.model_id, result.run_id)
    _log.info("=" * 60)

    from ..architectures import fingerprint, get_adapter
    from ..architectures.descriptor import SupportState

    desc = fingerprint(cfg.model_id)
    result.descriptor = desc

    _log.info(
        "fingerprint: family=%s mlp=%s layers=%d hidden=%d support=%s",
        desc.family, desc.mlp_type, desc.num_layers, desc.hidden_size,
        desc.support_state,
    )

    if desc.support_state in (SupportState.UNSUPPORTED,):
        result.error = f"architecture unsupported: {desc.family} ({desc.config_class})"
        _log.error(result.error)
        return result

    try:
        adapter = get_adapter(desc)
        result.adapter = adapter
    except ValueError as e:
        result.error = str(e)
        _log.error("adapter lookup failed: %s", e)
        return result

    outdir = _resolve_outdir(cfg)
    result.outdir = outdir

    if cfg.dry_run:
        _log.info("[dry-run] execution plan:")
        _log.info("  model:    %s", cfg.model_id)
        _log.info("  family:   %s (%s)", desc.family, desc.mlp_type)
        _log.info("  adapter:  %s", type(adapter).__name__)
        _log.info("  frontier: %d tiers", cfg.frontier)
        _log.info("  outdir:   %s", outdir)
        _log.info("  bench:    %s", "skip" if cfg.skip_bench else cfg.workloads)
        _log.info("  publish:  %s", "skip" if cfg.skip_publish else f"org={cfg.org}")
        _log.info("  log:      %s", "skip" if cfg.skip_log else "enabled")
        result.wall_time_s = time.time() - t0
        return result

    # ── Stage 2: Compile ──────────────────────────────────────────────────
    _log.info("stage: compile")
    from .compile_stage import run_compile_stage

    compile_result = run_compile_stage(
        model_id=cfg.model_id,
        outdir=outdir,
        frontier=cfg.frontier,
        deterministic=cfg.deterministic,
        device=cfg.device,
        dtype_str=cfg.dtype_str,
        adapter=adapter,
        max_ppl_multiplier=cfg.max_ppl_multiplier,
        speed_profile=cfg.speed_profile,
        use_risk_schedule=cfg.use_risk_schedule,
        protection_threshold=cfg.protection_threshold,
    )
    result.compile_result = compile_result

    if not compile_result.tiers:
        result.error = "compilation produced no viable tiers"
        _log.error(result.error)
        result.wall_time_s = time.time() - t0
        return result

    _log.info("compile complete: %d tiers", len(compile_result.tiers))

    # ── Stage 3: Benchmark ────────────────────────────────────────────────
    if not cfg.skip_bench:
        _log.info("stage: benchmark")
        from .benchmark_stage import run_benchmark_stage

        model_dirs = []
        for tier in compile_result.tiers:
            model_dirs.append(str(tier.model_dir))

        result.benchmark_csv = run_benchmark_stage(
            model_dirs=model_dirs,
            baseline_model_id=cfg.model_id,
            outdir=outdir,
            workloads=cfg.workloads,
            device=cfg.device,
            dtype_str=cfg.dtype_str,
            deterministic=cfg.deterministic,
        )
    else:
        _log.info("stage: benchmark (skipped)")

    # ── Stage 4: Publish ──────────────────────────────────────────────────
    if not cfg.skip_publish:
        _log.info("stage: publish")
        from .publish_stage import run_publish_stage

        result.published_repos = run_publish_stage(
            compile_result=compile_result,
            org=cfg.org,
            benchmark_csv=result.benchmark_csv,
            hf_token=cfg.hf_token,
        )
    else:
        _log.info("stage: publish (skipped)")

    # ── Stage 5: Log ──────────────────────────────────────────────────────
    if not cfg.skip_log:
        _log.info("stage: log")
        from .log_stage import run_log_stage

        result.dataset_record = run_log_stage(
            compile_result=compile_result,
            descriptor=desc,
            run_id=result.run_id,
            benchmark_csv=result.benchmark_csv,
            dataset_path=cfg.dataset_path,
        )
    else:
        _log.info("stage: log (skipped)")

    result.wall_time_s = time.time() - t0

    # ── Completeness scoring ──────────────────────────────────────────────
    if result.dataset_record:
        from ..dataset.schema import DatasetRecord as _DR
        rec = _DR.from_dict(result.dataset_record)
        result.completeness = rec.completeness_score()
        _log.info("completeness: %s", result.completeness)

    # ── Run manifest ──────────────────────────────────────────────────────
    _write_run_manifest(result, cfg, desc, outdir)

    # ── Stage 6: Sync dataset to HuggingFace ─────────────────────────────
    if cfg.dataset_path and result.dataset_record:
        from ..dataset.sync import sync_dataset_to_hub
        sync_dataset_to_hub(cfg.dataset_path)

    _log.info(
        "FACTORY COMPLETE: %s — %d tiers in %.0fs  [run_id=%s]",
        cfg.model_id, len(compile_result.tiers), result.wall_time_s,
        result.run_id,
    )
    return result
