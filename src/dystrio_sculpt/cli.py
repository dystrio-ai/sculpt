"""CLI entrypoint: dystrio sculpt."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(name="dystrio", add_completion=False, rich_markup_mode="rich")
factory_app = typer.Typer(name="factory", add_completion=False, help="Optimization factory pipeline.")
app.add_typer(factory_app, name="factory")
dataset_app = typer.Typer(name="dataset", add_completion=False, help="Efficiency dataset management.")
app.add_typer(dataset_app, name="dataset")


@app.callback()
def main(
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Minimal output: suppress most logs and progress bars.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Debug output: show external library logs and request tracing.",
    ),
) -> None:
    """Dystrio Sculpt — structural FFN compiler for decoder-only LLMs."""
    if quiet and verbose:
        typer.echo("Error: --quiet and --verbose are mutually exclusive.", err=True)
        raise typer.Exit(code=1)

    from .logging_utils import configure_logging
    configure_logging(quiet=quiet, verbose=verbose)


def _print_summary_table(selected, baseline_metrics) -> None:
    """Print a compact summary table that users screenshot."""
    base_ppl = baseline_metrics.get("ppl_w103_valid", 1.0)

    _up = "\u2191"
    header = (
        f"{'Name':<30} {'keep_frac':>9} {'PPL ratio':>9} "
        f"{'Prefill' + _up:>9} {'Decode' + _up:>9} {'Risk':>6} {'Time(s)':>8}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for pt in selected:
        ppl_ratio = pt.ppl_w103 / max(1e-9, base_ppl)
        lines.append(
            f"{pt.label:<30} {pt.keep_frac:>9.3f} {ppl_ratio:>9.4f} "
            f"{pt.prefill_speedup:>8.2f}x {pt.decode_speedup:>8.2f}x "
            f"{pt.risk_score:>6.3f} {pt.wall_time_s:>8.0f}"
        )
    lines.append(sep)

    log = logging.getLogger("dystrio.sculpt")
    for line in lines:
        log.info(line)


@app.command()
def sculpt(
    model_id: str = typer.Option(..., "--model-id", help="HuggingFace model ID (required)."),
    outdir: str = typer.Option("sculpt_out", "--outdir", help="Output directory."),
    frontier: int = typer.Option(4, "--frontier", help="Number of frontier points to emit."),
    max_ppl_multiplier: Optional[float] = typer.Option(
        None, "--max-ppl-multiplier",
        help="Quality ceiling: PPL <= baseline * multiplier. [default: 2.0]",
    ),
    target_prefill_speedup: Optional[float] = typer.Option(
        None, "--target-prefill-speedup",
        help="Keep only points with prefill speedup >= target.",
    ),
    max_compile_hours: Optional[float] = typer.Option(
        None, "--max-compile-hours",
        help="Time budget for the full search (hours).",
    ),
    deterministic: bool = typer.Option(
        False, "--deterministic",
        help="Enable deterministic mode (seed all RNGs, disable TF32).",
    ),
    policy: Optional[str] = typer.Option(
        None, "--policy",
        help="Override auto-selected repair policy (advanced). "
             "Format: ss<N>_lr<X>_p<Y> or policy index 0-3.",
    ),
    workload: Optional[str] = typer.Option(
        None, "--workload",
        help="Workload preset: code, chat, or general. "
             "Sets calibration/repair corpus to match the target distribution. "
             "Individual --calib-* flags override the preset.",
    ),
    calib_dataset: Optional[str] = typer.Option(
        None, "--calib-dataset",
        help="HF dataset for calibration corpus (overrides --workload).",
    ),
    calib_config: Optional[str] = typer.Option(
        None, "--calib-config",
        help="HF dataset config name (overrides --workload).",
    ),
    calib_split: Optional[str] = typer.Option(
        None, "--calib-split",
        help="HF dataset split (overrides --workload).",
    ),
    calib_num_samples: Optional[int] = typer.Option(
        None, "--calib-num-samples",
        help="Max calibration samples (default: use all available up to n_texts_cal).",
    ),
    calib_seq_len: Optional[int] = typer.Option(
        None, "--calib-seq-len",
        help="Sequence length for calibration (default: model max_len).",
    ),
    calib_seed: Optional[int] = typer.Option(
        None, "--calib-seed",
        help="Seed for calibration sampling (default: same as --seed).",
    ),
    calib_text_field: Optional[str] = typer.Option(
        None, "--calib-text-field",
        help="Name of the text column in the HF dataset (overrides --workload).",
    ),
    distill: bool = typer.Option(
        False, "--distill",
        help="Enable knowledge distillation during repair (teacher = uncompressed model).",
    ),
    distill_alpha: Optional[float] = typer.Option(
        None, "--distill-alpha",
        help="Force distillation alpha at all compression levels (bypasses adaptive threshold).",
    ),
    push_dataset: bool = typer.Option(
        True, "--push-dataset/--no-push-dataset",
        help="Push results to the Dystrio Efficiency Dataset on HuggingFace.",
    ),
    speed_profile: Optional[str] = typer.Option(
        None, "--speed-profile",
        help="Workload speed profile: balanced, prefill_heavy, decode_heavy, "
             "rag, chatbot, throughput, latency.",
    ),
    use_risk_schedule: bool = typer.Option(
        False, "--use-risk-schedule",
        help="Derive per-layer keep_frac from structural risk scores "
             "instead of a uniform value.",
    ),
    protection_threshold: Optional[float] = typer.Option(
        None, "--protection-threshold",
        help="Risk score above which layers skip compression entirely. "
             "[default: 0.70]",
    ),
    downstream_threshold: Optional[float] = typer.Option(
        None, "--downstream-threshold",
        help="Downstream accuracy retention required for 'safe' classification. "
             "E.g. 0.95 means keep >= 95%% of baseline accuracy. [default: 0.95]",
    ),
) -> None:
    """Compile a model across a Pareto frontier of quality vs speed."""
    log = logging.getLogger("dystrio.sculpt")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_str = "bf16" if device == "cuda" else "fp32"

    log.info("Dystrio Sculpt starting")
    log.info("  model:         %s", model_id)
    log.info("  outdir:        %s", outdir)
    log.info("  frontier:      %d", frontier)
    log.info("  deterministic: %s", deterministic)
    log.info("  device:        %s", device)
    log.info("  ppl_ceiling:   %s", max_ppl_multiplier if max_ppl_multiplier else "2.0 (default)")
    if target_prefill_speedup is not None:
        log.info("  target_speed:  %.2fx", target_prefill_speedup)
    if max_compile_hours is not None:
        log.info("  time_budget:   %.1fh", max_compile_hours)
    if policy is not None:
        log.info("  policy:        %s (override)", policy)
    if distill_alpha is not None:
        distill = True
    if distill:
        log.info("  distill:       enabled (alpha=%s)", distill_alpha or "adaptive")
    if speed_profile is not None:
        log.info("  speed_profile: %s", speed_profile)
    if use_risk_schedule:
        log.info("  risk_schedule: enabled")
    if protection_threshold is not None:
        log.info("  protection:    %.2f", protection_threshold)
    if downstream_threshold is not None:
        log.info("  downstream_th: %.2f", downstream_threshold)

    # Resolve optional policy override
    policy_override = None
    if policy is not None:
        from .policy import build_policy_ladder
        ladder = build_policy_ladder(1.0)
        if policy.isdigit():
            idx = int(policy)
            if 0 <= idx < len(ladder):
                policy_override = ladder[idx]
            else:
                log.error("policy index %d out of range (0-%d)", idx, len(ladder) - 1)
                raise typer.Exit(code=1)
        else:
            match = [p for p in ladder if p.name.startswith(policy)]
            if match:
                policy_override = match[0]
            else:
                log.error("unknown policy: %s", policy)
                raise typer.Exit(code=1)

    from .search import FrontierSearch
    from .emit import emit_frontier_point, emit_run_metadata
    from .validate import validate_saved_model
    from ._data import CalibConfig, calib_config_for_workload, is_mixture_workload
    from .efficiency_dataset import record_from_frontier_point, push_record, append_local

    # Resolve workload preset, then layer on any explicit --calib-* overrides
    mixture_wl: Optional[str] = None
    if workload is not None:
        if is_mixture_workload(workload):
            mixture_wl = workload
            log.info("  workload:      %s (mixture)", workload)
        else:
            log.info("  workload:      %s", workload)
        calib_cfg = calib_config_for_workload(workload)
    else:
        calib_cfg = CalibConfig()

    if calib_dataset is not None:
        calib_cfg.dataset = calib_dataset
    if calib_config is not None:
        calib_cfg.config = calib_config
    if calib_split is not None:
        calib_cfg.split = calib_split
    if calib_text_field is not None:
        calib_cfg.text_field = calib_text_field
    if calib_num_samples is not None:
        calib_cfg.num_samples = calib_num_samples
    if calib_seq_len is not None:
        calib_cfg.seq_len = calib_seq_len
    if calib_seed is not None:
        calib_cfg.seed = calib_seed

    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)

    emit_run_metadata(outpath, {
        "deterministic": deterministic,
        "seed": 0,
        "dtype": dtype_str,
        **calib_cfg.to_dict(),
    })

    log.info("  calib:         %s / %s / %s", calib_cfg.dataset, calib_cfg.config, calib_cfg.split)

    search = FrontierSearch(
        model_id=model_id,
        n_frontier=frontier,
        max_ppl_multiplier=max_ppl_multiplier,
        target_prefill_speedup=target_prefill_speedup,
        max_compile_hours=max_compile_hours,
        deterministic=deterministic,
        device=device,
        dtype_str=dtype_str,
        policy_override=policy_override,
        outdir=outpath,
        calib=calib_cfg,
        distill=distill,
        distill_alpha=distill_alpha,
        speed_profile=speed_profile,
        use_risk_schedule=use_risk_schedule,
        protection_threshold=protection_threshold,
        downstream_threshold=downstream_threshold,
        mixture_workload=mixture_wl,
    )

    selected = search.run()
    if not selected:
        log.error("no viable frontier points found")
        raise typer.Exit(code=1)

    for pt in selected:
        cr = pt.compile_result
        if cr is None or cr.model is None:
            log.warning("frontier point %s has no model — skipping", pt.label)
            continue

        point_dir = emit_frontier_point(
            model=cr.model,
            tokenizer=cr.tokenizer,
            outdir=outpath,
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
            log.error("validation FAILED for %s — model may be corrupt", pt.label)
            raise typer.Exit(code=2)

        search_meta = {
            "candidates": [p.keep_frac for p in search.evaluated],
            "ceiling": search.max_ppl_multiplier,
            "risk_score": search.risk_score,
        }
        ds_record = record_from_frontier_point(
            pt, cr, search.baseline_metrics or {},
            search_meta=search_meta,
            workload=mixture_wl or workload,
            baseline_downstream_accuracy=search._baseline_downstream,
        )
        if push_dataset:
            push_record(ds_record)
        else:
            append_local(ds_record)

        del cr.model
        cr.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("=" * 80)
    log.info(
        "Sculpt complete: %d points emitted to %s  (risk=%.3f, ceiling=%.2fx)",
        len(selected), outpath, search.risk_score, search.max_ppl_multiplier,
    )
    _print_summary_table(selected, search.baseline_metrics or {})
    log.info("=" * 80)


# ── bench command ─────────────────────────────────────────────────────────────

@app.command()
def bench(
    models: List[str] = typer.Option(..., "--models", help="Model IDs to benchmark."),
    workloads: List[str] = typer.Option(
        ["wikitext", "chat", "rag", "code"], "--workloads",
        help="Workloads to run.",
    ),
    prompts_dir: Optional[str] = typer.Option(None, "--prompts-dir", help="Directory with JSONL prompt packs."),
    outdir: str = typer.Option("bench_out", "--outdir", help="Output directory."),
    dtype: str = typer.Option("bf16", "--dtype", help="Model dtype: bf16|fp16|fp32."),
    device: str = typer.Option("cuda", "--device", help="Device: cuda|cpu."),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
    deterministic: bool = typer.Option(False, "--deterministic", help="Enable deterministic mode."),
    baseline_model: Optional[str] = typer.Option(None, "--baseline-model", help="Baseline model for ppl_ratio."),
) -> None:
    """Benchmark baseline + sculpted models across workloads."""
    from .bench_runner import run_bench

    outpath = Path(outdir)
    pp = Path(prompts_dir) if prompts_dir else None

    csv_path = run_bench(
        models=models,
        workloads=workloads,
        prompts_dir=pp,
        outdir=outpath,
        device=device,
        dtype_str=dtype,
        seed=seed,
        deterministic=deterministic,
        baseline_model=baseline_model,
    )

    log = logging.getLogger("dystrio.bench")
    log.info("benchmarks.csv: %s", csv_path)

    # Auto-generate report + model card
    from .report import generate_report
    report_dir = outpath / "report"
    try:
        generate_report(outpath / "results", report_dir, bench_out=outpath)
    except Exception as exc:
        log.warning("report generation failed: %s", exc)

    # Auto-run audit
    from .audit import run_audit
    try:
        audit = run_audit(outpath)
        log.info("audit: %s", audit.get("overall", "?"))
    except Exception as exc:
        log.warning("audit failed: %s", exc)


# ── bench-report command ──────────────────────────────────────────────────────

@app.command("bench-report")
def bench_report(
    results_dir: str = typer.Option(..., "--results-dir", help="Path to results/ directory."),
    outdir: str = typer.Option("bench_out/report", "--outdir", help="Report output directory."),
    bench_out: Optional[str] = typer.Option(
        None, "--bench-out",
        help="Root bench output dir (parent of results/). Used for model card.",
    ),
) -> None:
    """Generate plots and model-card snippet from existing benchmark results."""
    from .report import generate_report
    bo = Path(bench_out) if bench_out else None
    generate_report(Path(results_dir), Path(outdir), bench_out=bo)


# ── bench-audit command ───────────────────────────────────────────────────────

@app.command("bench-audit")
def bench_audit(
    bench_out: str = typer.Option(..., "--bench-out", help="Root bench output dir."),
) -> None:
    """Audit benchmark results for publishability."""
    from .audit import run_audit
    result = run_audit(Path(bench_out))
    overall = result.get("overall", "?")
    summary = result.get("summary", {})
    log = logging.getLogger("dystrio.audit")
    log.info(
        "Audit %s — pass=%d  warn=%d  fail=%d",
        overall, summary.get("pass", 0), summary.get("warn", 0), summary.get("fail", 0),
    )


# ── factory commands ─────────────────────────────────────────────────────────

@factory_app.command("run")
def factory_run(
    model_id: str = typer.Option(..., "--model-id", help="HuggingFace model ID to process."),
    org: str = typer.Option("dystrio", "--org", help="HuggingFace org for publishing."),
    zoo_dir: str = typer.Option("zoo", "--zoo-dir", help="Root directory for factory outputs."),
    frontier: int = typer.Option(4, "--frontier", help="Number of frontier tiers."),
    deterministic: bool = typer.Option(True, "--deterministic/--no-deterministic", help="Deterministic mode."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show execution plan without running."),
    skip_bench: bool = typer.Option(False, "--skip-bench", help="Skip benchmark stage."),
    skip_publish: bool = typer.Option(False, "--skip-publish", help="Skip HuggingFace publish stage."),
    skip_log: bool = typer.Option(False, "--skip-log", help="Skip dataset logging stage."),
    workloads: List[str] = typer.Option(
        ["wikitext", "chat", "rag", "code"], "--workloads", help="Benchmark workloads.",
    ),
    max_ppl_multiplier: Optional[float] = typer.Option(
        None, "--max-ppl-multiplier", help="Quality ceiling.",
    ),
    speed_profile: Optional[str] = typer.Option(
        None, "--speed-profile", help="Workload speed profile.",
    ),
    use_risk_schedule: bool = typer.Option(False, "--use-risk-schedule"),
    protection_threshold: Optional[float] = typer.Option(None, "--protection-threshold"),
    dataset_path: Optional[str] = typer.Option(None, "--dataset-path", help="Path to the JSONL dataset file."),
) -> None:
    """Run the full factory pipeline: fingerprint → compile → bench → publish → log."""
    import torch
    from .factory.orchestrator import FactoryConfig, run_factory

    log = logging.getLogger("dystrio.factory")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_str = "bf16" if device == "cuda" else "fp32"

    cfg = FactoryConfig(
        model_id=model_id,
        org=org,
        zoo_dir=zoo_dir,
        frontier=frontier,
        deterministic=deterministic,
        device=device,
        dtype_str=dtype_str,
        dry_run=dry_run,
        skip_bench=skip_bench,
        skip_publish=skip_publish,
        skip_log=skip_log,
        workloads=workloads,
        max_ppl_multiplier=max_ppl_multiplier,
        speed_profile=speed_profile,
        use_risk_schedule=use_risk_schedule,
        protection_threshold=protection_threshold,
        dataset_path=dataset_path,
    )

    result = run_factory(cfg)

    if result.error:
        log.error("factory failed: %s", result.error)
        raise typer.Exit(code=1)

    log.info(
        "factory complete: %d tiers, %.0fs  [run_id=%s]",
        len(result.compile_result.tiers) if result.compile_result else 0,
        result.wall_time_s,
        result.run_id,
    )

    if result.completeness:
        log.info("completeness: %s", result.completeness)

    if result.published_repos:
        for repo in result.published_repos:
            log.info("  published: %s", repo)


@factory_app.command("validate")
def factory_validate(
    model_id: str = typer.Option(
        "mistralai/Mistral-7B-Instruct-v0.3", "--model-id",
        help="Model to validate. Defaults to known-good golden model.",
    ),
    zoo_dir: str = typer.Option("zoo", "--zoo-dir", help="Root directory for factory outputs."),
    frontier: int = typer.Option(2, "--frontier", help="Number of frontier tiers (2 keeps it fast)."),
    dataset_path: Optional[str] = typer.Option(None, "--dataset-path", help="JSONL dataset file."),
    workloads: List[str] = typer.Option(
        ["wikitext", "chat"], "--workloads",
        help="Benchmark workloads (subset for fast validation).",
    ),
    publish: bool = typer.Option(False, "--publish", help="Also publish to HuggingFace (off by default)."),
) -> None:
    """Validation run: compile + bench + log, no publish unless --publish.

    Runs the full factory pipeline with a known-good model, validates the
    dataset record against the rich-record contract, and prints the
    completeness score. Exits 0 on pass, 1 on failure.
    """
    import torch
    from .factory.orchestrator import FactoryConfig, run_factory

    log = logging.getLogger("dystrio.factory")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_str = "bf16" if device == "cuda" else "fp32"

    log.info("=" * 60)
    log.info("VALIDATION RUN: %s", model_id)
    log.info("  device:    %s", device)
    log.info("  frontier:  %d", frontier)
    log.info("  workloads: %s", workloads)
    log.info("  publish:   %s", publish)
    log.info("=" * 60)

    cfg = FactoryConfig(
        model_id=model_id,
        zoo_dir=zoo_dir,
        frontier=frontier,
        deterministic=True,
        device=device,
        dtype_str=dtype_str,
        skip_publish=not publish,
        workloads=workloads,
        dataset_path=dataset_path,
    )

    result = run_factory(cfg)

    if result.error:
        log.error("VALIDATION FAILED: %s", result.error)
        raise typer.Exit(code=1)

    if not result.dataset_record:
        log.error("VALIDATION FAILED: no dataset record produced")
        raise typer.Exit(code=1)

    # Validate the record against the rich-record contract
    from .dataset.schema import DatasetRecord
    rec = DatasetRecord.from_dict(result.dataset_record)
    issues = rec.validate()
    scores = rec.completeness_score()

    log.info("")
    log.info("=" * 60)
    log.info("VALIDATION RESULT")
    log.info("=" * 60)
    log.info("  Run ID:          %s", result.run_id)
    log.info("  Model:           %s", model_id)
    log.info("  Tiers:           %d", len(rec.tiers))
    log.info("  Wall time:       %.0fs", result.wall_time_s)
    log.info("")
    log.info("  Completeness:")
    for k, v in scores.items():
        status = "OK" if v >= 0.8 else ("WARN" if v >= 0.5 else "FAIL")
        log.info("    %-18s %.2f  [%s]", k, v, status)
    log.info("")

    if issues:
        log.error("  Rich-record contract FAILED (%d issues):", len(issues))
        for issue in issues:
            log.error("    - %s", issue)
        log.info("")
        log.info("VALIDATION: FAIL")
        raise typer.Exit(code=1)

    log.info("  Run manifest:    %s/run_manifest.json", result.outdir)
    if dataset_path:
        log.info("  Dataset:         %s", dataset_path)
    log.info("")
    log.info("VALIDATION: PASS")


@factory_app.command("fingerprint")
def factory_fingerprint(
    model_id: str = typer.Option(..., "--model-id", help="HuggingFace model ID."),
) -> None:
    """Fingerprint a model: detect architecture, support state, and adapter."""
    import json as _json
    from .architectures import fingerprint, get_adapter
    from .architectures.descriptor import SupportState

    log = logging.getLogger("dystrio.factory")
    desc = fingerprint(model_id)

    log.info("Model:      %s", model_id)
    log.info("Family:     %s", desc.family)
    log.info("MLP type:   %s", desc.mlp_type)
    log.info("Layers:     %d", desc.num_layers)
    log.info("Hidden:     %d", desc.hidden_size)
    log.info("FFN:        %d", desc.intermediate_size)
    log.info("MoE:        %s", desc.moe)
    log.info("Confidence: %.1f", desc.confidence)
    log.info("Support:    %s", desc.support_state)

    if desc.support_state == SupportState.SUPPORTED:
        adapter = get_adapter(desc)
        log.info("Adapter:    %s", type(adapter).__name__)
    elif desc.support_state == SupportState.NEEDS_ADAPTER:
        log.info("Adapter:    NONE (needs adapter implementation)")
    elif desc.support_state == SupportState.PARTIALLY_SUPPORTED:
        log.info("Adapter:    PARTIAL (some targets not yet supported)")
    else:
        log.info("Adapter:    UNSUPPORTED")


@factory_app.command("watch")
def factory_watch(
    interval: int = typer.Option(3600, "--interval", help="Seconds between polls."),
    architectures: Optional[List[str]] = typer.Option(
        None, "--architectures", help="Architecture families to scan.",
    ),
    min_downloads: int = typer.Option(100, "--min-downloads", help="Minimum download count."),
    max_params_b: float = typer.Option(15.0, "--max-params-b", help="Max parameter count in billions."),
    limit: int = typer.Option(50, "--limit", help="Models per architecture to scan."),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Discover only, don't queue."),
    once: bool = typer.Option(False, "--once", help="Run a single poll then exit."),
) -> None:
    """Watch HuggingFace Hub for new models to optimize."""
    from .factory.watcher import watch_loop

    log = logging.getLogger("dystrio.factory")
    log.info("starting watcher (interval=%ds, dry_run=%s)", interval, dry_run)

    watch_loop(
        interval_s=interval,
        architectures=architectures,
        min_downloads=min_downloads,
        max_params_b=max_params_b,
        limit=limit,
        dry_run=dry_run,
        max_iterations=1 if once else 0,
    )


# ── dataset commands ─────────────────────────────────────────────────────────

@dataset_app.command("backfill")
def dataset_backfill(
    data_dir: str = typer.Option("data", "--data-dir", help="Root data directory with bench_* subdirs."),
    dataset_path: Optional[str] = typer.Option(None, "--dataset-path", help="JSONL output file path."),
) -> None:
    """Backfill the efficiency dataset from existing benchmark CSVs."""
    from .dataset.backfill import run_backfill
    log = logging.getLogger("dystrio.dataset")
    count = run_backfill(data_dir=data_dir, dataset_path=dataset_path)
    log.info("backfill complete: %d records added", count)


@dataset_app.command("inspect")
def dataset_inspect(
    dataset_path: Optional[str] = typer.Option(None, "--dataset-path", help="JSONL file path."),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Filter to a specific model."),
    last: int = typer.Option(0, "--last", help="Show only the last N records."),
) -> None:
    """Inspect dataset records with completeness scoring."""
    from .dataset.logger import DatasetLogger
    log = logging.getLogger("dystrio.dataset")
    logger = DatasetLogger(dataset_path)
    records = logger.read_all()

    if not records:
        log.info("No records found in %s", logger.path)
        raise typer.Exit(code=0)

    if model_id:
        records = [r for r in records if r.model_id == model_id]
    if last > 0:
        records = records[-last:]

    for r in records:
        scores = r.completeness_score()
        issues = r.validate()

        log.info("")
        log.info("Record: %s", r.model_id)
        log.info("  Run ID:    %s", r.run_id)
        log.info("  Source:    %s (v%s)", r.source, r.schema_version)
        log.info("  Timestamp: %s", r.timestamp)
        log.info("")

        arch = r.architecture
        log.info("  Descriptor:      %.2f  [family=%s, layers=%s, hidden=%s, mlp=%s]",
                 scores["descriptor"],
                 arch.get("family", "?"), arch.get("num_layers", "?"),
                 arch.get("hidden_size", "?"), arch.get("mlp_type", "?"))

        tier_detail = f"{len(r.tiers)} tiers"
        if r.tiers:
            insights_total = sum(len(t.layer_insights) for t in r.tiers)
            tier_detail += f", {insights_total} layer insights"
        log.info("  Tiers:           %.2f  [%s]", scores["tiers"], tier_detail)

        rp = r.risk_profile
        rp_detail = f"aggregate={rp.aggregate_risk:.2f}, {len(rp.layer_risks)} layer risks"
        log.info("  Risk Profile:    %.2f  [%s]", scores["risk_profile"], rp_detail)

        pt = r.policy_trace
        pt_detail = f"initial={pt.initial_policy or '?'}, final={pt.final_policy or '?'}"
        if pt.total_repair_steps:
            pt_detail += f", {pt.total_repair_steps} repair steps"
        log.info("  Policy Trace:    %.2f  [%s]", scores["policy_trace"], pt_detail)

        dt_detail = f"{len(r.decision_trace)} chars" if r.decision_trace else "empty"
        log.info("  Decision Trace:  %.2f  [%s]", scores["decision_trace"], dt_detail)

        log.info("  %s", "\u2500" * 44)
        log.info("  Overall:         %.2f", scores["overall"])

        if issues:
            log.info("")
            log.info("  Validation issues (%d):", len(issues))
            for issue in issues:
                log.info("    - %s", issue)

        log.info("")
        for t in r.tiers:
            log.info(
                "    tier=%-14s kf=%.3f  ppl_ratio=%.3f  prefill=%8.1f  decode=%6.1f  "
                "weight=%.2fGB  layers=%d",
                t.name, t.keep_frac, t.ppl_ratio, t.prefill_tps, t.decode_tps,
                t.weights_gb, len(t.layer_insights),
            )


@dataset_app.command("stats")
def dataset_stats(
    dataset_path: Optional[str] = typer.Option(None, "--dataset-path", help="JSONL file path."),
) -> None:
    """Show statistics about the efficiency dataset."""
    from .dataset.logger import DatasetLogger
    log = logging.getLogger("dystrio.dataset")
    logger = DatasetLogger(dataset_path)
    records = logger.read_all()

    log.info("Dataset: %s", logger.path)
    log.info("Records: %d", len(records))

    if records:
        models = set(r.model_id for r in records)
        total_tiers = sum(len(r.tiers) for r in records)
        log.info("Models:  %d unique", len(models))
        log.info("Tiers:   %d total", total_tiers)
        for r in records:
            tier_names = [t.name for t in r.tiers]
            log.info("  %s: %s", r.model_id, ", ".join(tier_names))


if __name__ == "__main__":
    app()
