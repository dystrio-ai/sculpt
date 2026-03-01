"""CLI entrypoint: dystrio sculpt."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="dystrio", add_completion=False, rich_markup_mode="rich")


@app.callback()
def main() -> None:
    """Dystrio Sculpt — structural FFN compiler for decoder-only LLMs."""


def _print_summary_table(selected, baseline_metrics) -> None:
    """Print a compact summary table that users screenshot."""
    base_ppl = baseline_metrics.get("ppl_w103_valid", 1.0)

    header = (
        f"{'Name':<30} {'keep_frac':>9} {'PPL ratio':>9} "
        f"{'Prefill↑':>9} {'Decode↑':>9} {'Time(s)':>8}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for pt in selected:
        ppl_ratio = pt.ppl_w103 / max(1e-9, base_ppl)
        lines.append(
            f"{pt.label:<30} {pt.keep_frac:>9.3f} {ppl_ratio:>9.4f} "
            f"{pt.prefill_speedup:>8.2f}x {pt.decode_speedup:>8.2f}x "
            f"{pt.wall_time_s:>8.0f}"
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
        help="Discard points where PPL > baseline * multiplier.",
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
) -> None:
    """Compile a model across a Pareto frontier of quality vs speed."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
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
    if max_ppl_multiplier is not None:
        log.info("  max_ppl_mult:  %.2f", max_ppl_multiplier)
    if target_prefill_speedup is not None:
        log.info("  target_speed:  %.2fx", target_prefill_speedup)
    if max_compile_hours is not None:
        log.info("  time_budget:   %.1fh", max_compile_hours)
    if policy is not None:
        log.info("  policy:        %s (override)", policy)

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
    from .emit import emit_frontier_point
    from .validate import validate_saved_model

    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)

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
        )

        model_dir = point_dir / "model"
        ok = validate_saved_model(model_dir, device=device)
        if not ok:
            log.error("validation FAILED for %s — model may be corrupt", pt.label)
            raise typer.Exit(code=2)

        del cr.model
        cr.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("=" * 70)
    log.info("Sculpt complete: %d frontier points emitted to %s", len(selected), outpath)
    _print_summary_table(selected, search.baseline_metrics or {})
    log.info("=" * 70)
