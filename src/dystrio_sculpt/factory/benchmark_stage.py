"""Benchmark stage: wraps bench_runner for factory pipeline.

Runs all configured workloads against compiled tier models and the baseline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

_log = logging.getLogger(__name__)


def run_benchmark_stage(
    model_dirs: List[str],
    baseline_model_id: str,
    outdir: Path,
    *,
    workloads: Optional[List[str]] = None,
    device: str = "cuda",
    dtype_str: str = "bf16",
    seed: int = 0,
    deterministic: bool = True,
) -> Optional[Path]:
    """Run benchmarks for all tier model dirs + baseline.

    Returns path to the generated benchmarks.csv, or None on failure.
    """
    if workloads is None:
        workloads = ["wikitext", "chat", "rag", "code"]

    models = [baseline_model_id] + model_dirs

    outdir = Path(outdir)
    bench_out = outdir / "bench"
    bench_out.mkdir(parents=True, exist_ok=True)

    _log.info(
        "benchmark stage: %d models x %d workloads",
        len(models), len(workloads),
    )

    try:
        from ..bench_runner import run_bench

        csv_path = run_bench(
            models=models,
            workloads=workloads,
            prompts_dir=None,
            outdir=bench_out,
            device=device,
            dtype_str=dtype_str,
            seed=seed,
            deterministic=deterministic,
            baseline_model=baseline_model_id,
        )
        _log.info("benchmarks saved to %s", csv_path)
        return csv_path

    except Exception as exc:
        _log.error("benchmark stage failed: %s", exc, exc_info=True)
        return None
