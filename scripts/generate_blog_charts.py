#!/usr/bin/env python3
"""Generate publication-ready charts combining old ablation data + new cohesion results.

Reads lm_eval_results.json from each ablation cell directory and produces:
  1. Multi-line compression curve (all selectors, CS-7 avg and key benchmarks)
  2. GSM8K spotlight chart (where the old method hurt most)
  3. PPL comparison chart
  4. Combined summary table (markdown)

Usage:
    python scripts/generate_blog_charts.py \
        --old-results /path/to/ablation_results \
        --new-results /path/to/cohesion_results \
        --outdir blog_assets

    The script will find cells matching *_<selector>_kf<keep_frac>/ patterns.
    --new-results can be the same directory if cohesion cells are alongside old ones.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BENCHMARKS = {
    "mmlu": ("acc,none", "MMLU"),
    "hellaswag": ("acc_norm,none", "HellaSwag"),
    "arc_challenge": ("acc_norm,none", "ARC-C"),
    "arc_easy": ("acc,none", "ARC-E"),
    "winogrande": ("acc,none", "Winogrande"),
    "piqa": ("acc,none", "PIQA"),
    "boolq": ("acc,none", "BoolQ"),
    "openbookqa": ("acc_norm,none", "OBQA"),
    "gsm8k": ("exact_match,flexible-extract", "GSM8K"),
    "truthfulqa_mc2": ("acc,none", "TruthfulQA"),
}

CS7_BENCHMARKS = ["mmlu", "hellaswag", "arc_challenge", "arc_easy",
                  "winogrande", "piqa", "boolq"]

SELECTOR_DISPLAY = {
    "cohesion": "Physarum (cohesion)",
    "structural": "Physarum (diversity)",
    "sensitivity": "Sensitivity only",
    "magnitude": "Magnitude",
    "random": "Random",
}

SELECTOR_COLORS = {
    "cohesion": "#4CAF50",
    "structural": "#2196F3",
    "sensitivity": "#FF9800",
    "magnitude": "#9C27B0",
    "random": "#757575",
}

SELECTOR_MARKERS = {
    "cohesion": "D",
    "structural": "o",
    "sensitivity": "s",
    "magnitude": "^",
    "random": "x",
}


def _parse_cell_dir(name: str) -> Optional[Tuple[str, str, float]]:
    """Parse 'Model_selector_kfX.XX' into (model, selector, keep_frac)."""
    m = re.match(r"^(.+?)_(structural|cohesion|sensitivity|magnitude|random)_kf([\d.]+)$", name)
    if not m:
        return None
    return m.group(1), m.group(2), float(m.group(3))


def _load_results(cell_dir: Path) -> Optional[Dict[str, float]]:
    """Load lm_eval_results.json and extract benchmark scores."""
    results_file = cell_dir / "lm_eval_results.json"
    if not results_file.exists():
        return None
    with open(results_file) as f:
        data = json.load(f)
    results = data.get("results", {})

    scores = {}
    for bench_key, (metric_key, display_name) in BENCHMARKS.items():
        bench = results.get(bench_key, {})
        val = bench.get(metric_key)
        if val is not None:
            scores[bench_key] = float(val) * 100  # to percentage
    return scores


def _load_baseline(base_dir: Path, model_short: str) -> Optional[Dict[str, float]]:
    """Try to load baseline results."""
    baseline_dir = base_dir / f"baseline_{model_short}"
    if baseline_dir.exists():
        return _load_results(baseline_dir)
    for d in base_dir.iterdir():
        if d.is_dir() and d.name.startswith("baseline"):
            return _load_results(d)
    return None


def _cs7_avg(scores: Dict[str, float]) -> Optional[float]:
    vals = [scores[b] for b in CS7_BENCHMARKS if b in scores]
    return np.mean(vals) if len(vals) == len(CS7_BENCHMARKS) else None


def collect_all_data(
    *result_dirs: Path,
) -> Tuple[Dict[str, Dict[float, Dict[str, float]]], Optional[Dict[str, float]], str]:
    """Scan directories and return {selector: {kf: {bench: score}}}."""
    data: Dict[str, Dict[float, Dict[str, float]]] = {}
    baseline = None
    model_short = ""

    for result_dir in result_dirs:
        if not result_dir.exists():
            continue
        for cell_dir in sorted(result_dir.iterdir()):
            if not cell_dir.is_dir():
                continue

            parsed = _parse_cell_dir(cell_dir.name)
            if parsed is None:
                if cell_dir.name.startswith("baseline"):
                    baseline = _load_results(cell_dir)
                continue

            model, selector, kf = parsed
            if not model_short:
                model_short = model
            scores = _load_results(cell_dir)
            if scores is None:
                continue

            if selector not in data:
                data[selector] = {}
            data[selector][kf] = scores

    if baseline is None and model_short:
        for result_dir in result_dirs:
            baseline = _load_baseline(result_dir, model_short)
            if baseline:
                break

    return data, baseline, model_short


def generate_summary_table(
    data: Dict[str, Dict[float, Dict[str, float]]],
    baseline: Optional[Dict[str, float]],
    outpath: Path,
) -> str:
    """Generate a markdown summary table."""
    all_kfs = sorted(set(kf for sel in data.values() for kf in sel), reverse=True)
    selector_order = ["cohesion", "structural", "sensitivity", "magnitude", "random"]
    selectors = [s for s in selector_order if s in data]

    lines = ["# Ablation Results: Cohesion vs. Diversity vs. Baselines\n"]

    header_parts = ["Selector", "KF"]
    display_benches = ["mmlu", "hellaswag", "arc_challenge", "gsm8k", "truthfulqa_mc2"]
    for b in display_benches:
        header_parts.append(BENCHMARKS[b][1])
    header_parts.append("CS-7 Avg")
    lines.append("| " + " | ".join(header_parts) + " |")
    lines.append("|" + "|".join(["---"] * len(header_parts)) + "|")

    if baseline:
        row = ["**Baseline**", "1.00"]
        for b in display_benches:
            row.append(f"{baseline.get(b, 0):.1f}")
        cs7 = _cs7_avg(baseline)
        row.append(f"{cs7:.1f}" if cs7 else "-")
        lines.append("| " + " | ".join(row) + " |")

    for kf in all_kfs:
        for sel in selectors:
            if kf not in data[sel]:
                continue
            scores = data[sel][kf]
            row = [f"**{SELECTOR_DISPLAY.get(sel, sel)}**", f"{kf:.2f}"]
            for b in display_benches:
                val = scores.get(b)
                if val is not None and baseline and b in baseline:
                    pct = val / baseline[b] * 100
                    row.append(f"{val:.1f} ({pct:.0f}%)")
                elif val is not None:
                    row.append(f"{val:.1f}")
                else:
                    row.append("-")
            cs7 = _cs7_avg(scores)
            if cs7 is not None:
                base_cs7 = _cs7_avg(baseline) if baseline else None
                if base_cs7:
                    row.append(f"**{cs7:.1f}** ({cs7/base_cs7*100:.0f}%)")
                else:
                    row.append(f"**{cs7:.1f}**")
            else:
                row.append("-")
            lines.append("| " + " | ".join(row) + " |")

    # Delta table: cohesion vs structural
    if "cohesion" in data and "structural" in data:
        lines.append("\n## Cohesion Advantage (vs. Diversity Penalty)\n")
        lines.append("| KF | Δ MMLU | Δ GSM8K | Δ CS-7 Avg |")
        lines.append("|---|---|---|---|")
        for kf in all_kfs:
            if kf in data["cohesion"] and kf in data["structural"]:
                c = data["cohesion"][kf]
                s = data["structural"][kf]
                d_mmlu = c.get("mmlu", 0) - s.get("mmlu", 0)
                d_gsm = c.get("gsm8k", 0) - s.get("gsm8k", 0)
                c_cs7 = _cs7_avg(c) or 0
                s_cs7 = _cs7_avg(s) or 0
                d_cs7 = c_cs7 - s_cs7
                lines.append(f"| {kf:.2f} | {d_mmlu:+.1f}pp | {d_gsm:+.1f}pp | **{d_cs7:+.1f}pp** |")

    content = "\n".join(lines) + "\n"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(content)
    return content


def generate_charts(
    data: Dict[str, Dict[float, Dict[str, float]]],
    baseline: Optional[Dict[str, float]],
    model_short: str,
    outdir: Path,
) -> List[Path]:
    """Generate matplotlib charts. Returns list of created file paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError:
        print("matplotlib not installed — skipping charts. pip install matplotlib", file=sys.stderr)
        return []

    outdir.mkdir(parents=True, exist_ok=True)
    created = []

    selector_order = ["cohesion", "structural", "sensitivity", "magnitude", "random"]
    selectors = [s for s in selector_order if s in data]

    def _plot_metric(metric_fn, ylabel, title, filename, baseline_val=None, higher_better=True):
        fig, ax = plt.subplots(figsize=(10, 6))

        for sel in selectors:
            kfs = sorted(data[sel].keys(), reverse=True)
            vals = [metric_fn(data[sel][kf]) for kf in kfs]
            if any(v is None for v in vals):
                kfs_clean = [k for k, v in zip(kfs, vals) if v is not None]
                vals_clean = [v for v in vals if v is not None]
            else:
                kfs_clean, vals_clean = kfs, vals

            compressions = [(1 - kf) * 100 for kf in kfs_clean]
            lw = 3.0 if sel == "cohesion" else 2.0 if sel == "structural" else 1.5
            ls = "--" if sel == "random" else "-"
            ax.plot(compressions, vals_clean,
                    color=SELECTOR_COLORS[sel], marker=SELECTOR_MARKERS[sel],
                    linewidth=lw, linestyle=ls, markersize=8,
                    label=SELECTOR_DISPLAY.get(sel, sel), zorder=10 if sel == "cohesion" else 5)

        if baseline_val is not None:
            ax.axhline(y=baseline_val, color="#333333", linestyle=":", linewidth=1, alpha=0.5, label="Baseline")

        ax.set_xlabel("Neurons Removed (%)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(f"{title}\n{model_short}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        path = outdir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        created.append(path)
        print(f"  Created {path}")

    # 1. CS-7 Average
    base_cs7 = _cs7_avg(baseline) if baseline else None
    _plot_metric(
        lambda s: _cs7_avg(s),
        "Commonsense-7 Average (%)", "Benchmark Quality vs. Compression",
        "cs7_avg_comparison.png", baseline_val=base_cs7,
    )

    # 2. GSM8K spotlight
    base_gsm = baseline.get("gsm8k") if baseline else None
    _plot_metric(
        lambda s: s.get("gsm8k"),
        "GSM8K Accuracy (%)", "Math Reasoning (GSM8K) vs. Compression",
        "gsm8k_comparison.png", baseline_val=base_gsm,
    )

    # 3. MMLU
    base_mmlu = baseline.get("mmlu") if baseline else None
    _plot_metric(
        lambda s: s.get("mmlu"),
        "MMLU Accuracy (%)", "Knowledge (MMLU) vs. Compression",
        "mmlu_comparison.png", baseline_val=base_mmlu,
    )

    # 4. Cohesion vs Structural delta bar chart
    if "cohesion" in data and "structural" in data:
        common_kfs = sorted(
            set(data["cohesion"].keys()) & set(data["structural"].keys()), reverse=True
        )
        if common_kfs:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for ax, (bench, label) in zip(axes, [("gsm8k", "GSM8K"), ("mmlu", "MMLU"), (None, "CS-7 Avg")]):
                compressions = [(1 - kf) * 100 for kf in common_kfs]
                if bench:
                    deltas = [data["cohesion"][kf].get(bench, 0) - data["structural"][kf].get(bench, 0)
                              for kf in common_kfs]
                else:
                    deltas = [(_cs7_avg(data["cohesion"][kf]) or 0) - (_cs7_avg(data["structural"][kf]) or 0)
                              for kf in common_kfs]

                colors = ["#4CAF50" if d >= 0 else "#f44336" for d in deltas]
                bars = ax.bar(range(len(compressions)), deltas, color=colors, width=0.6, edgecolor="white")
                ax.set_xticks(range(len(compressions)))
                ax.set_xticklabels([f"{c:.0f}%" for c in compressions])
                ax.set_xlabel("Neurons Removed", fontsize=11)
                ax.set_ylabel("Δ percentage points", fontsize=11)
                ax.set_title(f"{label}: Cohesion − Diversity", fontsize=12, fontweight="bold")
                ax.axhline(y=0, color="#333", linewidth=0.8)
                ax.grid(True, alpha=0.2, axis="y")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                for bar, d in zip(bars, deltas):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.3 if d >= 0 else -0.8),
                            f"{d:+.1f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=10, fontweight="bold")

            fig.suptitle(f"Cohesion Advantage Over Diversity Penalty — {model_short}",
                         fontsize=14, fontweight="bold", y=1.02)
            fig.tight_layout()
            path = outdir / "cohesion_delta_bars.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            created.append(path)
            print(f"  Created {path}")

    return created


def main():
    parser = argparse.ArgumentParser(description="Generate blog charts from ablation data")
    parser.add_argument("--old-results", required=True, help="Dir with existing ablation cells")
    parser.add_argument("--new-results", default=None, help="Dir with cohesion cells (can be same dir)")
    parser.add_argument("--outdir", default="blog_assets", help="Output directory")
    parser.add_argument("--baseline-json", default=None, help="Path to baseline lm_eval_results.json")
    args = parser.parse_args()

    dirs = [Path(args.old_results)]
    if args.new_results:
        dirs.append(Path(args.new_results))

    data, baseline, model_short = collect_all_data(*dirs)

    if args.baseline_json:
        baseline_path = Path(args.baseline_json)
        if baseline_path.is_file():
            baseline = _load_results(baseline_path.parent)
            if baseline is None:
                # Try loading directly as a file in a fake parent
                with open(baseline_path) as f:
                    raw = json.load(f)
                results = raw.get("results", {})
                baseline = {}
                for bench_key, (metric_key, _) in BENCHMARKS.items():
                    bench = results.get(bench_key, {})
                    val = bench.get(metric_key)
                    if val is not None:
                        baseline[bench_key] = float(val) * 100

    if not data:
        print("ERROR: No ablation cells found.", file=sys.stderr)
        sys.exit(1)

    print(f"Model: {model_short}")
    print(f"Selectors: {list(data.keys())}")
    print(f"Keep fracs: {sorted(set(kf for s in data.values() for kf in s))}")
    print(f"Baseline: {'yes' if baseline else 'no'}")
    print()

    outdir = Path(args.outdir)

    table_path = outdir / "summary_table.md"
    content = generate_summary_table(data, baseline, table_path)
    print(f"Summary table: {table_path}")
    print(content)

    charts = generate_charts(data, baseline, model_short, outdir)
    if charts:
        print(f"\nGenerated {len(charts)} charts in {outdir}/")
    else:
        print("\nNo charts generated (install matplotlib for charts)")


if __name__ == "__main__":
    main()
