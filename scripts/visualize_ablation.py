#!/usr/bin/env python3
"""Visualize Physarum ablation study results.

Reads sculpt output directories and produces:
  1. Quality-vs-compression chart (one line per selector; needs matplotlib)
  2. Per-benchmark retention charts (matplotlib)
  3. Summary markdown table
  4. Optional --csv long-format export (no matplotlib; chart in Excel/Sheets/Python)

Usage:
    python scripts/visualize_ablation.py ablation_results/ --model Llama-3.1-8B-Instruct
    python scripts/visualize_ablation.py ablation_results/ --model Llama-3.1-8B-Instruct --csv
    python scripts/visualize_ablation.py -V   # print build id only

The first lines of a normal run include build id and absolute script path. If they are missing,
`git pull` in this repo did not update the script you are executing.
"""
from __future__ import annotations

import argparse
import csv
import glob as glob_std
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

ABLATION_VIZ_BUILD = "2026-04-10a"  # always printed on run; bump when CLI/discovery changes

SELECTORS = ["structural", "sensitivity", "magnitude", "random"]
SELECTOR_LABELS = {
    "structural": "Physarum (full)",
    "sensitivity": "Sensitivity only",
    "magnitude": "Magnitude",
    "random": "Random",
}
SELECTOR_STYLES = {
    "structural": {"color": "#2196F3", "marker": "o", "linewidth": 2.5},
    "sensitivity": {"color": "#FF9800", "marker": "s", "linewidth": 2.0},
    "magnitude": {"color": "#9C27B0", "marker": "^", "linewidth": 1.5},
    "random": {"color": "#757575", "marker": "x", "linewidth": 1.5, "linestyle": "--"},
}

if __name__ == "__main__":
    # First line of real runs; skipped for -V/--version only.
    _argv_rest = sys.argv[1:]
    if _argv_rest not in (["-V"], ["--version"]):
        sys.stderr.write(f"[visualize_ablation {ABLATION_VIZ_BUILD}] exec {__file__}\n")
        sys.stderr.flush()

BENCHMARKS = {
    "mmlu": "MMLU",
    "hellaswag": "HellaSwag",
    "arc_challenge": "ARC-C",
    "arc_easy": "ARC-E",
    "winogrande": "Winogrande",
    "piqa": "PIQA",
    "boolq": "BoolQ",
    "openbookqa": "OBQA",
    "gsm8k": "GSM8K",
    "truthfulqa_mc2": "TruthfulQA",
}

COMMONSENSE_7 = ["boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]

METRIC_KEYS = {
    "mmlu": ["mmlu_acc", "acc,none"],
    "hellaswag": ["hellaswag_acc_norm", "acc_norm,none"],
    "arc_challenge": ["arc_challenge_acc_norm", "acc_norm,none"],
    "arc_easy": ["arc_easy_acc", "acc,none", "acc_norm,none"],
    "winogrande": ["winogrande_acc", "acc,none"],
    "piqa": ["piqa_acc", "acc,none", "acc_norm,none"],
    "boolq": ["boolq_acc", "acc,none"],
    "openbookqa": ["openbookqa_acc_norm", "acc_norm,none", "acc,none"],
    "gsm8k": ["gsm8k_acc", "exact_match,strict-match", "exact_match,flexible-extract", "acc,none"],
    "truthfulqa_mc2": ["truthfulqa_mc2", "mc2,none", "acc,none"],
}


def _benchmarks_from_raw(raw: Any) -> Dict[str, float]:
    """Map lm-eval task dict to our benchmark keys."""
    if not isinstance(raw, dict):
        return {}
    results: Dict[str, float] = {}
    for bench_key in BENCHMARKS:
        for task_key in [bench_key, f"leaderboard_{bench_key}", f"hendrycks_{bench_key}"]:
            if task_key not in raw:
                continue
            task_data = raw[task_key]
            if isinstance(task_data, dict):
                for mk in METRIC_KEYS.get(bench_key, ["acc,none"]):
                    if mk in task_data:
                        val = task_data[mk]
                        if isinstance(val, str):
                            val = val.rstrip("%")
                        results[bench_key] = float(val)
                        break
            elif isinstance(task_data, (int, float)):
                results[bench_key] = float(task_data)
            break
    return results


def _load_lm_eval_json_file(json_path: Path) -> Dict[str, float]:
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    raw = data.get("results", data)
    return _benchmarks_from_raw(raw)


def _scan_tree_for_lm_eval(root: Path) -> Dict[str, float]:
    """Use the richest results*.json under root (prefers lm_eval_output/)."""
    if not root.is_dir():
        return {}
    candidates = sorted(root.glob("**/results*.json"))
    preferred = [c for c in candidates if "lm_eval_output" in c.parts]
    ordered = preferred + [c for c in candidates if c not in preferred]
    best: Dict[str, float] = {}
    for c in ordered:
        parsed = _load_lm_eval_json_file(c)
        if len(parsed) > len(best):
            best = parsed
    return best


def load_lm_eval_results(path: Path) -> Dict[str, float]:
    """Load lm-eval results, handling various output formats.

    Supports:
      - lm-eval 0.3.x: {"results": {"task_name": {"acc,none": 0.5}}}
      - lm-eval 0.4.x: {"results": {"task_name": {"acc,none": 0.5, "alias": "..."}}}
      - Flat format: {"task_name": 0.5}
      - Directory format: searches for results*.json if path is a directory
      - Missing / empty lm_eval_results.json: scans run dir for results*.json
    """
    if path.is_dir():
        return _scan_tree_for_lm_eval(path)

    if not path.exists():
        return _scan_tree_for_lm_eval(path.parent)

    primary = _load_lm_eval_json_file(path)
    if primary:
        return primary
    return _scan_tree_for_lm_eval(path.parent)


def load_metrics_json(path: Path) -> Dict[str, Any]:
    """Load metrics.json from a sculpt run."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _pick_run_metrics(entry: Path) -> Dict[str, Any]:
    """metrics.json usually lives under frontier_*/ not the run root."""
    candidates = sorted(entry.glob("**/metrics.json"))
    if not candidates:
        return {}
    for c in candidates:
        if "frontier_0_production" in c.parts:
            return load_metrics_json(c)
    return load_metrics_json(candidates[-1])


def _ppl_from_metrics(metrics: Dict[str, Any]) -> Optional[float]:
    """Best-effort PPL from sculpt metrics.json (keys / types vary slightly)."""
    if not metrics:
        return None
    for key in ("ppl_w103_valid", "ppl_w103"):
        v = metrics.get(key)
        if v is None:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        try:
            return float(str(v).split()[0])
        except (TypeError, ValueError):
            continue
    return None


def discover_runs(base_dir: Path, model_short: str) -> Dict[str, Dict[float, Dict]]:
    """Discover all ablation runs organized by (selector, keep_frac)."""
    runs: Dict[str, Dict[float, Dict]] = defaultdict(dict)
    base_dir = base_dir.expanduser().resolve()

    # Anchored regex: prefix-matching "magnitude" vs "random" etc. is error-prone.
    run_dir_re = re.compile(
        rf"^{re.escape(model_short)}_(structural|sensitivity|magnitude|random)_kf([0-9.]+)$"
    )

    # Glob is more reliable than iterdir() on symlinked / ephemeral result trees.
    prefix = glob_std.escape(model_short)
    for entry in sorted(base_dir.glob(f"{prefix}_*_kf*")):
        if not entry.is_dir():
            continue
        m = run_dir_re.match(entry.name.strip())
        if not m:
            continue
        sel, kf_str = m.group(1), m.group(2)
        try:
            kf = float(kf_str)
        except ValueError:
            continue

        run_data: Dict[str, Any] = {"path": entry, "keep_frac": kf}

        metrics = _pick_run_metrics(entry)
        run_data["metrics"] = metrics
        run_data["ppl"] = _ppl_from_metrics(metrics)

        lm_eval_path = entry / "lm_eval_results.json"
        run_data["benchmarks"] = load_lm_eval_results(lm_eval_path)

        runs[sel][kf] = run_data

    return dict(runs)


def load_baseline(base_dir: Path, model_short: str) -> Dict[str, float]:
    """Load baseline lm-eval results."""
    baseline_dir = base_dir / f"baseline_{model_short}"
    return load_lm_eval_results(baseline_dir / "lm_eval_results.json")


def generate_quality_chart(
    runs: Dict[str, Dict[float, Dict]],
    baseline: Dict[str, float],
    output_path: Path,
    metric: str = "mmlu",
) -> None:
    """Generate quality-vs-compression chart with one line per selector."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    baseline_val = baseline.get(metric)

    for sel in SELECTORS:
        if sel not in runs:
            continue
        kfs = sorted(runs[sel].keys(), reverse=True)
        vals = []
        kf_list = []
        for kf in kfs:
            bench = runs[sel][kf].get("benchmarks", {})
            val = bench.get(metric)
            if val is not None:
                kf_list.append(1.0 - kf)
                vals.append(val * 100)

        if not vals:
            continue

        style = SELECTOR_STYLES.get(sel, {})
        ax.plot(
            kf_list, vals,
            label=SELECTOR_LABELS.get(sel, sel),
            marker=style.get("marker", "o"),
            color=style.get("color"),
            linewidth=style.get("linewidth", 2),
            linestyle=style.get("linestyle", "-"),
            markersize=8,
        )

    if baseline_val is not None:
        ax.axhline(
            y=baseline_val * 100, color="#4CAF50", linestyle=":",
            linewidth=1.5, label=f"Baseline ({baseline_val*100:.1f}%)",
        )

    ax.set_xlabel("Compression (fraction removed)", fontsize=12)
    ax.set_ylabel(f"{BENCHMARKS.get(metric, metric)} Accuracy (%)", fontsize=12)
    ax.set_title(f"Selector Ablation: {BENCHMARKS.get(metric, metric)} vs Compression", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _commonsense_avg(bench: Dict[str, float]) -> Optional[float]:
    """Average of the 7-benchmark commonsense suite, or None if <4 present."""
    vals = [bench[b] for b in COMMONSENSE_7 if b in bench]
    return sum(vals) / len(vals) if len(vals) >= 4 else None


def export_chart_csv(
    runs: Dict[str, Dict[float, Dict]],
    baseline: Dict[str, float],
    output_path: Path,
) -> None:
    """Long-format CSV for Excel, Google Sheets, or any plotting stack (no matplotlib)."""
    fieldnames = [
        "selector_key",
        "selector_label",
        "keep_frac",
        "fraction_removed",
        "benchmark_key",
        "benchmark_label",
        "accuracy_pct",
        "baseline_pct",
        "retention_pct_of_baseline",
        "ppl",
    ]
    rows: list[dict[str, Any]] = []

    if baseline:
        for bk, blabel in BENCHMARKS.items():
            if bk not in baseline:
                continue
            bp = baseline[bk] * 100
            rows.append(
                {
                    "selector_key": "baseline",
                    "selector_label": "Baseline",
                    "keep_frac": 1.0,
                    "fraction_removed": 0.0,
                    "benchmark_key": bk,
                    "benchmark_label": blabel,
                    "accuracy_pct": round(bp, 6),
                    "baseline_pct": round(bp, 6),
                    "retention_pct_of_baseline": 100.0,
                    "ppl": "",
                }
            )
        cs = _commonsense_avg(baseline)
        if cs is not None:
            cp = cs * 100
            rows.append(
                {
                    "selector_key": "baseline",
                    "selector_label": "Baseline",
                    "keep_frac": 1.0,
                    "fraction_removed": 0.0,
                    "benchmark_key": "commonsense_7_avg",
                    "benchmark_label": "Commonsense-7 Avg",
                    "accuracy_pct": round(cp, 6),
                    "baseline_pct": round(cp, 6),
                    "retention_pct_of_baseline": 100.0,
                    "ppl": "",
                }
            )

    for sel in SELECTORS:
        if sel not in runs:
            continue
        for kf in sorted(runs[sel].keys(), reverse=True):
            run = runs[sel][kf]
            bench = run.get("benchmarks", {})
            ppl = run.get("ppl")
            ppl_out = f"{float(ppl):.6f}" if ppl is not None else ""
            label = SELECTOR_LABELS.get(sel, sel)
            fr = round(1.0 - kf, 6)
            for bk, blabel in BENCHMARKS.items():
                val = bench.get(bk)
                bv = baseline.get(bk) if baseline else None
                acc_pct: Any = round(val * 100, 6) if val is not None else ""
                b_pct: Any = round(bv * 100, 6) if bv is not None else ""
                ret_pct: Any = (
                    round(val / bv * 100, 4) if val is not None and bv not in (None, 0) else ""
                )
                rows.append(
                    {
                        "selector_key": sel,
                        "selector_label": label,
                        "keep_frac": kf,
                        "fraction_removed": fr,
                        "benchmark_key": bk,
                        "benchmark_label": blabel,
                        "accuracy_pct": acc_pct,
                        "baseline_pct": b_pct,
                        "retention_pct_of_baseline": ret_pct,
                        "ppl": ppl_out,
                    }
                )
            cs = _commonsense_avg(bench)
            b_cs = _commonsense_avg(baseline) if baseline else None
            if cs is not None:
                acc_pct = round(cs * 100, 6)
                b_pct: Any = round(b_cs * 100, 6) if b_cs is not None else ""
                ret_pct: Any = (
                    round(cs / b_cs * 100, 4) if b_cs not in (None, 0) else ""
                )
                rows.append(
                    {
                        "selector_key": sel,
                        "selector_label": label,
                        "keep_frac": kf,
                        "fraction_removed": fr,
                        "benchmark_key": "commonsense_7_avg",
                        "benchmark_label": "Commonsense-7 Avg",
                        "accuracy_pct": acc_pct,
                        "baseline_pct": b_pct,
                        "retention_pct_of_baseline": ret_pct,
                        "ppl": ppl_out,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def generate_ppl_chart(
    runs: Dict[str, Dict[float, Dict]],
    output_path: Path,
) -> None:
    """Generate perplexity-vs-compression chart (works without lm-eval)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for sel in SELECTORS:
        if sel not in runs:
            continue
        kfs = sorted(runs[sel].keys(), reverse=True)
        vals = []
        kf_list = []
        for kf in kfs:
            ppl = runs[sel][kf].get("ppl")
            if ppl is not None:
                kf_list.append(1.0 - kf)
                vals.append(ppl)

        if not vals:
            continue

        style = SELECTOR_STYLES.get(sel, {})
        ax.plot(
            kf_list, vals,
            label=SELECTOR_LABELS.get(sel, sel),
            marker=style.get("marker", "o"),
            color=style.get("color"),
            linewidth=style.get("linewidth", 2),
            linestyle=style.get("linestyle", "-"),
            markersize=8,
        )

    ax.set_xlabel("Compression (fraction removed)", fontsize=12)
    ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
    ax.set_title("Selector Ablation: Perplexity vs Compression", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_commonsense_avg_chart(
    runs: Dict[str, Dict[float, Dict]],
    baseline: Dict[str, float],
    output_path: Path,
) -> None:
    """Generate chart of average commonsense-7 accuracy vs compression."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    baseline_avg = _commonsense_avg(baseline) if baseline else None

    for sel in SELECTORS:
        if sel not in runs:
            continue
        kfs = sorted(runs[sel].keys(), reverse=True)
        vals = []
        kf_list = []
        for kf in kfs:
            bench = runs[sel][kf].get("benchmarks", {})
            avg = _commonsense_avg(bench)
            if avg is not None:
                kf_list.append(1.0 - kf)
                vals.append(avg * 100)

        if not vals:
            continue

        style = SELECTOR_STYLES.get(sel, {})
        ax.plot(
            kf_list, vals,
            label=SELECTOR_LABELS.get(sel, sel),
            marker=style.get("marker", "o"),
            color=style.get("color"),
            linewidth=style.get("linewidth", 2),
            linestyle=style.get("linestyle", "-"),
            markersize=8,
        )

    if baseline_avg is not None:
        ax.axhline(
            y=baseline_avg * 100, color="#4CAF50", linestyle=":",
            linewidth=1.5, label=f"Baseline ({baseline_avg*100:.1f}%)",
        )

    ax.set_xlabel("Compression (fraction removed)", fontsize=12)
    ax.set_ylabel("Commonsense-7 Average Accuracy (%)", fontsize=12)
    ax.set_title("Selector Ablation: Commonsense-7 Average vs Compression", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_summary_table(
    runs: Dict[str, Dict[float, Dict]],
    baseline: Dict[str, float],
    output_path: Path,
) -> None:
    """Generate markdown summary table."""
    lines = ["# Physarum Ablation Study Results\n"]

    all_kfs = sorted(set(kf for sel_runs in runs.values() for kf in sel_runs.keys()), reverse=True)
    benchmarks_present = set()
    for sel_runs in runs.values():
        for run in sel_runs.values():
            benchmarks_present.update(run.get("benchmarks", {}).keys())

    has_commonsense = len(benchmarks_present & set(COMMONSENSE_7)) >= 4

    if benchmarks_present:
        bench_cols = [b for b in BENCHMARKS if b in benchmarks_present]
        headers = ["Selector", "KF"] + [BENCHMARKS[b] for b in bench_cols]
        if has_commonsense:
            headers.append("CS-7 Avg")
        headers.append("PPL")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")

        if baseline:
            cells = ["**Baseline**", "1.00"]
            for b in bench_cols:
                val = baseline.get(b)
                cells.append(f"{val*100:.1f}" if val else "-")
            if has_commonsense:
                avg = _commonsense_avg(baseline)
                cells.append(f"{avg*100:.1f}" if avg else "-")
            cells.append("-")
            lines.append("| " + " | ".join(cells) + " |")

        for kf in all_kfs:
            for sel in SELECTORS:
                if sel not in runs or kf not in runs[sel]:
                    continue
                run = runs[sel][kf]
                bench = run.get("benchmarks", {})
                cells = [SELECTOR_LABELS.get(sel, sel), f"{kf:.2f}"]
                for b in bench_cols:
                    val = bench.get(b)
                    if val is not None and baseline.get(b):
                        retention = val / baseline[b] * 100
                        cells.append(f"{val*100:.1f} ({retention:.0f}%)")
                    elif val is not None:
                        cells.append(f"{val*100:.1f}")
                    else:
                        cells.append("-")
                if has_commonsense:
                    avg = _commonsense_avg(bench)
                    if avg is not None and _commonsense_avg(baseline):
                        retention = avg / _commonsense_avg(baseline) * 100
                        cells.append(f"{avg*100:.1f} ({retention:.0f}%)")
                    elif avg is not None:
                        cells.append(f"{avg*100:.1f}")
                    else:
                        cells.append("-")
                ppl = run.get("ppl")
                cells.append(f"{ppl:.2f}" if ppl else "-")
                lines.append("| " + " | ".join(cells) + " |")
    else:
        headers = ["Selector", "KF", "PPL"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for kf in all_kfs:
            for sel in SELECTORS:
                if sel not in runs or kf not in runs[sel]:
                    continue
                ppl = runs[sel][kf].get("ppl")
                cells = [SELECTOR_LABELS.get(sel, sel), f"{kf:.2f}", f"{ppl:.2f}" if ppl else "-"]
                lines.append("| " + " | ".join(cells) + " |")

    # Physarum advantage summary
    lines.append("\n## Physarum Advantage (Structural vs Sensitivity-only)\n")
    if "structural" in runs and "sensitivity" in runs:
        for kf in all_kfs:
            if kf not in runs["structural"] or kf not in runs["sensitivity"]:
                continue
            phys_bench = runs["structural"][kf].get("benchmarks", {})
            sens_bench = runs["sensitivity"][kf].get("benchmarks", {})

            lines.append(f"\n### keep_frac={kf:.2f} ({(1-kf)*100:.0f}% removed)\n")
            for b in BENCHMARKS:
                if b not in benchmarks_present:
                    continue
                phys = phys_bench.get(b)
                sens = sens_bench.get(b)
                if phys is not None and sens is not None:
                    delta = (phys - sens) * 100
                    sign = "+" if delta > 0 else ""
                    lines.append(f"- {BENCHMARKS[b]}: {sign}{delta:.2f}pp")

            phys_avg = _commonsense_avg(phys_bench)
            sens_avg = _commonsense_avg(sens_bench)
            if phys_avg is not None and sens_avg is not None:
                delta = (phys_avg - sens_avg) * 100
                sign = "+" if delta > 0 else ""
                lines.append(f"- **Commonsense-7 Avg: {sign}{delta:.2f}pp**")

    md = "\n".join(lines) + "\n"
    with open(output_path, "w") as f:
        f.write(md)
    print(f"  Saved: {output_path}")
    print()
    print(md)


def main():
    parser = argparse.ArgumentParser(description="Visualize Physarum ablation results")
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s build {ABLATION_VIZ_BUILD}",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print resolved script path")
    parser.add_argument("results_dir", nargs="?", help="Path to ablation results directory")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model short name (e.g. Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--csv",
        nargs="?",
        default=None,
        const="",
        metavar="FILE",
        help="Export long-format CSV for charting (Excel/Sheets/plotly); no matplotlib. "
        "Default file: <results_dir>/ablation_chart_data.csv",
    )
    args = parser.parse_args()

    if not args.results_dir or not args.model:
        parser.error("results_dir and --model are required (use -V to print build id)")

    base = Path(args.results_dir)
    if not base.exists():
        print(f"Results directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    _script_path = Path(__file__).resolve()
    print(
        f"Loading results from {base}/ for model {args.model}...  "
        f"(visualize_ablation {ABLATION_VIZ_BUILD})"
    )
    print(f"  Script path: {_script_path}")
    if args.verbose:
        print(f"  cwd: {Path.cwd()}")
    runs = discover_runs(base, args.model)
    baseline = load_baseline(base, args.model)

    if not runs:
        print("No ablation runs found.", file=sys.stderr)
        sys.exit(1)

    found = {sel: len(kfs) for sel, kfs in runs.items()}
    print(f"  Found runs: {found}")
    print(f"  Baseline benchmarks: {list(baseline.keys()) or 'none'}")

    prefix_esc = glob_std.escape(args.model)
    random_dirs = sorted(p for p in base.glob(f"{prefix_esc}_random_kf*") if p.is_dir())
    if not random_dirs:
        print(
            f"  Note: no random runs under {base}/ (expected dirs like "
            f"{args.model}_random_kf0.90). The random selector loop may not have finished."
        )

    for sel, kfs in runs.items():
        for kf, run in kfs.items():
            if not run.get("benchmarks"):
                print(
                    f"  Warning: no lm-eval benchmarks for {sel} keep_frac={kf} "
                    f"— check {run['path']}/lm_eval.log and lm_eval_output/"
                )

    if args.csv is not None:
        csv_path = (
            base / "ablation_chart_data.csv" if args.csv == "" else Path(args.csv).expanduser()
        )
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
        export_chart_csv(runs, baseline, csv_path)
        print(f"  Saved: {csv_path}")

    charts_dir = base / "charts"
    charts_dir.mkdir(exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")

        generate_ppl_chart(runs, charts_dir / "ppl_vs_compression.png")

        for bench_key in BENCHMARKS:
            has_data = any(
                bench_key in run.get("benchmarks", {})
                for sel_runs in runs.values()
                for run in sel_runs.values()
            )
            if has_data:
                generate_quality_chart(
                    runs, baseline,
                    charts_dir / f"{bench_key}_vs_compression.png",
                    metric=bench_key,
                )

        generate_commonsense_avg_chart(runs, baseline, charts_dir / "commonsense7_avg_vs_compression.png")
    except ImportError:
        print("  matplotlib not available, skipping charts.")
        print("  Install: pip install matplotlib  # or: pip install -e '.[viz]'")

    generate_summary_table(runs, baseline, base / "ablation_summary.md")


if __name__ == "__main__":
    main()
