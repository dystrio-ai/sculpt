#!/usr/bin/env python3
"""Visualize workload-aware sculpt showcase results.

Reads sculpt outputs from the showcase directory and generates:
  1. Per-layer risk heatmap (layers x workloads) per model
  2. Benchmark retention radar chart per model
  3. Expert survival grid (MoE only, experts x workloads x layer)
  4. Summary comparison table (markdown + CSV)

Usage:
    python scripts/visualize_showcase.py \
        --results-dir /data/workload_showcase \
        --output-dir /data/workload_showcase/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MODELS = {
    "meta-llama_Llama-3.1-8B-Instruct": {
        "display": "Llama-3.1-8B-Instruct",
        "type": "dense",
    },
    "allenai_OLMoE-1B-7B-0924": {
        "display": "OLMoE-1B-7B",
        "type": "moe",
    },
}

WORKLOADS = ["general_v2", "code_v1", "chat", "math"]
WORKLOAD_LABELS = {
    "general_v2": "General",
    "code_v1": "Code",
    "chat": "Chat",
    "math": "Math",
}

BENCHMARKS = [
    "arc_challenge", "hellaswag", "mmlu", "truthfulqa_mc2", "winogrande", "gsm8k",
]
BENCH_LABELS = {
    "arc_challenge": "ARC-C",
    "hellaswag": "HellaSwag",
    "mmlu": "MMLU",
    "truthfulqa_mc2": "TruthfulQA",
    "winogrande": "WinoGrande",
    "gsm8k": "GSM8K",
}


def load_prescan(results_dir: Path, model_safe: str, workload: str) -> Optional[Dict]:
    path = results_dir / f"{model_safe}_{workload}" / "prescan_analysis.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_compile_report(results_dir: Path, model_safe: str, workload: str) -> Optional[Dict]:
    run_dir = results_dir / f"{model_safe}_{workload}"
    for tier_dir in sorted(run_dir.glob("frontier_*")):
        cr_path = tier_dir / "compile_report.json"
        if cr_path.exists():
            return json.loads(cr_path.read_text())
    return None


def load_metrics(results_dir: Path, model_safe: str, workload: str) -> Optional[Dict]:
    run_dir = results_dir / f"{model_safe}_{workload}"
    for tier_dir in sorted(run_dir.glob("frontier_*")):
        m_path = tier_dir / "metrics.json"
        if m_path.exists():
            return json.loads(m_path.read_text())
    return None


def load_lm_eval(results_dir: Path, model_safe: str, workload: str) -> Dict[str, float]:
    """Load lm-eval results for a sculpted model."""
    run_dir = results_dir / f"{model_safe}_{workload}"
    for tier_dir in sorted(run_dir.glob("frontier_*")):
        evals_dir = tier_dir / "evals"
        if not evals_dir.exists():
            continue
        for f in sorted(evals_dir.rglob("results_*.json")):
            data = json.loads(f.read_text())
            scores = {}
            for task, vals in data.get("results", {}).items():
                name = task.split(",")[0].strip()
                for key in ("acc_norm,none", "acc,none"):
                    if key in vals:
                        scores[name] = vals[key]
                        break
            if scores:
                return scores
    return {}


def load_baseline_lm_eval(results_dir: Path, model_safe: str) -> Dict[str, float]:
    """Load lm-eval baseline results."""
    baseline_dir = results_dir / f"{model_safe}_baseline"
    if not baseline_dir.exists():
        return {}
    for f in sorted(baseline_dir.rglob("results_*.json")):
        data = json.loads(f.read_text())
        scores = {}
        for task, vals in data.get("results", {}).items():
            name = task.split(",")[0].strip()
            for key in ("acc_norm,none", "acc,none"):
                if key in vals:
                    scores[name] = vals[key]
                    break
        if scores:
            return scores
    return {}


# ── 1. Risk Heatmap ──────────────────────────────────────────────────────────

def generate_risk_heatmap(
    results_dir: Path, output_dir: Path, model_safe: str, model_info: Dict,
) -> None:
    """Generate per-layer risk heatmap across workloads (matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  SKIP risk heatmap (matplotlib not installed)")
        return

    all_risks: Dict[str, Dict[int, float]] = {}
    max_layers = 0

    for wl in WORKLOADS:
        prescan = load_prescan(results_dir, model_safe, wl)
        if prescan is None:
            continue
        layer_risks = {}
        for key, val in prescan.get("per_layer_risk", {}).items():
            try:
                li = int(key)
                risk = val.get("risk_score", 0.5) if isinstance(val, dict) else float(val)
                layer_risks[li] = risk
                max_layers = max(max_layers, li + 1)
            except (ValueError, TypeError):
                continue
        all_risks[wl] = layer_risks

    if not all_risks or max_layers == 0:
        print(f"  SKIP risk heatmap for {model_info['display']} (no prescan data)")
        return

    matrix = np.full((len(WORKLOADS), max_layers), np.nan)
    for wi, wl in enumerate(WORKLOADS):
        if wl in all_risks:
            for li, risk in all_risks[wl].items():
                matrix[wi, li] = risk

    fig, ax = plt.subplots(figsize=(max(12, max_layers * 0.4), 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)

    ax.set_yticks(range(len(WORKLOADS)))
    ax.set_yticklabels([WORKLOAD_LABELS[w] for w in WORKLOADS])
    ax.set_xlabel("Layer Index")
    ax.set_title(f"Per-Layer Structural Risk — {model_info['display']}")

    tick_step = max(1, max_layers // 20)
    ax.set_xticks(range(0, max_layers, tick_step))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Risk Score")

    plt.tight_layout()
    out_path = output_dir / f"risk_heatmap_{model_safe}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  risk heatmap: {out_path}")


# ── 2. Benchmark Retention Radar ──────────────────────────────────────────────

def generate_benchmark_radar(
    results_dir: Path, output_dir: Path, model_safe: str, model_info: Dict,
) -> None:
    """Generate benchmark retention radar chart across workloads."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  SKIP radar (matplotlib not installed)")
        return

    baseline = load_baseline_lm_eval(results_dir, model_safe)
    if not baseline:
        print(f"  SKIP radar for {model_info['display']} (no baseline lm-eval)")
        return

    bench_names = [b for b in BENCHMARKS if b in baseline]
    if len(bench_names) < 3:
        print(f"  SKIP radar for {model_info['display']} (fewer than 3 benchmarks)")
        return

    angles = np.linspace(0, 2 * np.pi, len(bench_names), endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    colors = {"general_v2": "#2196F3", "code_v1": "#4CAF50", "chat": "#FF9800", "math": "#9C27B0"}

    for wl in WORKLOADS:
        sculpted = load_lm_eval(results_dir, model_safe, wl)
        if not sculpted:
            continue

        retention = []
        for b in bench_names:
            base_val = baseline.get(b, 0)
            sculpt_val = sculpted.get(b, 0)
            ret = (sculpt_val / base_val * 100) if base_val > 0 else 100
            retention.append(ret)
        retention.append(retention[0])

        ax.plot(angles, retention, "-o", label=WORKLOAD_LABELS[wl],
                color=colors.get(wl, "#666"), linewidth=2, markersize=5)
        ax.fill(angles, retention, alpha=0.1, color=colors.get(wl, "#666"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([BENCH_LABELS.get(b, b) for b in bench_names], size=9)
    ax.set_ylim(70, 105)
    ax.set_yticks([75, 85, 95, 100])
    ax.set_yticklabels(["75%", "85%", "95%", "100%"], size=8)
    ax.set_title(f"Benchmark Retention by Workload — {model_info['display']}", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    out_path = output_dir / f"benchmark_radar_{model_safe}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  benchmark radar: {out_path}")


# ── 3. Expert Survival Grid (MoE only) ───────────────────────────────────────

def generate_expert_grid(
    results_dir: Path, output_dir: Path, model_safe: str, model_info: Dict,
) -> None:
    """Generate expert survival grid for MoE models."""
    if model_info["type"] != "moe":
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("  SKIP expert grid (matplotlib not installed)")
        return

    grids: Dict[str, np.ndarray] = {}
    n_layers = 0
    n_experts = 0

    for wl in WORKLOADS:
        cr = load_compile_report(results_dir, model_safe, wl)
        if cr is None:
            continue

        for key, val in cr.items():
            try:
                li = int(key)
            except ValueError:
                continue
            orig = val.get("original_ffn", 0)
            kept = val.get("kept_blocks", 0)
            n_layers = max(n_layers, li + 1)
            n_experts = max(n_experts, orig)

        grid = np.zeros((n_layers, n_experts))
        for key, val in cr.items():
            try:
                li = int(key)
            except ValueError:
                continue
            kept = val.get("kept_blocks", 0)
            orig = val.get("original_ffn", 0)
            grid[li, :kept] = 1.0
        grids[wl] = grid

    if not grids or n_experts < 2:
        print(f"  SKIP expert grid for {model_info['display']} (no MoE compile data)")
        return

    n_wl = len(grids)
    fig, axes = plt.subplots(1, n_wl, figsize=(4 * n_wl, max(6, n_layers * 0.3)),
                              sharey=True)
    if n_wl == 1:
        axes = [axes]

    cmap = ListedColormap(["#ef5350", "#66bb6a"])

    for ax, (wl, grid) in zip(axes, grids.items()):
        ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xlabel("Expert Index")
        ax.set_title(WORKLOAD_LABELS[wl])
        if ax == axes[0]:
            ax.set_ylabel("Layer Index")
        tick_step = max(1, n_experts // 10)
        ax.set_xticks(range(0, n_experts, tick_step))

    fig.suptitle(f"Expert Survival by Workload — {model_info['display']}", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = output_dir / f"expert_grid_{model_safe}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  expert grid: {out_path}")


# ── 4. Summary Table ─────────────────────────────────────────────────────────

def generate_summary_table(results_dir: Path, output_dir: Path) -> None:
    """Generate markdown + CSV summary tables."""
    rows: List[Dict[str, Any]] = []

    for model_safe, model_info in MODELS.items():
        baseline = load_baseline_lm_eval(results_dir, model_safe)

        for wl in WORKLOADS:
            metrics = load_metrics(results_dir, model_safe, wl)
            prescan = load_prescan(results_dir, model_safe, wl)
            sculpted = load_lm_eval(results_dir, model_safe, wl)

            kf = metrics.get("keep_frac", "?") if metrics else "?"
            weights_gb = metrics.get("weights_gb") if metrics else None
            baseline_gb = metrics.get("baseline_weights_gb") if metrics else None
            ppl_ratio = metrics.get("ppl_ratio") if metrics else None
            risk = prescan.get("aggregate_risk") if prescan else None

            size_reduction = None
            if baseline_gb and weights_gb:
                size_reduction = round((1 - weights_gb / baseline_gb) * 100, 1)

            bench_deltas = {}
            for b in BENCHMARKS:
                bv = baseline.get(b)
                sv = sculpted.get(b)
                if bv is not None and sv is not None:
                    bench_deltas[b] = round((sv - bv) * 100, 1)

            rows.append({
                "model": model_info["display"],
                "type": model_info["type"],
                "workload": WORKLOAD_LABELS[wl],
                "keep_frac": kf,
                "size_reduction_pct": size_reduction,
                "risk_score": round(risk, 3) if risk else None,
                "ppl_ratio": round(ppl_ratio, 3) if ppl_ratio else None,
                **{f"delta_{b}": bench_deltas.get(b) for b in BENCHMARKS},
            })

    if not rows:
        print("  SKIP summary table (no data)")
        return

    # Markdown
    md_lines = [
        "# Workload-Aware Sculpt Showcase Results",
        "",
        "| Model | Type | Workload | keep_frac | Size Reduction | Risk | PPL Ratio | ARC-C | HS | MMLU | TQA | WG | GSM8K |",
        "|-------|------|----------|-----------|---------------|------|-----------|-------|------|------|------|------|-------|",
    ]
    for r in rows:
        def _f(v, fmt="{:.1f}"):
            return fmt.format(v) if v is not None else "—"
        def _d(key):
            v = r.get(key)
            if v is None:
                return "—"
            sign = "+" if v >= 0 else ""
            return f"{sign}{v:.1f}"

        md_lines.append(
            f"| {r['model']} | {r['type']} | {r['workload']} | {_f(r['keep_frac'], '{}')} | "
            f"{_f(r['size_reduction_pct'])}% | {_f(r['risk_score'], '{:.3f}')} | "
            f"{_f(r['ppl_ratio'], '{:.3f}')} | "
            f"{_d('delta_arc_challenge')} | {_d('delta_hellaswag')} | "
            f"{_d('delta_mmlu')} | {_d('delta_truthfulqa_mc2')} | "
            f"{_d('delta_winogrande')} | {_d('delta_gsm8k')} |"
        )

    md_path = output_dir / "showcase_results.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"  summary markdown: {md_path}")

    # CSV
    csv_path = output_dir / "showcase_results.csv"
    fieldnames = [
        "model", "type", "workload", "keep_frac", "size_reduction_pct",
        "risk_score", "ppl_ratio",
    ] + [f"delta_{b}" for b in BENCHMARKS]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  summary CSV: {csv_path}")

    # Redundancy comparison bar chart
    _generate_redundancy_chart(rows, output_dir)


def _generate_redundancy_chart(rows: List[Dict], output_dir: Path) -> None:
    """Bar chart: redundancy found per model per workload."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    models_seen = []
    for r in rows:
        if r["model"] not in models_seen:
            models_seen.append(r["model"])

    n_models = len(models_seen)
    n_wl = len(WORKLOADS)
    x = np.arange(n_wl)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2196F3", "#FF9800"]

    for mi, model_name in enumerate(models_seen):
        vals = []
        for wl in WORKLOADS:
            matching = [r for r in rows if r["model"] == model_name and r["workload"] == WORKLOAD_LABELS[wl]]
            if matching and matching[0]["size_reduction_pct"] is not None:
                vals.append(matching[0]["size_reduction_pct"])
            else:
                vals.append(0)

        offset = (mi - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model_name,
                      color=colors[mi % len(colors)], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([WORKLOAD_LABELS[w] for w in WORKLOADS])
    ax.set_ylabel("Size Reduction (%)")
    ax.set_title("Structural Redundancy Found by Workload")
    ax.legend()
    ax.set_ylim(0, max(r.get("size_reduction_pct", 0) or 0 for r in rows) * 1.25 or 30)

    plt.tight_layout()
    out_path = output_dir / "redundancy_by_workload.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  redundancy chart: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize workload showcase results")
    parser.add_argument("--results-dir", required=True, help="Path to showcase results")
    parser.add_argument("--output-dir", required=True, help="Path to save figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results: {results_dir}")
    print(f"Output:  {output_dir}")
    print()

    for model_safe, model_info in MODELS.items():
        print(f"Processing {model_info['display']} ({model_info['type']})...")

        generate_risk_heatmap(results_dir, output_dir, model_safe, model_info)
        generate_benchmark_radar(results_dir, output_dir, model_safe, model_info)
        generate_expert_grid(results_dir, output_dir, model_safe, model_info)

    print()
    print("Generating summary tables...")
    generate_summary_table(results_dir, output_dir)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
