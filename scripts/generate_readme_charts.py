#!/usr/bin/env python3
"""Generate README charts from existing ablation and published model data.

Produces:
  - assets/selector_degradation.png  (quality retention vs compression, per selector)
  - assets/cross_model_efficiency.png (benchmark retention vs memory reduction, per model)

No GPU or model loading required -- uses hardcoded results from the ablation study
and published model evaluations.

Usage:
    python scripts/generate_readme_charts.py
"""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "assets"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data: Ablation study (Llama-3.1-8B-Instruct, from blog_assets/summary_table.md)
# CS-7 Avg scores at each keep_frac for each selector method.
# Baseline CS-7 Avg inferred from the ablation context (~69.2 for Llama-3.1-8B).
# ---------------------------------------------------------------------------

BASELINE_CS7 = 69.2

ABLATION = {
    "Physarum (default)": {
        "keep_fracs": [1.00, 0.90, 0.85, 0.80, 0.75],
        "cs7_avg":    [BASELINE_CS7, 67.6, 63.6, 63.9, 60.7],
    },
    "Sensitivity": {
        "keep_fracs": [1.00, 0.90, 0.85, 0.80, 0.75],
        "cs7_avg":    [BASELINE_CS7, 65.6, 63.8, 63.3, 61.5],
    },
    "Magnitude": {
        "keep_fracs": [1.00, 0.90, 0.85, 0.80, 0.75],
        "cs7_avg":    [BASELINE_CS7, 66.3, 65.2, 62.7, 61.1],
    },
    "Random": {
        "keep_fracs": [1.00, 0.90, 0.85, 0.80, 0.75],
        "cs7_avg":    [BASELINE_CS7, 64.7, 63.3, 61.5, 59.5],
    },
}

SELECTOR_STYLES = {
    "Physarum (default)":       {"color": "#2563EB", "marker": "o", "lw": 2.5, "zorder": 4},
    "Sensitivity":              {"color": "#F59E0B", "marker": "s", "lw": 1.8, "zorder": 3},
    "Magnitude":                {"color": "#8B5CF6", "marker": "^", "lw": 1.8, "zorder": 3},
    "Random":                   {"color": "#6B7280", "marker": "x", "lw": 1.5, "zorder": 2, "ls": "--"},
}

# ---------------------------------------------------------------------------
# Data: Published models (from root README "Published Models" table)
# ---------------------------------------------------------------------------

PUBLISHED_MODELS = [
    {"name": "Mistral-7B (prod)",     "mem_reduction": 11, "bench_delta": -0.2},
    {"name": "Mistral-7B (throughput)", "mem_reduction": 18, "bench_delta": -1.4},
    {"name": "Llama-3.1-8B",          "mem_reduction": 12, "bench_delta": -0.5},
    {"name": "Llama-3.2-3B",          "mem_reduction": 10, "bench_delta": -0.3},
    {"name": "Qwen2.5-3B",            "mem_reduction": 11, "bench_delta": -0.4},
    {"name": "gemma-2-2b",            "mem_reduction": 9,  "bench_delta": -0.1},
    {"name": "OLMoE-1B-7B (MoE)",    "mem_reduction": 9,  "bench_delta": +0.04},
]


def chart_selector_degradation():
    """Graph A: Quality retention vs parameters removed, one line per selector."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, data in ABLATION.items():
        style = SELECTOR_STYLES.get(name, {"color": "#333", "marker": ".", "lw": 1.5, "zorder": 2})
        retention = np.array(data["cs7_avg"]) / BASELINE_CS7 * 100
        removed_pct = (1.0 - np.array(data["keep_fracs"])) * 100

        ax.plot(
            removed_pct, retention,
            label=name,
            color=style["color"],
            marker=style["marker"],
            linewidth=style["lw"],
            linestyle=style.get("ls", "-"),
            markersize=8,
            zorder=style["zorder"],
        )

    ax.axhline(y=100, color="#D1D5DB", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Parameters removed (%)", fontsize=12)
    ax.set_ylabel("Quality retained (% of baseline)", fontsize=12)
    ax.set_title(
        "Built-in Selection Strategies — Llama-3.1-8B-Instruct\n"
        "Same pipeline, different neuron-ranking logic",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower left")
    ax.set_xlim(-2, 27)
    ax.set_ylim(84, 101)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "selector_degradation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def chart_cross_model_efficiency():
    """Graph B: Benchmark retention vs memory reduction across published models."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    xs = [m["mem_reduction"] for m in PUBLISHED_MODELS]
    ys = [100 + m["bench_delta"] for m in PUBLISHED_MODELS]
    names = [m["name"] for m in PUBLISHED_MODELS]

    ax.scatter(xs, ys, s=90, color="#2563EB", zorder=3, edgecolors="white", linewidths=0.8)

    for x, y, name in zip(xs, ys, names):
        offset_y = 0.15 if y > 99 else -0.25
        ax.annotate(
            name, (x, y),
            textcoords="offset points",
            xytext=(6, 6 if offset_y > 0 else -12),
            fontsize=8.5, color="#374151",
        )

    ax.axhline(y=100, color="#D1D5DB", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Memory reduction (%)", fontsize=12)
    ax.set_ylabel("Avg benchmark score (% of baseline)", fontsize=12)
    ax.set_title("Cross-Model Efficiency — Published Sculpted Models", fontsize=13, fontweight="bold")
    ax.set_xlim(6, 21)
    ax.set_ylim(97.5, 101)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "cross_model_efficiency.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Data: Per-model degradation curves (from HF model cards / eval dataset)
# Average benchmark score at each tier.
# ---------------------------------------------------------------------------

# Qwen3.5-9B: baseline + 4 tiers (from model card lm-eval table)
# Benchmarks: MMLU, HellaSwag, ARC-C, TruthfulQA, Winogrande, GSM8K
_QWEN35_BASELINE = np.mean([78.7, 78.1, 55.6, 53.7, 73.0, 87.3])
_QWEN35_TIERS = {
    0.95: np.mean([76.2, 75.8, 56.4, 52.6, 68.7, 81.5]),
    0.90: np.mean([73.9, 75.1, 56.8, 47.3, 69.8, 74.5]),
    0.88: np.mean([70.8, 74.0, 57.2, 52.0, 70.7, 69.6]),
    0.82: np.mean([70.2, 70.7, 53.6, 47.6, 66.6, 54.7]),
}

# Llama-3.1-8B-Instruct: baseline + 4 tiers (from ablation study, Physarum selector)
# CS-7 Avg (7 commonsense benchmarks)
_LLAMA31_BASELINE = BASELINE_CS7  # 69.2
_LLAMA31_TIERS = {
    0.90: 67.6,
    0.85: 63.6,
    0.80: 63.9,
    0.75: 60.7,
}

MODEL_CURVES = {
    "Qwen3.5-9B": {
        "baseline": _QWEN35_BASELINE,
        "tiers": _QWEN35_TIERS,
        "color": "#2563EB",
        "marker": "o",
    },
    "Llama-3.1-8B-Instruct": {
        "baseline": _LLAMA31_BASELINE,
        "tiers": _LLAMA31_TIERS,
        "color": "#E11D48",
        "marker": "s",
    },
}


def chart_per_model_degradation():
    """Graph C: Per-model degradation curves — one line per model across compression."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, data in MODEL_CURVES.items():
        baseline = data["baseline"]
        removed_pcts = [0.0]
        retentions = [100.0]

        for kf in sorted(data["tiers"].keys(), reverse=True):
            removed_pcts.append((1.0 - kf) * 100)
            retentions.append(data["tiers"][kf] / baseline * 100)

        ax.plot(
            removed_pcts, retentions,
            label=name,
            color=data["color"],
            marker=data["marker"],
            linewidth=2.2,
            markersize=8,
        )

    ax.axhline(y=100, color="#D1D5DB", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Parameters removed (%)", fontsize=12)
    ax.set_ylabel("Avg benchmark score (% of baseline)", fontsize=12)
    ax.set_title(
        "Degradation by Model — Sculpt Structural Selector\n"
        "Each line is a different model compressed at multiple levels",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower left")
    ax.set_xlim(-2, 27)
    ax.set_ylim(82, 105)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "per_model_degradation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    chart_selector_degradation()
    chart_cross_model_efficiency()
    chart_per_model_degradation()
    print("Done.")
