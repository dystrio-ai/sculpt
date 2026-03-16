"""Plot the Dystrio efficiency frontier: MLP compression vs downstream accuracy."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "charts"
OUT_DIR.mkdir(exist_ok=True)

results = json.loads((DATA_DIR / "eval_results.json").read_text())
configs = json.loads((DATA_DIR / "eval_configs.json").read_text())

MODEL_FAMILIES = {
    "Gemma 2 2B": {
        "prefix": "gemma",
        "baseline": "gemma_baseline",
        "tiers": ["gemma_sculpt", "gemma_production", "gemma_throughput", "gemma_experimental"],
        "color": "#4285F4",
        "marker": "o",
    },
    "Llama 3.2 3B": {
        "prefix": "llama32",
        "baseline": "llama32",
        "tiers": ["llama32_sculpt", "llama32_production", "llama32_throughput", "llama32_experimental"],
        "color": "#9B59B6",
        "marker": "s",
    },
    "Qwen 2.5 3B": {
        "prefix": "qwen3b",
        "baseline": "qwen3b",
        "tiers": ["qwen3b_sculpt", "qwen3b_production", "qwen3b_throughput", "qwen3b_experimental"],
        "color": "#2ECC71",
        "marker": "^",
    },
    "Mistral 7B": {
        "prefix": "mistral7b",
        "baseline": "mistral7b",
        "tiers": ["mistral7b_sculpt", "mistral7b_production", "mistral7b_throughput", "mistral7b_experimental"],
        "color": "#E74C3C",
        "marker": "D",
    },
    "Llama 3.1 8B": {
        "prefix": "llama31",
        "baseline": "llama31",
        "tiers": ["llama31_sculpt", "llama31_production", "llama31_throughput", "llama31_experimental"],
        "color": "#F39C12",
        "marker": "v",
    },
    "Qwen 2.5 7B": {
        "prefix": "qwen7b",
        "baseline": "qwen7b",
        "tiers": ["qwen7b_sculpt", "qwen7b_production", "qwen7b_throughput"],
        "color": "#1ABC9C",
        "marker": "P",
    },
}

BENCHMARKS = ["arc_challenge", "hellaswag", "mmlu", "truthfulqa_mc2"]


def avg_accuracy(key: str) -> float:
    r = results[key]
    return np.mean([r[b] for b in BENCHMARKS])


def compression_pct(key: str, family: dict) -> float:
    base_size = configs[family["baseline"]]["intermediate_size"]
    tier_size = configs[key]["intermediate_size"]
    return (1.0 - tier_size / base_size) * 100


def accuracy_retention(key: str, family: dict) -> float:
    base_avg = avg_accuracy(family["baseline"])
    tier_avg = avg_accuracy(key)
    return (tier_avg / base_avg) * 100


# ── Chart 1: Average accuracy retention vs compression ────────────────────────

fig, ax = plt.subplots(figsize=(12, 7))

for name, fam in MODEL_FAMILIES.items():
    xs, ys = [0.0], [100.0]  # baseline point
    for tier_key in fam["tiers"]:
        if tier_key in results and tier_key in configs:
            xs.append(compression_pct(tier_key, fam))
            ys.append(accuracy_retention(tier_key, fam))

    ax.plot(xs, ys, color=fam["color"], marker=fam["marker"], markersize=9,
            linewidth=2.2, label=name, zorder=3)

ax.axhline(y=100, color="#888", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axhline(y=90, color="#ccc", linestyle=":", linewidth=0.8, alpha=0.5)

ax.set_xlabel("MLP Width Reduction (%)", fontsize=13, fontweight="bold")
ax.set_ylabel("Avg Accuracy Retention (% of baseline)", fontsize=13, fontweight="bold")
ax.set_title("Dystrio Efficiency Frontier\nStructural Compression vs Downstream Accuracy", fontsize=15, fontweight="bold")
ax.legend(fontsize=11, loc="lower left")
ax.set_xlim(-2, 52)
ax.set_ylim(55, 105)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "efficiency_frontier.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'efficiency_frontier.png'}")


# ── Chart 2: Per-benchmark breakdown ──────────────────────────────────────────

fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
bench_labels = {"arc_challenge": "ARC-Challenge", "hellaswag": "HellaSwag",
                "mmlu": "MMLU", "truthfulqa_mc2": "TruthfulQA MC2"}

for ax, bench in zip(axes.flat, BENCHMARKS):
    for name, fam in MODEL_FAMILIES.items():
        base_val = results[fam["baseline"]][bench]
        xs, ys = [0.0], [100.0]
        for tier_key in fam["tiers"]:
            if tier_key in results and tier_key in configs:
                x = compression_pct(tier_key, fam)
                y = (results[tier_key][bench] / base_val) * 100
                xs.append(x)
                ys.append(y)
        ax.plot(xs, ys, color=fam["color"], marker=fam["marker"], markersize=7,
                linewidth=1.8, label=name)

    ax.set_title(bench_labels[bench], fontsize=12, fontweight="bold")
    ax.set_xlim(-2, 52)
    ax.set_ylim(40, 105)
    ax.axhline(y=100, color="#888", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

axes[0, 0].legend(fontsize=8, loc="lower left")
fig2.supxlabel("MLP Width Reduction (%)", fontsize=13, fontweight="bold")
fig2.supylabel("Accuracy Retention (% of baseline)", fontsize=13, fontweight="bold")
fig2.suptitle("Per-Benchmark Efficiency Frontiers", fontsize=15, fontweight="bold", y=1.01)
fig2.tight_layout()
fig2.savefig(OUT_DIR / "efficiency_frontier_per_benchmark.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'efficiency_frontier_per_benchmark.png'}")

plt.close("all")
