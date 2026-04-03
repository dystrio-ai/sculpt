#!/usr/bin/env python3
"""Compare distillation A/B results from the efficiency dataset.

Loads records from HuggingFace `dystrio/efficiency-dataset`, groups them
by (model_id, keep_frac), and produces a side-by-side comparison of
forward-KL vs JSD distillation across benchmarks.

Usage:
    # Compare all available paired records
    python scripts/compare_distill_ab.py

    # Filter to a specific model
    python scripts/compare_distill_ab.py --model "meta-llama/Llama-3.1-8B-Instruct"

    # Save CSV output
    python scripts/compare_distill_ab.py --csv results/distill_ab.csv

Requires: datasets, tabulate (pip install datasets tabulate)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

BENCHMARK_KEYS = [
    "mmlu_acc", "hellaswag_acc_norm", "arc_challenge_acc_norm",
    "truthfulqa_mc2", "winogrande_acc",
]
BENCHMARK_SHORT = {
    "mmlu_acc": "MMLU",
    "hellaswag_acc_norm": "HellaSwag",
    "arc_challenge_acc_norm": "ARC-C",
    "truthfulqa_mc2": "TruthfulQA",
    "winogrande_acc": "Winogrande",
}


def load_records(token: Optional[str] = None) -> List[Dict[str, Any]]:
    """Pull all records from HuggingFace efficiency dataset."""
    from datasets import load_dataset

    token = token or os.environ.get("HF_TOKEN")
    ds = load_dataset("dystrio/efficiency-dataset", split="optimization_runs", token=token)
    return [dict(r) for r in ds]


def _get_loss_fn(record: Dict) -> Optional[str]:
    """Extract the distillation loss function from a record."""
    distill = record.get("distillation", {})
    if isinstance(distill, str):
        try:
            distill = json.loads(distill)
        except (json.JSONDecodeError, TypeError):
            return None

    if not isinstance(distill, dict):
        return None

    if not distill.get("enabled"):
        return "none"

    return distill.get("loss_fn", "kl")


def _get_benchmarks(record: Dict) -> Dict[str, Optional[float]]:
    """Extract benchmark values from a record."""
    out: Dict[str, Optional[float]] = {}
    benchmarks = record.get("benchmarks", {})
    if isinstance(benchmarks, str):
        try:
            benchmarks = json.loads(benchmarks)
        except (json.JSONDecodeError, TypeError):
            benchmarks = {}

    for key in BENCHMARK_KEYS:
        val = benchmarks.get(key) if isinstance(benchmarks, dict) else None
        if val is None:
            val = record.get(key)
        out[key] = float(val) if val is not None else None
    return out


def _group_key(record: Dict) -> str:
    model = record.get("model_id", "unknown")
    kf = record.get("keep_frac", 0)
    return f"{model}|{kf:.3f}"


def build_comparison(
    records: List[Dict[str, Any]],
    model_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build paired comparison rows: one per (model, keep_frac) with both KL and JSD."""
    groups: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    for r in records:
        if model_filter and model_filter not in r.get("model_id", ""):
            continue
        loss_fn = _get_loss_fn(r)
        if loss_fn is None or loss_fn == "none":
            continue
        key = _group_key(r)
        groups[key][loss_fn].append(r)

    rows = []
    for gkey, by_loss in sorted(groups.items()):
        kl_records = by_loss.get("kl", [])
        jsd_records = by_loss.get("jsd", [])
        if not kl_records or not jsd_records:
            continue

        kl_best = kl_records[-1]
        jsd_best = jsd_records[-1]

        kl_bench = _get_benchmarks(kl_best)
        jsd_bench = _get_benchmarks(jsd_best)

        model, kf_str = gkey.split("|")
        row: Dict[str, Any] = {
            "model": model,
            "keep_frac": float(kf_str),
        }
        for bkey in BENCHMARK_KEYS:
            short = BENCHMARK_SHORT[bkey]
            kl_val = kl_bench.get(bkey)
            jsd_val = jsd_bench.get(bkey)
            row[f"{short}_kl"] = kl_val
            row[f"{short}_jsd"] = jsd_val
            if kl_val is not None and jsd_val is not None:
                row[f"{short}_delta"] = round(jsd_val - kl_val, 4)
            else:
                row[f"{short}_delta"] = None
        rows.append(row)

    return rows


def print_markdown_table(rows: List[Dict[str, Any]]) -> str:
    """Format comparison as a markdown table."""
    if not rows:
        return "No paired KL/JSD records found in the dataset.\n"

    headers = ["Model", "KF"]
    for bkey in BENCHMARK_KEYS:
        short = BENCHMARK_SHORT[bkey]
        headers.extend([f"{short} KL", f"{short} JSD", f"{short} Δ"])

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for row in rows:
        model_short = row["model"].split("/")[-1][:25]
        cells = [model_short, f"{row['keep_frac']:.2f}"]
        for bkey in BENCHMARK_KEYS:
            short = BENCHMARK_SHORT[bkey]
            kl = row.get(f"{short}_kl")
            jsd = row.get(f"{short}_jsd")
            delta = row.get(f"{short}_delta")
            cells.append(f"{kl:.3f}" if kl is not None else "-")
            cells.append(f"{jsd:.3f}" if jsd is not None else "-")
            if delta is not None:
                sign = "+" if delta > 0 else ""
                cells.append(f"{sign}{delta:.3f}")
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """Save comparison to CSV."""
    import csv

    if not rows:
        print("No rows to save.", file=sys.stderr)
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {path}", file=sys.stderr)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    """Print aggregate summary of JSD vs KL improvements."""
    if not rows:
        return

    deltas: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for bkey in BENCHMARK_KEYS:
            short = BENCHMARK_SHORT[bkey]
            d = row.get(f"{short}_delta")
            if d is not None:
                deltas[short].append(d)

    print("\n=== Aggregate JSD vs KL Deltas ===")
    for short in BENCHMARK_SHORT.values():
        vals = deltas.get(short, [])
        if not vals:
            print(f"  {short:12s}: no data")
            continue
        mean = sum(vals) / len(vals)
        wins = sum(1 for v in vals if v > 0)
        sign = "+" if mean > 0 else ""
        print(f"  {short:12s}: {sign}{mean:.3f}pp avg  ({wins}/{len(vals)} wins)")


def main():
    parser = argparse.ArgumentParser(description="Compare KL vs JSD distillation results")
    parser.add_argument("--model", type=str, default=None, help="Filter to model name substring")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token")
    args = parser.parse_args()

    print("Loading records from dystrio/efficiency-dataset...", file=sys.stderr)
    records = load_records(args.token)
    print(f"Loaded {len(records)} records.", file=sys.stderr)

    rows = build_comparison(records, model_filter=args.model)
    print(f"Found {len(rows)} paired (KL, JSD) comparisons.\n", file=sys.stderr)

    md = print_markdown_table(rows)
    print(md)
    print_summary(rows)

    if args.csv:
        save_csv(rows, args.csv)

    if not rows:
        print(
            "\nNo paired data yet. Run sculpt with --distill --distill-loss jsd\n"
            "and --distill --distill-loss kl on the same (model, keep_frac)\n"
            "to generate comparison data.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
