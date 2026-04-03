#!/usr/bin/env python3
"""Push workload showcase results to the Dystrio Efficiency Dataset.

Reads sculpt outputs + lm-eval results from the showcase directory and
builds canonical dataset records for each (model, workload) pair.

Usage:
    python scripts/push_showcase_to_dataset.py \
        --showcase-dir /data/workload_showcase \
        --repo-id dystrio/efficiency-dataset
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dystrio_sculpt.efficiency_dataset import build_record, push_to_hub

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "allenai/OLMoE-1B-7B-0924",
]

WORKLOADS = ["general_v2", "code_v1", "chat", "math"]


def _safe(model_id: str) -> str:
    return model_id.replace("/", "_")


def load_lm_eval(results_dir: Path) -> dict:
    scores = {}
    for f in sorted(results_dir.rglob("results_*.json")):
        data = json.loads(f.read_text())
        for task, vals in data.get("results", {}).items():
            name = task.split(",")[0].strip()
            for key in ("acc_norm,none", "acc,none"):
                if key in vals:
                    scores[name] = vals[key]
                    break
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--showcase-dir", required=True)
    parser.add_argument("--repo-id", default="dystrio/efficiency-dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print records without pushing")
    args = parser.parse_args()

    showcase = Path(args.showcase_dir)
    records = []

    for model_id in MODELS:
        safe = _safe(model_id)

        # Baseline record
        baseline_dir = showcase / f"{safe}_baseline"
        if baseline_dir.exists():
            bl_scores = load_lm_eval(baseline_dir)
            if bl_scores:
                records.append(build_record(
                    model_id=model_id,
                    record_type="showcase_baseline",
                    tier="baseline",
                    keep_frac=1.0,
                    arc_challenge_acc_norm=bl_scores.get("arc_challenge"),
                    hellaswag_acc_norm=bl_scores.get("hellaswag"),
                    mmlu_acc=bl_scores.get("mmlu"),
                    truthfulqa_mc2_acc=bl_scores.get("truthfulqa_mc2"),
                    winogrande_acc=bl_scores.get("winogrande"),
                    gsm8k_acc=bl_scores.get("gsm8k"),
                    eval_engine="lm-eval",
                    notes="Workload showcase baseline",
                ))
                print(f"  baseline {safe}: {len(bl_scores)} benchmarks")
        else:
            print(f"  SKIP baseline {safe} (not found)")

        # Per-workload sculpted records
        for wl in WORKLOADS:
            run_dir = showcase / f"{safe}_{wl}"
            if not run_dir.exists():
                print(f"  SKIP {safe} / {wl} (not found)")
                continue

            # Load prescan analysis
            prescan_path = run_dir / "prescan_analysis.json"
            risk_score = None
            if prescan_path.exists():
                prescan = json.loads(prescan_path.read_text())
                risk_score = prescan.get("aggregate_risk")

            # Load metrics + lm-eval from first frontier point
            for tier_dir in sorted(run_dir.glob("frontier_*")):
                metrics_path = tier_dir / "metrics.json"
                if not metrics_path.exists():
                    continue
                metrics = json.loads(metrics_path.read_text())

                lm_scores = load_lm_eval(tier_dir / "evals") if (tier_dir / "evals").exists() else {}

                # Load downstream probe from prescan
                probe_data = {}
                if "pruning_decisions" in (prescan if prescan_path.exists() else {}):
                    pass  # probe data is in the main search output

                records.append(build_record(
                    model_id=model_id,
                    record_type="showcase_sculpt",
                    tier=tier_dir.name,
                    keep_frac=metrics.get("keep_frac"),
                    workload=wl,
                    ppl_w103=metrics.get("ppl_w103_valid"),
                    ppl_ratio=metrics.get("ppl_ratio"),
                    prefill_speedup=metrics.get("prefill_speedup"),
                    decode_speedup=metrics.get("decode_speedup"),
                    risk_score=risk_score,
                    compile_time_s=metrics.get("compile_wall_time_s"),
                    weights_gb=metrics.get("weights_gb"),
                    num_params_compressed=metrics.get("num_params"),
                    e2e_speedup_chat=metrics.get("e2e_speedup_chat"),
                    e2e_speedup_rag=metrics.get("e2e_speedup_rag"),
                    e2e_speedup_batch=metrics.get("e2e_speedup_batch"),
                    distillation_enabled=True,
                    distill_alpha=0.5,
                    arc_challenge_acc_norm=lm_scores.get("arc_challenge"),
                    hellaswag_acc_norm=lm_scores.get("hellaswag"),
                    mmlu_acc=lm_scores.get("mmlu"),
                    truthfulqa_mc2_acc=lm_scores.get("truthfulqa_mc2"),
                    winogrande_acc=lm_scores.get("winogrande"),
                    gsm8k_acc=lm_scores.get("gsm8k"),
                    eval_engine="lm-eval" if lm_scores else None,
                    notes=f"Workload showcase: {wl}",
                ))
                kf = metrics.get("keep_frac", "?")
                print(f"  {safe} / {wl}: kf={kf}, lm-eval={len(lm_scores)} tasks")
                break  # first frontier point only

    print(f"\nTotal records: {len(records)}")

    if args.dry_run:
        for r in records:
            print(json.dumps({k: v for k, v in r.items() if v is not None}, indent=2, default=str))
        return

    print(f"\nPushing to {args.repo_id}...")
    url = push_to_hub(records, repo_id=args.repo_id)
    print(f"Done: {url}")


if __name__ == "__main__":
    main()
