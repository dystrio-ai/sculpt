#!/usr/bin/env python3
"""Collect lm-eval baseline results from /data/zoo_ab/*_baseline/ and push
them plus the zoo A/B downstream eval results to the Dystrio Efficiency Dataset.

This script:
1. Scans for baseline result JSONs (from lm_eval)
2. Scans for zoo A/B sculpted model downstream results
3. Builds canonical dataset records for each
4. Appends them to the local cache
5. Pushes everything to HuggingFace
"""

import json
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dystrio_sculpt.dataset import build_record, append_local, push_to_hub, LOCAL_CACHE

ZOO_DIR = Path("/data/zoo_ab")

BASELINE_MODELS = [
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]


def extract_lm_eval_results(results_dir: Path) -> dict:
    """Extract benchmark scores from an lm-eval output directory."""
    for f in sorted(results_dir.rglob("results_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        r = data.get("results", {})
        return {
            "arc_challenge_acc_norm": r.get("arc_challenge", {}).get("acc_norm,none"),
            "hellaswag_acc_norm": r.get("hellaswag", {}).get("acc_norm,none"),
            "mmlu_acc": r.get("mmlu", {}).get("acc,none"),
            "truthfulqa_mc2_acc": r.get("truthfulqa_mc2", {}).get("acc,none"),
        }
    return {}


def ingest_baselines() -> int:
    """Ingest baseline (uncompressed) model eval results."""
    count = 0
    for model_id in BASELINE_MODELS:
        safe = model_id.replace("/", "_")
        baseline_dir = ZOO_DIR / f"{safe}_baseline"
        if not baseline_dir.exists():
            print(f"  SKIP: {baseline_dir} not found")
            continue

        scores = extract_lm_eval_results(baseline_dir)
        if not any(scores.values()):
            print(f"  SKIP: {model_id} — no scores found")
            continue

        record = build_record(
            model_id=model_id,
            model_family=model_id.split("/")[-1],
            method="baseline",
            tier="baseline",
            keep_frac=1.0,
            record_type="baseline_eval",
            eval_engine="lm-eval",
            eval_date="2026-03-18",
            gpu_name="A100-80GB",
            dtype="bf16",
            **scores,
        )
        append_local(record)
        count += 1
        print(f"  OK: {model_id} — ARC={scores.get('arc_challenge_acc_norm', '?')}"
              f" HS={scores.get('hellaswag_acc_norm', '?')}"
              f" MMLU={scores.get('mmlu_acc', '?')}"
              f" TQA={scores.get('truthfulqa_mc2_acc', '?')}")
    return count


def ingest_zoo_ab_evals() -> int:
    """Ingest downstream eval results from zoo A/B sculpted models."""
    count = 0
    for result_file in sorted(ZOO_DIR.rglob("frontier_*/evals/**/results_*.json")):
        path_parts = str(result_file.relative_to(ZOO_DIR)).split("/")
        variant_dir = path_parts[0]  # e.g. "google_gemma-2-2b-it_distill"
        tier = path_parts[1]  # e.g. "frontier_0_conservative"

        # Parse variant info
        parts = variant_dir.rsplit("_", 1)
        if len(parts) == 2:
            model_safe, distill_variant = parts
            model_id = model_safe.replace("_", "/", 1)
        else:
            model_id = variant_dir.replace("_", "/", 1)
            distill_variant = "unknown"

        with open(result_file) as fh:
            data = json.load(fh)
        r = data.get("results", {})

        scores = {
            "arc_challenge_acc_norm": r.get("arc_challenge", {}).get("acc_norm,none"),
            "hellaswag_acc_norm": r.get("hellaswag", {}).get("acc_norm,none"),
            "mmlu_acc": r.get("mmlu", {}).get("acc,none"),
            "truthfulqa_mc2_acc": r.get("truthfulqa_mc2", {}).get("acc,none"),
        }

        # Try to get keep_frac from the parent summary.csv
        keep_frac = None
        summary_csv = ZOO_DIR / variant_dir / "summary.csv"
        if summary_csv.exists():
            import csv
            with open(summary_csv) as cf:
                reader = csv.DictReader(cf)
                for row in reader:
                    if row.get("name") == tier:
                        keep_frac = float(row["keep_frac"])
                        break

        record = build_record(
            model_id=model_id,
            model_family=model_id.split("/")[-1] if "/" in model_id else model_id,
            method="sculpt",
            tier=tier,
            keep_frac=keep_frac,
            distillation_enabled=(distill_variant == "distill"),
            record_type="zoo_ab_downstream",
            eval_engine="lm-eval",
            eval_date="2026-03-18",
            gpu_name="A100-80GB",
            dtype="bf16",
            **scores,
        )
        append_local(record)
        count += 1

    return count


def main():
    # Don't clear the existing cache — we want to ADD to what's already there.
    # But avoid duplicates by checking record_type.
    existing_types = set()
    if LOCAL_CACHE.exists():
        for line in LOCAL_CACHE.read_text().splitlines():
            try:
                r = json.loads(line)
                existing_types.add(r.get("record_type"))
            except json.JSONDecodeError:
                continue

    if "baseline_eval" in existing_types or "zoo_ab_downstream" in existing_types:
        print("WARNING: baseline_eval or zoo_ab_downstream records already exist in local cache.")
        print("Proceeding will add duplicates. Delete data/efficiency_dataset.jsonl first if rerunning.")
        print("Continuing anyway...")

    print("Ingesting baseline evals...")
    n1 = ingest_baselines()
    print(f"  {n1} baseline records")

    print("\nIngesting zoo A/B downstream evals...")
    n2 = ingest_zoo_ab_evals()
    print(f"  {n2} zoo A/B downstream records")

    total = n1 + n2
    print(f"\nTotal new records: {total}")
    print(f"Local cache: {LOCAL_CACHE}")

    if total == 0:
        print("Nothing new to push.")
        return

    print("\nPushing to HuggingFace...")
    try:
        url = push_to_hub(private=True)
        print(f"Done: {url}")
    except Exception as exc:
        print(f"Push failed: {exc}")
        print(f"Records saved locally at {LOCAL_CACHE}")


if __name__ == "__main__":
    main()
