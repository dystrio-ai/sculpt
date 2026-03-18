#!/usr/bin/env python3
"""Dystrio Optimization Factory — minimal automation skeleton.

Polls HuggingFace for popular/new causal-LM models, runs `dystrio sculpt`
on each, evaluates with `lm-eval`, and pushes results to the efficiency
dataset.  Designed to run as a cron job or long-lived daemon.

Usage:
    # Dry-run: discover models, print what would be optimized
    python scripts/factory.py --dry-run

    # Run one cycle (optimize any new models, then exit)
    python scripts/factory.py --once

    # Continuous loop (poll every --interval hours)
    python scripts/factory.py --interval 12
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [factory] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("factory")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTDIR = Path("/data/factory")
DONE_LOG = "factory_done.jsonl"

# Models we already ship or have optimized — skip these
SKIP_MODELS: Set[str] = {
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3.5-27B",
}

# Hard filters for model discovery
MAX_PARAMS_B = 15  # skip models > 15B params (single-GPU constraint)
MIN_DOWNLOADS_WEEK = 500

# lm-eval tasks for downstream evaluation
EVAL_TASKS = "arc_challenge,hellaswag,mmlu,truthfulqa_mc2"

# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def discover_models(
    max_results: int = 20,
    min_downloads: int = MIN_DOWNLOADS_WEEK,
) -> List[Dict[str, Any]]:
    """Find trending causal-LM models on HuggingFace.

    Uses the HF API to list models sorted by downloads, filtered to
    text-generation pipeline tag and reasonable size.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.error("huggingface_hub not installed — pip install huggingface_hub")
        return []

    api = HfApi()
    candidates: List[Dict[str, Any]] = []

    try:
        models = api.list_models(
            pipeline_tag="text-generation",
            sort="downloads",
            direction=-1,
            limit=max_results * 3,
        )
    except Exception as exc:
        log.error("HF API query failed: %s", exc)
        return []

    for m in models:
        model_id = m.modelId
        if model_id in SKIP_MODELS:
            continue

        # Filter by safetensors availability and reasonable size
        safetensors = any(
            s.rfilename.endswith(".safetensors")
            for s in (m.siblings or [])
        )
        if not safetensors:
            continue

        downloads = getattr(m, "downloads", 0) or 0
        if downloads < min_downloads:
            continue

        candidates.append({
            "model_id": model_id,
            "downloads": downloads,
            "tags": list(m.tags or []),
            "library_name": getattr(m, "library_name", None),
        })

        if len(candidates) >= max_results:
            break

    return candidates


def load_done_set(outdir: Path) -> Set[str]:
    """Load the set of model IDs we've already processed."""
    done_path = outdir / DONE_LOG
    done: Set[str] = set()
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            try:
                record = json.loads(line)
                done.add(record["model_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def mark_done(outdir: Path, model_id: str, status: str, details: Optional[Dict] = None) -> None:
    """Append a completion record to the done log."""
    done_path = outdir / DONE_LOG
    record = {
        "model_id": model_id,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(details or {}),
    }
    with open(done_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Sculpt + Eval pipeline
# ---------------------------------------------------------------------------


def run_sculpt(model_id: str, outdir: Path, workload: str = "general") -> Optional[Path]:
    """Run dystrio sculpt on a model. Returns the output directory or None on failure."""
    safe_name = model_id.replace("/", "_")
    sculpt_out = outdir / safe_name
    sculpt_log = outdir / f"{safe_name}_sculpt.log"

    cmd = [
        sys.executable, "-m", "dystrio_sculpt.cli", "sculpt",
        "--model-id", model_id,
        "--outdir", str(sculpt_out),
        "--workload", workload,
        "--no-push-dataset",
    ]

    # Try using the installed CLI first
    try:
        result = subprocess.run(
            ["dystrio", "--help"],
            capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            cmd = [
                "dystrio", "sculpt",
                "--model-id", model_id,
                "--outdir", str(sculpt_out),
                "--workload", workload,
                "--no-push-dataset",
            ]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    log.info("sculpt: %s -> %s", model_id, sculpt_out)
    log.info("command: %s", " ".join(cmd))

    with open(sculpt_log, "w") as logf:
        proc = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            timeout=None,
        )

    if proc.returncode != 0:
        log.error("sculpt FAILED for %s (exit %d) — see %s", model_id, proc.returncode, sculpt_log)
        return None

    log.info("sculpt OK: %s", sculpt_out)
    return sculpt_out


def run_lm_eval(model_dir: Path, eval_out: Path) -> Optional[Path]:
    """Run lm-eval on a saved model directory. Returns results path or None."""
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_dir},trust_remote_code=True",
        "--tasks", EVAL_TASKS,
        "--batch_size", "auto",
        "--output_path", str(eval_out),
    ]

    log.info("lm-eval: %s", model_dir)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        log.error("lm-eval timed out for %s", model_dir)
        return None

    if proc.returncode != 0:
        log.error("lm-eval FAILED (exit %d): %s", proc.returncode, proc.stderr[-500:] if proc.stderr else "")
        return None

    return eval_out


def push_results(sculpt_out: Path) -> None:
    """Push sculpt + eval results to the efficiency dataset."""
    try:
        src_root = Path(__file__).resolve().parent.parent / "src"
        sys.path.insert(0, str(src_root))
        from dystrio_sculpt.dataset import append_local, push_to_hub, build_record

        # Walk frontier directories and collect records
        for frontier_dir in sorted(sculpt_out.glob("frontier_*")):
            summary = frontier_dir / "summary.json"
            if not summary.exists():
                continue
            with open(summary) as f:
                data = json.load(f)

            record = build_record(
                model_id=data.get("model_id", sculpt_out.name),
                tier=frontier_dir.name,
                keep_frac=data.get("keep_frac"),
                ppl_w103=data.get("ppl_w103"),
                ppl_ratio=data.get("ppl_ratio"),
                record_type="factory_run",
            )
            append_local(record)

        push_to_hub(private=True)
        log.info("pushed results to HF dataset")
    except Exception as exc:
        log.error("dataset push failed: %s", exc)


def process_model(model_id: str, outdir: Path, workload: str = "general") -> str:
    """Full pipeline for one model: sculpt -> eval -> push. Returns status string."""
    sculpt_out = run_sculpt(model_id, outdir, workload=workload)
    if sculpt_out is None:
        return "sculpt_failed"

    # Run lm-eval on each frontier point
    eval_ok = 0
    for frontier_dir in sorted(sculpt_out.glob("frontier_*")):
        model_dir = frontier_dir / "model"
        if not model_dir.exists():
            continue
        eval_out = frontier_dir / "evals"
        result = run_lm_eval(model_dir, eval_out)
        if result is not None:
            eval_ok += 1

    if eval_ok == 0:
        log.warning("no evals succeeded for %s", model_id)

    push_results(sculpt_out)
    return "ok"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_cycle(outdir: Path, dry_run: bool = False, workload: str = "general") -> int:
    """Run one discovery + optimization cycle. Returns number of models processed."""
    outdir.mkdir(parents=True, exist_ok=True)
    done = load_done_set(outdir)

    candidates = discover_models()
    new = [c for c in candidates if c["model_id"] not in done]

    log.info("discovered %d candidates, %d new", len(candidates), len(new))
    for c in new[:5]:
        log.info("  %s (downloads=%d)", c["model_id"], c["downloads"])

    if dry_run:
        log.info("dry-run: would process %d models", len(new))
        return 0

    processed = 0
    for c in new:
        model_id = c["model_id"]
        log.info("=" * 60)
        log.info("processing: %s", model_id)

        try:
            status = process_model(model_id, outdir, workload=workload)
        except Exception as exc:
            log.error("unhandled error for %s: %s", model_id, exc)
            status = f"error: {exc}"

        mark_done(outdir, model_id, status)
        processed += 1
        log.info("done: %s -> %s", model_id, status)

    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Dystrio Optimization Factory")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--dry-run", action="store_true", help="Discover models but don't optimize")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--interval", type=float, default=12, help="Hours between cycles (continuous mode)")
    parser.add_argument("--workload", default="general", choices=["general", "code", "chat"])
    args = parser.parse_args()

    if args.dry_run or args.once:
        run_cycle(args.outdir, dry_run=args.dry_run, workload=args.workload)
        return

    log.info("factory starting — interval=%dh, outdir=%s", args.interval, args.outdir)
    while True:
        try:
            n = run_cycle(args.outdir, workload=args.workload)
            log.info("cycle complete: %d models processed", n)
        except KeyboardInterrupt:
            log.info("interrupted — exiting")
            break
        except Exception as exc:
            log.error("cycle failed: %s", exc)

        sleep_s = args.interval * 3600
        log.info("sleeping %.1fh until next cycle", args.interval)
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
