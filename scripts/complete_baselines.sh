#!/usr/bin/env bash
# Complete remaining baseline evals and push ALL baseline + zoo results to the
# Dystrio Efficiency Dataset.
#
# Run on A100:
#   git pull && pip install -e . && bash scripts/complete_baselines.sh
#
# Prerequisites: HF_TOKEN must be set for gated model access and dataset push.

set -euo pipefail

export TMPDIR="${TMPDIR:-/data/tmp}"
export HF_HOME="${HF_HOME:-/data/hf_cache}"
mkdir -p "$TMPDIR" "$HF_HOME"

ZOO_DIR="/data/zoo_ab"
EVAL_LOG="$ZOO_DIR/eval_baselines_complete.log"

# ── Step 1: Run any missing baseline evals ────────────────────────────────────

MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
)

echo "$(date) checking for missing baseline evals" | tee -a "$EVAL_LOG"

for m in "${MODELS[@]}"; do
    safe=$(echo "$m" | tr '/' '_')
    OUTDIR="$ZOO_DIR/${safe}_baseline"

    # Skip if results already exist
    if find "$OUTDIR" -name "results_*.json" 2>/dev/null | grep -q .; then
        echo "$(date) SKIP: $safe baseline already has results" | tee -a "$EVAL_LOG"
        continue
    fi

    mkdir -p "$OUTDIR"
    echo "$(date) START: $safe baseline" | tee -a "$EVAL_LOG"

    lm_eval \
        --model hf \
        --model_args "pretrained=$m,trust_remote_code=True" \
        --tasks arc_challenge,hellaswag,mmlu,truthfulqa_mc2 \
        --batch_size auto \
        --output_path "$OUTDIR" \
        2>&1 | tail -10 | tee -a "$EVAL_LOG"

    echo "$(date) DONE: $safe baseline" | tee -a "$EVAL_LOG"
done

echo "$(date) === ALL MISSING BASELINES COMPLETE ===" | tee -a "$EVAL_LOG"

# ── Step 2: Collect all baseline results and push to dataset ──────────────────

echo ""
echo "$(date) collecting baselines and pushing to dataset..."

python3 scripts/push_baselines_to_dataset.py

echo "$(date) === DONE ==="
