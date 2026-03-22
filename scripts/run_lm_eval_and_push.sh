#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Run lm_eval on all frontier models + baseline, then push to HuggingFace
#
# Usage:
#   cd ~/BumbleB
#   source .venv/bin/activate
#   nohup bash scripts/run_lm_eval_and_push.sh > lm_eval_run.log 2>&1 &
###############################################################################

SCULPT_DIR="sculpt_out_qwen35_9b_live"
BASE_MODEL="Qwen/Qwen3.5-9B"
TASKS="arc_challenge,hellaswag,mmlu,truthfulqa_mc2,winogrande,gsm8k"
ORG="dystrio"

echo "============================================================"
echo "  lm_eval + HF Push — Qwen3.5-9B Frontier Models"
echo "  $(date)"
echo "============================================================"

# ── 1. Run baseline lm_eval ──────────────────────────────────────
echo ""
echo ">> [1/5] Running baseline lm_eval on ${BASE_MODEL}..."
echo "   Started: $(date)"

lm_eval --model hf \
    --model_args "pretrained=${BASE_MODEL},dtype=bfloat16" \
    --tasks ${TASKS} \
    --batch_size auto \
    --device cuda \
    --output_path "${SCULPT_DIR}/baseline_lm_eval"

echo "   Baseline done: $(date)"

# ── 2. Run lm_eval on each frontier model ────────────────────────
TIERS=("frontier_0_default" "frontier_1_production" "frontier_2_throughput" "frontier_3_experimental")
TIER_NUM=2

for tier in "${TIERS[@]}"; do
    echo ""
    echo ">> [${TIER_NUM}/5] Running lm_eval on ${tier}..."
    echo "   Started: $(date)"

    lm_eval --model hf \
        --model_args "pretrained=${SCULPT_DIR}/${tier}/model,dtype=bfloat16" \
        --tasks ${TASKS} \
        --batch_size auto \
        --device cuda \
        --output_path "${SCULPT_DIR}/${tier}/lm_eval_results"

    echo "   ${tier} done: $(date)"
    TIER_NUM=$((TIER_NUM + 1))
done

echo ""
echo "============================================================"
echo "  All lm_eval runs complete: $(date)"
echo "============================================================"

# ── 3. Collect results and push to HF ────────────────────────────
echo ""
echo ">> Collecting results and pushing to HuggingFace..."

python3 scripts/collect_and_push.py \
    --sculpt-dir "${SCULPT_DIR}" \
    --base-model "${BASE_MODEL}" \
    --org "${ORG}"

echo ""
echo "============================================================"
echo "  DONE: $(date)"
echo "============================================================"
