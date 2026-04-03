#!/usr/bin/env bash
# Workload-Aware Sculpt Showcase
#
# Demonstrates that sculpt finds workload-specific structural redundancy:
# different workloads → different neurons/experts pruned → different quality profiles.
#
# Models:
#   Dense:  Llama-3.1-8B-Instruct  (most recognized open-weight LLM)
#   MoE:    OLMoE-1B-7B-0924       (64 experts/layer, dramatic pruning divergence)
#
# Workloads:
#   general_v2  — broad coverage (WikiText-103, MMLU, OpenHermes, HellaSwag, GSM8K, OpenOrca)
#   code_v1     — code-focused (CodeAlpaca, MBPP, HumanEval, WikiText-103, OpenHermes)
#   chat        — conversation (UltraChat-200k)
#   math        — reasoning (GSM8K)
#
# Run on A100:
#   git pull && pip install -e ".[dev]"
#   nohup bash scripts/workload_showcase.sh > showcase.log 2>&1 &

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${TMPDIR:-/data/tmp}"
export HF_HOME="${HF_HOME:-/data/hf_cache}"
mkdir -p "$TMPDIR" "$HF_HOME"

OUTBASE="${SHOWCASE_OUTDIR:-/data/workload_showcase}"
LOG="$OUTBASE/showcase.log"
mkdir -p "$OUTBASE"

DENSE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
MOE_MODEL="allenai/OLMoE-1B-7B-0924"

MODELS=("$DENSE_MODEL" "$MOE_MODEL")
WORKLOADS=("general_v2" "code_v1" "chat" "math")

EVAL_TASKS="arc_challenge,hellaswag,mmlu,truthfulqa_mc2,winogrande,gsm8k"

echo "$(date) === WORKLOAD SHOWCASE START ===" | tee -a "$LOG"
echo "  Dense: $DENSE_MODEL" | tee -a "$LOG"
echo "  MoE:   $MOE_MODEL" | tee -a "$LOG"
echo "  Workloads: ${WORKLOADS[*]}" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── Phase 1: Baseline lm-eval ────────────────────────────────────────────────

for m in "${MODELS[@]}"; do
    safe=$(echo "$m" | tr '/' '_')
    BASELINE_DIR="$OUTBASE/${safe}_baseline"

    if find "$BASELINE_DIR" -name "results_*.json" 2>/dev/null | grep -q .; then
        echo "$(date) SKIP baseline eval: $safe (already done)" | tee -a "$LOG"
        continue
    fi

    echo "$(date) START baseline eval: $safe" | tee -a "$LOG"
    mkdir -p "$BASELINE_DIR"

    lm_eval \
        --model hf \
        --model_args "pretrained=$m,trust_remote_code=True" \
        --tasks $EVAL_TASKS \
        --batch_size auto \
        --output_path "$BASELINE_DIR" \
        2>&1 | tail -10 | tee -a "$LOG"

    echo "$(date) DONE baseline eval: $safe" | tee -a "$LOG"
done

# ── Phase 2: Sculpt each (model, workload) pair ──────────────────────────────

for m in "${MODELS[@]}"; do
    safe=$(echo "$m" | tr '/' '_')

    for wl in "${WORKLOADS[@]}"; do
        RUN_DIR="$OUTBASE/${safe}_${wl}"

        if [ -d "$RUN_DIR" ] && find "$RUN_DIR" -name "summary.csv" 2>/dev/null | grep -q .; then
            echo "$(date) SKIP sculpt: $safe / $wl (already done)" | tee -a "$LOG"
        else
            echo "$(date) START sculpt: $safe / $wl" | tee -a "$LOG"

            dystrio sculpt \
                --model-id "$m" \
                --outdir "$RUN_DIR" \
                --workload "$wl" \
                --distill-alpha 0.5 \
                --frontier 1 \
                --no-push-dataset \
                --save-prescan \
                2>&1 | tail -30 | tee -a "$LOG"

            echo "$(date) DONE sculpt: $safe / $wl" | tee -a "$LOG"
        fi

        # lm-eval on the sculpted model
        for tier_dir in "$RUN_DIR"/frontier_*/; do
            [ -d "$tier_dir" ] || continue
            tier=$(basename "$tier_dir")
            MODEL_DIR="${tier_dir}model"
            EVAL_OUT="${tier_dir}evals"

            if [ ! -d "$MODEL_DIR" ]; then
                echo "$(date) SKIP eval: $safe / $wl / $tier (no model dir)" | tee -a "$LOG"
                continue
            fi

            if find "$EVAL_OUT" -name "results_*.json" 2>/dev/null | grep -q .; then
                echo "$(date) SKIP eval: $safe / $wl / $tier (already done)" | tee -a "$LOG"
                continue
            fi

            echo "$(date) START eval: $safe / $wl / $tier" | tee -a "$LOG"
            mkdir -p "$EVAL_OUT"

            lm_eval \
                --model hf \
                --model_args "pretrained=$MODEL_DIR,trust_remote_code=True" \
                --tasks $EVAL_TASKS \
                --batch_size auto \
                --output_path "$EVAL_OUT" \
                2>&1 | tail -5 | tee -a "$LOG"

            echo "$(date) DONE eval: $safe / $wl / $tier" | tee -a "$LOG"
        done
    done
done

# ── Phase 3: Generate comparison table and visualizations ─────────────────────

echo "" | tee -a "$LOG"
echo "$(date) === GENERATING RESULTS ===" | tee -a "$LOG"

python3 scripts/visualize_showcase.py \
    --results-dir "$OUTBASE" \
    --output-dir "$OUTBASE/figures" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "$(date) === SHOWCASE COMPLETE ===" | tee -a "$LOG"
echo "Results in: $OUTBASE" | tee -a "$LOG"
echo "Figures in: $OUTBASE/figures/" | tee -a "$LOG"
