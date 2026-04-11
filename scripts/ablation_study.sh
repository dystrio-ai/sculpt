#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Physarum Ablation Study
#
# Compares 4 selection algorithms at multiple compression levels to isolate
# the contribution of each component in the Physarum structural selector.
#
# Selectors:
#   structural  — full Physarum (sensitivity + conductance + diversity + novelty)
#   sensitivity — operator sensitivity ranking only (no Physarum)
#   magnitude   — weight L2 norm ranking
#   random      — uniform random block selection
#
# Usage:
#   bash scripts/ablation_study.sh                    # full study
#   MODEL=microsoft/Phi-3-mini-4k-instruct bash scripts/ablation_study.sh  # quick validation
#   SKIP_LMEVAL=1 bash scripts/ablation_study.sh     # perplexity only (fast)
# =============================================================================

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
KEEP_FRACS="${KEEP_FRACS:-0.90,0.85,0.80,0.75}"
SELECTORS="${SELECTORS:-structural sensitivity magnitude random}"
WORKLOAD="${WORKLOAD:-general_v2}"
OUTBASE="${OUTBASE:-ablation_results}"

LMEVAL_TASKS="mmlu,hellaswag,arc_challenge,arc_easy,winogrande,piqa,boolq,openbookqa,gsm8k,truthfulqa_mc2"
LMEVAL_LIMIT="${LMEVAL_LIMIT:-500}"
SKIP_LMEVAL="${SKIP_LMEVAL:-0}"

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')

echo "=============================================="
echo "  Physarum Ablation Study"
echo "=============================================="
echo "  Model:      $MODEL"
echo "  Keep fracs: $KEEP_FRACS"
echo "  Selectors:  $SELECTORS"
echo "  Workload:   $WORKLOAD"
echo "  Output:     $OUTBASE/"
echo "  lm-eval:    $([ "$SKIP_LMEVAL" = "1" ] && echo "SKIPPED" || echo "$LMEVAL_TASKS (limit=$LMEVAL_LIMIT)")"
echo "=============================================="

mkdir -p "$OUTBASE"

# Phase 1: Baseline evaluation
echo ""
echo "[Phase 1] Baseline evaluation"
BASELINE_DIR="$OUTBASE/baseline_${MODEL_SHORT}"

_collect_lm_eval_results() {
    local outdir="$1"
    local target="$2"
    # lm-eval 0.4+ writes results into subdirectories; find the latest JSON
    local found
    found=$(find "$outdir" -name "results*.json" -type f 2>/dev/null | sort | tail -1)
    if [ -n "$found" ]; then
        cp "$found" "$target"
        echo "    Collected results -> $target"
    fi
}

if [ ! -f "$BASELINE_DIR/lm_eval_results.json" ] && [ "$SKIP_LMEVAL" != "1" ]; then
    echo "  Running lm-eval on uncompressed model..."
    mkdir -p "$BASELINE_DIR"
    BASELINE_LMEVAL_OUT="$BASELINE_DIR/lm_eval_output"
    set +e
    lm_eval --model hf \
        --model_args "pretrained=$MODEL,dtype=bfloat16" \
        --tasks "$LMEVAL_TASKS" \
        --limit "$LMEVAL_LIMIT" \
        --batch_size auto \
        --output_path "$BASELINE_LMEVAL_OUT" \
        2>&1 | tee "$BASELINE_DIR/lm_eval.log"
    BASELINE_LMEVAL_EXIT=${PIPESTATUS[0]}
    set -e
    if [ "$BASELINE_LMEVAL_EXIT" -eq 0 ]; then
        _collect_lm_eval_results "$BASELINE_LMEVAL_OUT" "$BASELINE_DIR/lm_eval_results.json"
        echo "  Baseline eval complete."
    else
        echo "  Baseline lm-eval FAILED (exit $BASELINE_LMEVAL_EXIT) — fix CUDA/HF and re-run, or SKIP_LMEVAL=1."
        exit 1
    fi
else
    echo "  Baseline eval exists or skipped."
fi

# Phase 2: Sculpt with each selector at each keep_frac
echo ""
echo "[Phase 2] Sculpt runs"

IFS=',' read -ra KF_ARRAY <<< "$KEEP_FRACS"

for SELECTOR in $SELECTORS; do
    for KF in "${KF_ARRAY[@]}"; do
        RUN_DIR="$OUTBASE/${MODEL_SHORT}_${SELECTOR}_kf${KF}"

        # Skip only when this cell finished successfully (not merely run_metadata from a failed sculpt).
        if [ -d "$RUN_DIR" ]; then
            if [ "$SKIP_LMEVAL" = "1" ]; then
                if find "$RUN_DIR" -path '*/model/config.json' -type f 2>/dev/null | head -1 | grep -q .; then
                    echo "  [$SELECTOR kf=$KF] Sculpt output exists, skipping."
                    continue
                fi
            elif [ -f "$RUN_DIR/lm_eval_results.json" ]; then
                echo "  [$SELECTOR kf=$KF] Already complete (lm-eval results present), skipping."
                continue
            fi
        fi

        echo ""
        echo "  [$SELECTOR kf=$KF] Starting sculpt..."
        mkdir -p "$RUN_DIR"

        SCULPT_OK=0
        if dystrio sculpt \
            --model-id "$MODEL" \
            --outdir "$RUN_DIR" \
            --selector "$SELECTOR" \
            --keep-fracs "$KF" \
            --frontier 1 \
            --workload "$WORKLOAD" \
            --distill \
            --no-push-dataset \
            --save-prescan \
            --deterministic \
            2>&1 | tee "$RUN_DIR/sculpt.log"; then
            echo "  [$SELECTOR kf=$KF] Sculpt complete."
            SCULPT_OK=1
        else
            echo "  [$SELECTOR kf=$KF] Sculpt FAILED — recorded as failure. See $RUN_DIR/sculpt.log"
            echo '{"failed": true, "selector": "'"$SELECTOR"'", "keep_frac": '"$KF"'}' > "$RUN_DIR/run_metadata.json"
            echo "  [$SELECTOR kf=$KF] Skipping lm-eval (no trusted checkpoint). To retry: rm -rf ${RUN_DIR@Q}"
        fi

        # lm-eval only after a successful sculpt — partial dirs after a crash still have config.json
        # and will break transformers / waste hours.
        if [ "$SCULPT_OK" -eq 1 ] && [ "$SKIP_LMEVAL" != "1" ]; then
            SCULPTED_MODEL=$(find "$RUN_DIR" -name "config.json" -path "*/model/*" -exec dirname {} \; | head -1)
            if [ -n "$SCULPTED_MODEL" ]; then
                echo "  [$SELECTOR kf=$KF] Running lm-eval on $SCULPTED_MODEL..."
                RUN_LMEVAL_OUT="$RUN_DIR/lm_eval_output"
                set +e
                lm_eval --model hf \
                    --model_args "pretrained=$SCULPTED_MODEL,dtype=bfloat16" \
                    --tasks "$LMEVAL_TASKS" \
                    --limit "$LMEVAL_LIMIT" \
                    --batch_size auto \
                    --output_path "$RUN_LMEVAL_OUT" \
                    2>&1 | tee "$RUN_DIR/lm_eval.log"
                CELL_LMEVAL_EXIT=${PIPESTATUS[0]}
                set -e
                if [ "$CELL_LMEVAL_EXIT" -eq 0 ]; then
                    _collect_lm_eval_results "$RUN_LMEVAL_OUT" "$RUN_DIR/lm_eval_results.json"
                    echo "  [$SELECTOR kf=$KF] lm-eval complete."
                else
                    echo "  [$SELECTOR kf=$KF] lm-eval FAILED (exit $CELL_LMEVAL_EXIT) — continuing with other cells."
                fi
            else
                echo "  [$SELECTOR kf=$KF] WARNING: Sculpt reported OK but no model/ found, skipping lm-eval."
            fi
        fi
    done
done

# Phase 3: Visualization
echo ""
echo "[Phase 3] Generating comparison charts"
python3 scripts/visualize_ablation.py "$OUTBASE" --model "$MODEL_SHORT" || echo "  Visualization failed (non-fatal)."

echo ""
echo "=============================================="
echo "  Ablation study complete!"
echo "  Results: $OUTBASE/"
echo "=============================================="
