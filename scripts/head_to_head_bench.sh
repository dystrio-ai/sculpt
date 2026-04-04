#!/usr/bin/env bash
set -uo pipefail

# =============================================================================
# Head-to-Head Benchmark: Sculpt vs Published Structured Pruning Methods
#
# Runs Sculpt (Physarum structural selector) on LLaMA-2-7B at sparsity levels
# matching published results from DDP (ICML'26), SlimLLM, LoRAP, etc.
#
# Published targets (DDP Table 2, LLaMA-2-7B):
#   20% total sparsity: Wiki2=14.39 (DDP), 15.28 (SlimLLM), 14.67 (LoRAP)
#   50% total sparsity: Wiki2=26.34 (DDP), 27.29 (SlimLLM), 26.26 (LoRAP)
#
# Since Sculpt prunes MLP channels only (not attention heads), we run at
# two sets of keep_fracs:
#   "matched": MLP keep_frac calibrated to match the same total param reduction
#   "direct":  MLP keep_frac = 1 - sparsity_ratio (for direct comparison)
#
# Usage:
#   bash scripts/head_to_head_bench.sh                # full run
#   SKIP_LMEVAL=1 bash scripts/head_to_head_bench.sh  # perplexity screening
# =============================================================================

MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
WORKLOAD="${WORKLOAD:-general_v2}"
OUTBASE="${OUTBASE:-h2h_results}"
SKIP_LMEVAL="${SKIP_LMEVAL:-0}"

# DDP paper benchmarks: ARC-e, ARC-c, OBQA, WinoGrande, PIQA, HellaSwag,
# MathQA, RTE, BoolQ — 9 tasks
LMEVAL_TASKS="arc_easy,arc_challenge,openbookqa,winogrande,piqa,hellaswag,mathqa,rte,boolq"

# LLaMA-2-7B parameter breakdown:
#   Embeddings:  ~262M  (3.9%)
#   Attention:   ~2147M (31.9%)  [32 layers × 4 × 4096 × 4096]
#   MLP:         ~4329M (64.3%)  [32 layers × 3 × 4096 × 11008]
#   Total:       ~6738M
#
# DDP prunes attention heads + MLP channels jointly. We prune MLP only.
# To match 20% total param reduction via MLP-only pruning:
#   0.20 × 6738M = 1347.6M from MLP → MLP keep = 1 - (1347.6/4329) ≈ 0.69
# To match 50% total param reduction via MLP-only pruning:
#   0.50 × 6738M = 3369M from MLP → MLP keep = 1 - (3369/4329) ≈ 0.22
#   But 0.22 is extreme for MLP-only; cap at 0.50 and note the difference.

# Run configurations: label, keep_frac
#
# We can realistically match 20% total sparsity via MLP-only (kf=0.69).
# 50% total is not achievable through MLP-only pruning without destroying
# the model — those methods prune attention heads too. We include kf=0.50
# as our aggressive point for reference (actual total reduction ~32%).
CONFIGS=(
    "matched_20pct,0.69"
    "direct_20pct,0.80"
    "direct_15pct,0.85"
    "aggressive_32pct,0.50"
)

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')

echo "=============================================="
echo "  Head-to-Head Benchmark"
echo "=============================================="
echo "  Model:    $MODEL"
echo "  Workload: $WORKLOAD"
echo "  Output:   $OUTBASE/"
echo "  lm-eval:  $([ "$SKIP_LMEVAL" = "1" ] && echo "SKIPPED" || echo "$LMEVAL_TASKS")"
echo "=============================================="

mkdir -p "$OUTBASE"

_collect_lm_eval_results() {
    local outdir="$1"
    local target="$2"
    local found
    found=$(find "$outdir" -name "results*.json" -type f 2>/dev/null | sort | tail -1)
    if [ -n "$found" ]; then
        cp "$found" "$target"
        echo "    Collected results -> $target"
    fi
}

# ── Phase 1: Baseline lm-eval ───────────────────────────────────────────────

BASELINE_DIR="$OUTBASE/baseline_${MODEL_SHORT}"

if [ ! -f "$BASELINE_DIR/lm_eval_results.json" ] && [ "$SKIP_LMEVAL" != "1" ]; then
    echo ""
    echo "[Phase 1] Baseline lm-eval on dense model..."
    mkdir -p "$BASELINE_DIR"
    BASELINE_LMEVAL_OUT="$BASELINE_DIR/lm_eval_output"
    lm_eval --model hf \
        --model_args "pretrained=$MODEL,dtype=bfloat16" \
        --tasks "$LMEVAL_TASKS" \
        --batch_size auto \
        --output_path "$BASELINE_LMEVAL_OUT" \
        2>&1 | tee "$BASELINE_DIR/lm_eval.log"
    _collect_lm_eval_results "$BASELINE_LMEVAL_OUT" "$BASELINE_DIR/lm_eval_results.json"
    echo "  Baseline eval complete."
else
    echo ""
    echo "[Phase 1] Baseline eval exists or skipped."
fi

# ── Phase 2: Sculpt runs ────────────────────────────────────────────────────

echo ""
echo "[Phase 2] Sculpt runs"

for CONFIG in "${CONFIGS[@]}"; do
    LABEL=$(echo "$CONFIG" | cut -d',' -f1)
    KF=$(echo "$CONFIG" | cut -d',' -f2)

    RUN_DIR="$OUTBASE/${MODEL_SHORT}_${LABEL}_kf${KF}"

    if [ -d "$RUN_DIR" ] && [ -f "$RUN_DIR/run_metadata.json" ]; then
        echo "  [$LABEL kf=$KF] Already complete, skipping."
        continue
    fi

    echo ""
    echo "  [$LABEL kf=$KF] Starting sculpt..."
    mkdir -p "$RUN_DIR"

    if dystrio sculpt \
        --model-id "$MODEL" \
        --outdir "$RUN_DIR" \
        --selector structural \
        --keep-fracs "$KF" \
        --frontier 1 \
        --workload "$WORKLOAD" \
        --distill \
        --no-push-dataset \
        --save-prescan \
        --deterministic \
        2>&1 | tee "$RUN_DIR/sculpt.log"; then
        echo "  [$LABEL kf=$KF] Sculpt complete."
    else
        echo "  [$LABEL kf=$KF] Sculpt FAILED (exit $?) — recorded."
        echo '{"failed": true, "label": "'"$LABEL"'", "keep_frac": '"$KF"'}' > "$RUN_DIR/run_metadata.json"
    fi

    # lm-eval on sculpted model
    if [ "$SKIP_LMEVAL" != "1" ]; then
        SCULPTED_MODEL=$(find "$RUN_DIR" -name "config.json" -path "*/model/*" -exec dirname {} \; | head -1)
        if [ -n "$SCULPTED_MODEL" ]; then
            echo "  [$LABEL kf=$KF] Running lm-eval..."
            RUN_LMEVAL_OUT="$RUN_DIR/lm_eval_output"
            lm_eval --model hf \
                --model_args "pretrained=$SCULPTED_MODEL,dtype=bfloat16" \
                --tasks "$LMEVAL_TASKS" \
                --batch_size auto \
                --output_path "$RUN_LMEVAL_OUT" \
                2>&1 | tee "$RUN_DIR/lm_eval.log"
            _collect_lm_eval_results "$RUN_LMEVAL_OUT" "$RUN_DIR/lm_eval_results.json"
            echo "  [$LABEL kf=$KF] lm-eval complete."
        else
            echo "  [$LABEL kf=$KF] WARNING: No sculpted model found, skipping lm-eval."
        fi
    fi
done

# ── Phase 3: Summary ────────────────────────────────────────────────────────

echo ""
echo "=============================================="
echo "  Head-to-Head Benchmark Complete"
echo "=============================================="
echo ""
echo "Published targets (DDP Table 2, LLaMA-2-7B):"
echo "  20% sparsity: Wiki2=14.39 (DDP/SOTA), Mean Acc=64.82%"
echo "  50% sparsity: Wiki2=26.34 (DDP/SOTA), Mean Acc=56.70%"
echo ""
echo "Our results:"
for CONFIG in "${CONFIGS[@]}"; do
    LABEL=$(echo "$CONFIG" | cut -d',' -f1)
    KF=$(echo "$CONFIG" | cut -d',' -f2)
    RUN_DIR="$OUTBASE/${MODEL_SHORT}_${LABEL}_kf${KF}"
    PPL=$(grep -a 'ppl_ratio' "$RUN_DIR/sculpt.log" 2>/dev/null | tail -1 | grep -oP 'ppl_ratio=\S+' || echo "N/A")
    echo "  $LABEL (kf=$KF): $PPL"
done
echo ""
echo "Results directory: $OUTBASE/"
echo "=============================================="
