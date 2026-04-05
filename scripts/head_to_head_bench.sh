#!/usr/bin/env bash
set -uo pipefail

# =============================================================================
# Head-to-Head Benchmark: Sculpt Frontier Curve vs Published Methods
#
# Runs Sculpt across a sweep of keep_fracs to map the full quality-vs-
# compression curve, then compares against published structured pruning
# results from DDP (ICML'26), SlimLLM (ICML'25), LoRAP, etc.
#
# Published targets (DDP Table 2, LLaMA-2-7B, 20% total sparsity):
#   DDP:     Wiki2=14.39, Mean Acc=64.82%
#   SlimLLM: Wiki2=15.28, Mean Acc=61.70%
#   LoRAP:   Wiki2=14.67, Mean Acc=61.20%
#
# Sculpt prunes MLP channels only (not attention heads). Parameter mapping:
#   LLaMA-2-7B: MLP = 64.3% of params
#   kf=0.90 → ~6.4% total reduction    kf=0.60 → ~25.7% total reduction
#   kf=0.80 → ~12.9% total reduction   kf=0.50 → ~32.1% total reduction
#   kf=0.69 → ~20.0% total reduction   kf=0.40 → ~38.6% total reduction
#
# Usage:
#   bash scripts/head_to_head_bench.sh                # full run with lm-eval
#   SKIP_LMEVAL=1 bash scripts/head_to_head_bench.sh  # perplexity sweep only
# =============================================================================

MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
WORKLOAD="${WORKLOAD:-general_v2}"
OUTBASE="${OUTBASE:-h2h_results}"
SKIP_LMEVAL="${SKIP_LMEVAL:-0}"

# Sweep from light to aggressive compression
KEEP_FRACS="${KEEP_FRACS:-0.85,0.80,0.75,0.69,0.60,0.50}"

# DDP paper eval: 9 zero-shot tasks via lm-eval harness
LMEVAL_TASKS="arc_easy,arc_challenge,openbookqa,winogrande,piqa,hellaswag,mathqa,rte,boolq"

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')

echo "=============================================="
echo "  Head-to-Head Frontier Sweep"
echo "=============================================="
echo "  Model:      $MODEL"
echo "  Keep fracs: $KEEP_FRACS"
echo "  Workload:   $WORKLOAD"
echo "  Output:     $OUTBASE/"
echo "  lm-eval:    $([ "$SKIP_LMEVAL" = "1" ] && echo "SKIPPED" || echo "$LMEVAL_TASKS")"
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

# ── Phase 1: Baseline ───────────────────────────────────────────────────────

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
    echo "  Baseline lm-eval complete."
else
    echo ""
    echo "[Phase 1] Baseline eval exists or skipped."
fi

# ── Phase 2: Sculpt frontier sweep ──────────────────────────────────────────

echo ""
echo "[Phase 2] Sculpt frontier sweep"

IFS=',' read -ra KF_ARRAY <<< "$KEEP_FRACS"

for KF in "${KF_ARRAY[@]}"; do
    RUN_DIR="$OUTBASE/${MODEL_SHORT}_kf${KF}"

    if [ -d "$RUN_DIR" ] && [ -f "$RUN_DIR/run_metadata.json" ]; then
        echo "  [kf=$KF] Already complete, skipping."
        continue
    fi

    echo ""
    echo "  [kf=$KF] Starting sculpt..."
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
        echo "  [kf=$KF] Sculpt complete."
    else
        echo "  [kf=$KF] Sculpt FAILED (exit $?) — recorded."
        echo '{"failed": true, "keep_frac": '"$KF"'}' > "$RUN_DIR/run_metadata.json"
    fi

    # lm-eval on sculpted model
    if [ "$SKIP_LMEVAL" != "1" ]; then
        SCULPTED_MODEL=$(find "$RUN_DIR" -name "config.json" -path "*/model/*" -exec dirname {} \; | head -1)
        if [ -n "$SCULPTED_MODEL" ]; then
            echo "  [kf=$KF] Running lm-eval..."
            RUN_LMEVAL_OUT="$RUN_DIR/lm_eval_output"
            lm_eval --model hf \
                --model_args "pretrained=$SCULPTED_MODEL,dtype=bfloat16" \
                --tasks "$LMEVAL_TASKS" \
                --batch_size auto \
                --output_path "$RUN_LMEVAL_OUT" \
                2>&1 | tee "$RUN_DIR/lm_eval.log"
            _collect_lm_eval_results "$RUN_LMEVAL_OUT" "$RUN_DIR/lm_eval_results.json"
            echo "  [kf=$KF] lm-eval complete."
        else
            echo "  [kf=$KF] WARNING: No sculpted model found, skipping lm-eval."
        fi
    fi
done

# ── Phase 3: Summary table ──────────────────────────────────────────────────

echo ""
echo "=============================================="
echo "  Frontier Sweep Results"
echo "=============================================="
echo ""
printf "%-8s %-12s %-14s %s\n" "kf" "total_reduc" "ppl_ratio" "status"
printf "%-8s %-12s %-14s %s\n" "----" "----------" "---------" "------"
for KF in "${KF_ARRAY[@]}"; do
    RUN_DIR="$OUTBASE/${MODEL_SHORT}_kf${KF}"
    # MLP is 64.3% of LLaMA-2-7B params
    TOTAL_REDUC=$(python3 -c "print(f'{(1-$KF)*64.3:.1f}%')" 2>/dev/null || echo "?")
    PPL=$(grep -a 'ppl_ratio' "$RUN_DIR/sculpt.log" 2>/dev/null | tail -1 | grep -oP 'ppl_ratio=\S+' || echo "N/A")
    FAILED=$(grep -a 'failed=True' "$RUN_DIR/sculpt.log" 2>/dev/null | tail -1)
    if [ -n "$FAILED" ]; then
        STATUS="FAILED"
    else
        STATUS="ok"
    fi
    printf "%-8s %-12s %-14s %s\n" "$KF" "$TOTAL_REDUC" "$PPL" "$STATUS"
done

echo ""
echo "Published comparison points (LLaMA-2-7B, 20% total sparsity):"
echo "  DDP (ICML'26 SOTA):  Wiki2=14.39  Acc=64.82%"
echo "  SlimLLM (ICML'25):   Wiki2=15.28  Acc=61.70%"
echo "  LoRAP:               Wiki2=14.67  Acc=61.20%"
echo "  Dense baseline:      Wiki2=12.18  Acc=66.63%"
echo ""
echo "  Sculpt kf=0.69 → ~20% total reduction (direct comparison point)"
echo ""
echo "=============================================="
