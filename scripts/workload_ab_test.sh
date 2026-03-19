#!/usr/bin/env bash
# Workload A/B test: compare workload-aware repair vs WikiText-only repair.
#
# Uses 2 small models (fastest to sculpt) with distillation enabled.
# Runs 3 variants each:
#   1. general (WikiText — matches existing zoo A/B results)
#   2. code   (bigcode/the-stack-smol)
#   3. chat   (tatsu-lab/alpaca)
#
# Then runs lm-eval on all outputs for apples-to-apples downstream comparison.
#
# Run on A100:
#   git pull && pip install -e .
#   pip install bitsandbytes
#   nohup bash scripts/workload_ab_test.sh > workload_ab.log 2>&1 &

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${TMPDIR:-/data/tmp}"
export HF_HOME="${HF_HOME:-/data/hf_cache}"
mkdir -p "$TMPDIR" "$HF_HOME"

OUTBASE="/data/workload_ab"
LOG="$OUTBASE/workload_ab.log"
mkdir -p "$OUTBASE"

MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
    "google/gemma-2-2b-it"
)

WORKLOADS=("general" "code" "chat")

EVAL_TASKS="arc_challenge,hellaswag,mmlu,truthfulqa_mc2,winogrande,gsm8k"

echo "$(date) === WORKLOAD A/B TEST START ===" | tee -a "$LOG"

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
                --frontier 3 \
                --no-push-dataset \
                2>&1 | tail -20 | tee -a "$LOG"

            echo "$(date) DONE sculpt: $safe / $wl" | tee -a "$LOG"
        fi

        # Run lm-eval on each frontier point
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

echo "" | tee -a "$LOG"
echo "$(date) === ALL SCULPT + EVAL COMPLETE ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Print comparison table
echo "$(date) === RESULTS COMPARISON ===" | tee -a "$LOG"
python3 -c "
import json, glob

print(f\"{'Model':<35} {'Workload':<10} {'Tier':<28} {'KF':>5} {'ARC':>6} {'HS':>6} {'MMLU':>6} {'TQA':>6}\")
print('-' * 110)

for f in sorted(glob.glob('$OUTBASE/*/frontier_*/evals/**/results_*.json', recursive=True)):
    path = f.split('$OUTBASE/')[1]
    parts = path.split('/')
    variant = parts[0]
    tier = parts[1]

    # Parse model and workload from variant name
    # e.g. Qwen_Qwen2.5-3B-Instruct_code -> model=Qwen/Qwen2.5-3B-Instruct, wl=code
    last_underscore = variant.rfind('_')
    wl = variant[last_underscore+1:]
    model_safe = variant[:last_underscore]

    with open(f) as fh:
        d = json.load(fh)
    r = d.get('results', {})
    arc = r.get('arc_challenge', {}).get('acc_norm,none', 0)
    hs = r.get('hellaswag', {}).get('acc_norm,none', 0)
    mmlu = r.get('mmlu', {}).get('acc,none', 0)
    tqa = r.get('truthfulqa_mc2', {}).get('acc,none', 0)
    
    # Get keep_frac from summary
    import csv
    kf = '?'
    summary_csv = '$OUTBASE/' + variant + '/summary.csv'
    try:
        with open(summary_csv) as cf:
            for row in csv.DictReader(cf):
                if row.get('name') == tier:
                    kf = row['keep_frac']
                    break
    except: pass

    print(f'{model_safe:<35} {wl:<10} {tier:<28} {kf:>5} {arc:>6.4f} {hs:>6.4f} {mmlu:>6.4f} {tqa:>6.4f}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "$(date) === DONE ===" | tee -a "$LOG"
