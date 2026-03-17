#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=~/BumbleB/src
export HF_TOKEN="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"

TASKS="arc_challenge,hellaswag,mmlu,truthfulqa_mc2"
BATCH="auto"

ZOO_DIR="/data/zoo_ab"
EVAL_LOG="/data/zoo_ab/eval_ab.log"

MODELS=(
  "google_gemma-2-2b-it"
  "mistralai_Mistral-7B-Instruct-v0.3"
  "meta-llama_Llama-3.2-3B-Instruct"
  "Qwen_Qwen2.5-3B-Instruct"
  "meta-llama_Llama-3.1-8B-Instruct"
)

VARIANTS=("nodistill" "distill")

for m in "${MODELS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    RUN_DIR="${ZOO_DIR}/${m}_${v}"
    if [ ! -d "$RUN_DIR" ]; then
      echo "$(date) SKIP: $RUN_DIR does not exist" | tee -a "$EVAL_LOG"
      continue
    fi

    for tier_dir in "$RUN_DIR"/frontier_*/; do
      [ -d "$tier_dir" ] || continue
      tier=$(basename "$tier_dir")
      MODEL_DIR="${tier_dir}model"
      EVAL_OUT="${tier_dir}evals"

      if [ ! -d "$MODEL_DIR" ]; then
        echo "$(date) SKIP: $MODEL_DIR does not exist" | tee -a "$EVAL_LOG"
        continue
      fi

      if [ -d "$EVAL_OUT" ] && [ "$(ls -A "$EVAL_OUT" 2>/dev/null)" ]; then
        echo "$(date) SKIP: $EVAL_OUT already has results" | tee -a "$EVAL_LOG"
        continue
      fi

      echo "$(date) START: ${m}_${v} / ${tier}" | tee -a "$EVAL_LOG"
      mkdir -p "$EVAL_OUT"

      lm_eval --model hf \
        --model_args "pretrained=${MODEL_DIR},dtype=bfloat16,trust_remote_code=True" \
        --tasks "$TASKS" \
        --batch_size "$BATCH" \
        --output_path "$EVAL_OUT" \
        2>&1 | tee -a "$EVAL_LOG"

      echo "$(date) DONE: ${m}_${v} / ${tier}" | tee -a "$EVAL_LOG"
    done
  done
done

echo "$(date) === ALL AB EVALS COMPLETE ===" | tee -a "$EVAL_LOG"
