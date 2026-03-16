#!/usr/bin/env bash
set -euo pipefail

export HF_TOKEN="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"

PAIRS=(
  "dystrio/gemma-2-2b-it-sculpt-production gemma_production"
  "dystrio/gemma-2-2b-it-sculpt-throughput gemma_throughput"
  "dystrio/gemma-2-2b-it-sculpt-experimental gemma_experimental"
  "dystrio/Llama-3.2-3B-Instruct-sculpt-production llama32_production"
  "dystrio/Llama-3.2-3B-Instruct-sculpt-throughput llama32_throughput"
  "dystrio/Llama-3.2-3B-Instruct-sculpt-experimental llama32_experimental"
  "dystrio/Qwen2.5-3B-Instruct-sculpt-production qwen3b_production"
  "dystrio/Qwen2.5-3B-Instruct-sculpt-throughput qwen3b_throughput"
  "dystrio/Qwen2.5-3B-Instruct-sculpt-experimental qwen3b_experimental"
  "dystrio/Mistral-7B-Instruct-v0.3-sculpt-production mistral7b_production"
  "dystrio/Mistral-7B-Instruct-v0.3-sculpt-throughput mistral7b_throughput"
  "dystrio/Mistral-7B-Instruct-v0.3-sculpt-experimental mistral7b_experimental"
  "dystrio/Llama-3.1-8B-Instruct-sculpt-default llama31_sculpt"
  "dystrio/Llama-3.1-8B-Instruct-sculpt-production llama31_production"
  "dystrio/Llama-3.1-8B-Instruct-sculpt-throughput llama31_throughput"
  "dystrio/Llama-3.1-8B-Instruct-sculpt-experimental llama31_experimental"
)

for pair in "${PAIRS[@]}"; do
  model=$(echo "$pair" | cut -d" " -f1)
  name=$(echo "$pair" | cut -d" " -f2)

  # Clear HF cache before each run to prevent disk full
  echo "=== $(date) Clearing HF cache ==="
  rm -rf ~/.cache/huggingface/hub
  mkdir -p ~/.cache/huggingface/hub
  echo "=== $(date) Disk free: $(df -h / | tail -1 | awk '{print $4}') ==="

  echo "=== $(date) Starting: $model ==="

  lm_eval --model hf \
    --model_args "pretrained=$model,dtype=bfloat16" \
    --tasks mmlu,hellaswag,arc_challenge,truthfulqa_mc2 \
    --batch_size auto \
    --output_path "/data/eval/$name"

  echo "=== $(date) Finished: $model (exit code: $?) ==="
done

echo "=== $(date) ALL TIER EVALS COMPLETE ==="
