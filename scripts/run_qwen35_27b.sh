#!/usr/bin/env bash
# Qwen 3.5 27B sculpt — WITH distillation + workload-aware repair.
#
# The teacher model is loaded in 8-bit quantization (~27GB) instead of
# a full-precision deepcopy (~54GB), so distillation fits on H200-141GB.
#
# Prerequisites on H200:
#   git pull && pip install -e .
#   pip install bitsandbytes   # required for 8-bit teacher
#
# Usage:
#   bash scripts/run_qwen35_27b.sh

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG="sculpt_qwen35_27b.log"

echo "$(date) starting Qwen 3.5 27B sculpt (8-bit teacher distillation)" | tee "$LOG"

nohup dystrio sculpt \
    --model-id Qwen/Qwen3.5-27B \
    --outdir ./Qwen3.5-27B-sculpt \
    --workload code \
    --distill-alpha 0.5 \
    --no-push-dataset \
    >> "$LOG" 2>&1 &

PID=$!
echo "$(date) launched PID=$PID — logs: tail -f $LOG"
