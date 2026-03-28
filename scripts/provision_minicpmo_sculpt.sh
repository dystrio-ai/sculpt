#!/usr/bin/env bash
# Provision an A100-80GB to sculpt openbmb/MiniCPM-o-4_5
# Target: prune Qwen3-8B LLM backbone to free VRAM for RTX 3070 Mobile (8GB)
#
# Usage:
#   export HF_TOKEN="hf_..."
#   export GITHUB_PAT="ghp_..."   # optional, for git clone auth
#   bash scripts/provision_minicpmo_sculpt.sh
set -euo pipefail

MODEL_ID="openbmb/MiniCPM-o-4_5"
WORKLOAD="multimodal_description"
OUTDIR="/ephemeral/sculpt_minicpmo"
LOG="$OUTDIR/sculpt_run.log"

echo "============================================================"
echo "  MiniCPM-o 4.5 Sculpt — LLM backbone pruning"
echo "  Model: $MODEL_ID"
echo "  Workload: $WORKLOAD"
echo "  Output: $OUTDIR"
echo "  $(date)"
echo "============================================================"

# ── Environment setup ─────────────────────────────────────────────
if [ ! -d ~/BumbleB ]; then
    echo ">> Cloning BumbleB..."
    if [ -n "${GITHUB_PAT:-}" ]; then
        git clone --branch experimental/distill-repair \
            "https://${GITHUB_PAT}@github.com/clusteroptimizerengine/BumbleB.git" ~/BumbleB
    else
        git clone --branch experimental/distill-repair \
            https://github.com/clusteroptimizerengine/BumbleB.git ~/BumbleB
    fi
else
    echo ">> Pulling latest..."
    cd ~/BumbleB && git pull || true
fi

cd ~/BumbleB

echo ">> Setting up Python environment..."
pip install -e ".[dev]" -q 2>&1 | tail -5
pip install bitsandbytes -q

# Show environment
python3 -c "
import torch, sys
print(f'   Python: {sys.version.split()[0]}')
print(f'   torch:  {torch.__version__}')
print(f'   CUDA:   {torch.version.cuda}')
print(f'   GPUs:   {torch.cuda.device_count()}')
try:
    import vllm; print(f'   vLLM:   {vllm.__version__}')
except: pass
"

# ── HuggingFace auth ──────────────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
    echo ">> Logging into HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN" 2>&1 | tail -1
fi

# ── Tests ─────────────────────────────────────────────────────────
echo ">> Running unit tests..."
python3 -m pytest tests/test_minicpm_adapter.py tests/test_starcoder2_adapter.py -q

# ── Run sculpt ────────────────────────────────────────────────────
mkdir -p "$OUTDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/ephemeral/hf_cache

echo ""
echo ">> Starting sculpt run..."
echo "   Model:     $MODEL_ID"
echo "   Workload:  $WORKLOAD"
echo "   Distill:   enabled (alpha=0.5, live teacher)"
echo "   Threshold: 0.85"
echo "   Frontier:  4 points"
echo ""

nohup python -u -m dystrio_sculpt sculpt \
    --model-id "$MODEL_ID" \
    --workload "$WORKLOAD" \
    --distill --distill-alpha 0.5 --no-distill-cache \
    --frontier 4 \
    --outdir "$OUTDIR" \
    --downstream-threshold 0.85 \
    > "$LOG" 2>&1 &

PID=$!
echo ">> Sculpt running in background (nohup)."
echo "   PID: $PID"
echo "   Log: $LOG"
echo ""
echo "Monitor with:"
echo "   tail -f $LOG"
echo ""
echo "Check progress:"
echo "   strings $LOG | grep -E 'baseline|prescan|keep_frac|SAFE|UNSAFE|error' | tail -20"
echo ""
echo "Estimated runtime: 4-8 hours (40 layers with live teacher)"
