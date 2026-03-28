#!/usr/bin/env bash
set -euo pipefail

# StarCoder2-15B Sculpt — Provision + Run
# Hardware: Single A100-80GB or H100-80GB
# Estimated runtime: 8-14 hours depending on GPU

REPO_URL="${REPO_URL:-https://github.com/clusteroptimizerengine/BumbleB.git}"
BRANCH="${BRANCH:-experimental/distill-repair}"
OUTDIR="${OUTDIR:-/ephemeral/sculpt_starcoder2_15b}"
FRONTIER="${FRONTIER:-4}"

echo "============================================================"
echo "  StarCoder2-15B Sculpt — Structural Compression"
echo "  Workload:  code_starcoder (bigcode/starcoderdata)"
echo "  Distill:   live teacher (alpha=0.5)"
echo "  Frontier:  ${FRONTIER} points"
echo "  Output:    ${OUTDIR}"
echo "  $(date)"
echo "============================================================"
echo ""

# GPU info
if command -v nvidia-smi &>/dev/null; then
    echo "GPU info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "GPU count: ${GPU_COUNT}"
    echo ""
fi

echo ">> Installing system dependencies..."
sudo apt-get update -qq && sudo apt-get install -y -qq git-lfs > /dev/null 2>&1 || true

echo ">> Setting up repository..."
if [ -d ~/BumbleB ]; then
    echo ">> Repo exists, pulling latest..."
    cd ~/BumbleB
    git checkout "${BRANCH}" 2>/dev/null || true
    git pull origin "${BRANCH}" || true
else
    echo ">> Cloning repository..."
    git clone --branch "${BRANCH}" "${REPO_URL}" ~/BumbleB
    cd ~/BumbleB
fi

echo ">> Setting up Python environment..."
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

pip install -q --upgrade pip
pip install -q -e ".[dev]" 2>/dev/null || pip install -q -e .
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128 2>/dev/null || true
pip install -q vllm lm_eval bitsandbytes 2>/dev/null || true

echo "   Python: $(python3 --version)"
echo "   torch:  $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "   CUDA:   $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"

echo ">> Logging into HuggingFace..."
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "${HF_TOKEN}" 2>/dev/null || \
        python3 -c "from huggingface_hub import login; login('${HF_TOKEN}')" 2>/dev/null || true
    echo "   done."
else
    echo "   WARNING: HF_TOKEN not set. bigcode/starcoderdata is gated — you must"
    echo "   accept The Stack Terms of Use at https://huggingface.co/datasets/bigcode/starcoderdata"
    echo "   and set HF_TOKEN before running."
fi

echo ">> Running unit tests..."
python3 -m pytest tests/ -x -q --ignore=tests/test_engine_contract.py 2>&1 | tail -3

echo ""
echo "============================================================"
echo "  STARTING: StarCoder2-15B Sculpt"
echo "  $(date)"
echo "============================================================"
echo ""

mkdir -p "${OUTDIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-/ephemeral/hf_cache}"

nohup python3 -u -m dystrio_sculpt sculpt \
    --model-id bigcode/starcoder2-15b \
    --workload code_starcoder \
    --distill \
    --distill-alpha 0.5 \
    --no-distill-cache \
    --frontier "${FRONTIER}" \
    --outdir "${OUTDIR}" \
    --downstream-threshold 0.85 \
    > "${OUTDIR}/sculpt_run.log" 2>&1 &

SCULPT_PID=$!
echo ">> Sculpt running in background (nohup)."
echo "   PID: ${SCULPT_PID}"
echo "   Log: ${OUTDIR}/sculpt_run.log"
echo ""
echo "Monitor with:"
echo "   tail -f ${OUTDIR}/sculpt_run.log"
echo ""
echo "Check status:"
echo "   strings ${OUTDIR}/sculpt_run.log | grep -E 'SAFE|UNSAFE|frontier|error|keep_frac|compile done'"
echo ""
echo "Estimated runtime: 8-14 hours on H100-80GB"
echo "                   12-20 hours on A100-80GB"
