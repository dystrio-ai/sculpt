#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# A100 Provisioning + Run Script
#
# Usage:
#   export HF_TOKEN="hf_..."
#   export GH_TOKEN="ghp_..."   # or use ssh clone
#   bash provision_a100.sh
#
# What it does:
#   1. Installs system deps + Python 3.10 if needed
#   2. Clones repo (experimental/distill-repair branch)
#   3. Installs dystrio-sculpt + lm-eval
#   4. Runs Gemma 2B test (fast validation, ~20 min)
#   5. If test passes, runs Qwen 3.5 27B flagship (~2-4 hours)
###############################################################################

BRANCH="experimental/distill-repair"
REPO="https://github.com/clusteroptimizerengine/BumbleB.git"
WORKDIR="$HOME/BumbleB"

echo "============================================================"
echo "  Dystrio Sculpt — A100 Provisioning"
echo "  Branch: $BRANCH"
echo "  $(date)"
echo "============================================================"

# ── 0. Preflight checks ─────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Is this a GPU box?"
    exit 1
fi

echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. Dataset push will fail."
    echo "  export HF_TOKEN=hf_..."
fi

# ── 1. System deps ──────────────────────────────────────────────
echo ">> Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-pip python3-venv > /dev/null 2>&1
echo "   done."

# ── 2. Clone repo ──────────────────────────────────────────────
if [ -d "$WORKDIR" ]; then
    echo ">> Repo exists, pulling latest..."
    cd "$WORKDIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo ">> Cloning repo..."
    git clone --branch "$BRANCH" "$REPO" "$WORKDIR"
    cd "$WORKDIR"
fi

# ── 3. Python environment ──────────────────────────────────────
echo ">> Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip setuptools wheel -q
pip install -e ".[dev]" -q
pip install lm-eval>=0.4 huggingface_hub -q
# Qwen3.5 needs bleeding-edge transformers (model_type qwen3_5 not in stable releases)
pip install "transformers @ git+https://github.com/huggingface/transformers.git@main" -q

echo "   Python: $(python --version)"
echo "   torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA:   $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# ── 4. HuggingFace login ──────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
    echo ">> Logging into HuggingFace..."
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
    echo "   done."
fi

# ── 5. Quick sanity check ─────────────────────────────────────
echo ">> Running test suite..."
python -m pytest tests/ -x -q 2>&1 | tail -5
echo ""

# ── 6. Qwen 3.5 9B sculpt run ────────────────────────────────
#
# 2.27M downloads, 108 quantized variants — popular local-inference
# model. Fits A100 80GB with distillation. Threshold 0.85 to build
# a full degradation curve across keep_frac values.
#
echo "============================================================"
echo "  SCULPT: Qwen 3.5 9B (distill-alpha=0.5, cached, threshold=0.85)"
echo "  $(date)"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
dystrio sculpt \
    --model-id Qwen/Qwen3.5-9B \
    --workload general_v2 \
    --distill-alpha 0.5 \
    --distill-cache \
    --frontier 4 \
    --downstream-threshold 0.85 \
    --outdir sculpt_out_qwen35_9b \
    --push-dataset \
    2>&1 | tee qwen35_9b_run.log

TEST_EXIT=$?

echo ""
echo "============================================================"
echo "  RUN COMPLETE"
echo "  $(date)"
echo "  Qwen 3.5 9B:  exit=$TEST_EXIT"
echo "============================================================"

if [ $TEST_EXIT -eq 0 ]; then
    echo ""
    echo ">> Results pushed to dystrio/efficiency-dataset"
    echo ">> Model artifacts in: sculpt_out_qwen35_9b/"
    echo ""
    echo "Next steps:"
    echo "  1. Run lm_eval on frontier points:"
    echo "     lm_eval --model hf --model_args pretrained=sculpt_out_qwen35_9b/frontier_0_default/model \\"
    echo "       --tasks arc_challenge,hellaswag,mmlu,truthfulqa_mc2,winogrande,gsm8k \\"
    echo "       --batch_size auto --device cuda"
    echo ""
    echo "  2. Run the 27B flagship on H200:"
    echo "     bash scripts/run_qwen35_27b.sh"
else
    echo ""
    echo "!! RUN FAILED (exit=$TEST_EXIT). Check qwen35_9b_run.log"
fi
