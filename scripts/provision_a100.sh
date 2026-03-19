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
python -m pytest tests/ -x -q --timeout=120 2>&1 | tail -5
echo ""

# ── 6. Gemma 2B test run (validation) ─────────────────────────
echo "============================================================"
echo "  STAGE 1: Gemma 2B validation run"
echo "  $(date)"
echo "============================================================"

dystrio sculpt \
    --model-id google/gemma-2-2b-it \
    --workload general_v2 \
    --distill-alpha 0.5 \
    --frontier 3 \
    --downstream-threshold 0.95 \
    --outdir sculpt_out_gemma2b_test \
    --no-push-dataset \
    2>&1 | tee gemma2b_test.log

GEMMA_EXIT=$?

if [ $GEMMA_EXIT -ne 0 ]; then
    echo ""
    echo "!! Gemma 2B test FAILED (exit=$GEMMA_EXIT). Stopping."
    echo "!! Check gemma2b_test.log for details."
    exit 1
fi

echo ""
echo ">> Gemma 2B test PASSED."
echo ""

# ── 7. Qwen 3.5 27B flagship run ─────────────────────────────
echo "============================================================"
echo "  STAGE 2: Qwen 2.5 32B flagship run"
echo "  $(date)"
echo "============================================================"
echo ""
echo "  NOTE: Qwen3.5-27B uses Gated DeltaNet (non-standard arch)"
echo "  and needs a dedicated adapter. Using Qwen2.5-32B-Instruct"
echo "  which is fully supported (SwiGLU dense, qwen2 model_type)."
echo ""

dystrio sculpt \
    --model-id Qwen/Qwen2.5-32B-Instruct \
    --workload general_v2 \
    --distill-alpha 0.5 \
    --frontier 4 \
    --downstream-threshold 0.95 \
    --outdir sculpt_out_qwen32b \
    --push-dataset \
    2>&1 | tee qwen32b_run.log

QWEN_EXIT=$?

echo ""
echo "============================================================"
echo "  RUN COMPLETE"
echo "  $(date)"
echo "  Gemma 2B:  exit=$GEMMA_EXIT"
echo "  Qwen 32B:  exit=$QWEN_EXIT"
echo "============================================================"

if [ $QWEN_EXIT -eq 0 ]; then
    echo ""
    echo ">> Results pushed to dystrio/efficiency-dataset"
    echo ">> Model artifacts in: sculpt_out_qwen32b/"
    echo ""
    echo "Next: run lm_eval on the frontier points:"
    echo "  lm_eval --model hf --model_args pretrained=sculpt_out_qwen32b/frontier_0_default/model \\"
    echo "    --tasks arc_challenge,hellaswag,mmlu,truthfulqa_mc2,winogrande,gsm8k \\"
    echo "    --batch_size auto --device cuda"
fi
