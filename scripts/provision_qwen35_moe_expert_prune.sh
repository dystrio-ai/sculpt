#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# A100 80GB Provisioning — Qwen3.5-35B-A3B Expert Pruning
#
# Usage:
#   export HF_TOKEN="hf_..."
#   bash scripts/provision_qwen35_moe_expert_prune.sh
#
# What it does:
#   1. Installs system deps + Python env
#   2. Clones repo (main branch)
#   3. Installs dystrio-sculpt + lm-eval
#   4. Runs Qwen3.5-35B-A3B expert pruning (keep_frac=0.80 → ~51 of 64 experts)
#
# Memory budget: ~35B active params at bf16 ≈ 70GB.  A100 80GB fits this
# with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True and no teacher cache.
###############################################################################

BRANCH="main"
REPO="${SCULPT_REPO:-https://github.com/dystrio/sculpt.git}"
WORKDIR="$HOME/BumbleB"
MODEL_ID="Qwen/Qwen3.5-35B-A3B"
OUTDIR="sculpt_out_qwen35_35b_moe"

echo "============================================================"
echo "  Dystrio Sculpt — Qwen3.5-35B-A3B Expert Pruning"
echo "  Branch: $BRANCH"
echo "  Model:  $MODEL_ID"
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

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${GPU_MEM}" -lt 75000 ]; then
    echo "WARNING: GPU has ${GPU_MEM}MiB — 80GB recommended for 35B model."
    echo "         Proceeding anyway, but OOM is possible."
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. Model download / dataset push may fail."
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

# ── 5. Sculpt: Expert Pruning ─────────────────────────────────
#
# Qwen3.5-35B-A3B: 64 experts per layer, top-4 routing.
# keep_frac=0.80 → keep ~51 experts → ~28B active params.
# No distillation teacher cache (saves ~35GB VRAM).
# No repair (expert merging via Physarum coupling is the recovery mechanism).
#
echo "============================================================"
echo "  SCULPT: $MODEL_ID Expert Pruning (keep_frac=0.80)"
echo "  $(date)"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
dystrio sculpt \
    --model-id "$MODEL_ID" \
    --workload general_v2 \
    --keep-frac 0.80 \
    --frontier 2 \
    --downstream-threshold 0.80 \
    --outdir "$OUTDIR" \
    --push-dataset \
    2>&1 | tee "${OUTDIR}_run.log"

RUN_EXIT=$?

echo ""
echo "============================================================"
echo "  RUN COMPLETE"
echo "  $(date)"
echo "  $MODEL_ID:  exit=$RUN_EXIT"
echo "============================================================"

if [ $RUN_EXIT -eq 0 ]; then
    echo ""
    echo ">> Results pushed to dystrio/efficiency-dataset"
    echo ">> Model artifacts in: $OUTDIR/"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate frontier points with lm_eval:"
    echo "     lm_eval --model hf --model_args pretrained=${OUTDIR}/frontier_0_default/model \\"
    echo "       --tasks arc_challenge,hellaswag,mmlu,truthfulqa_mc2,winogrande,gsm8k \\"
    echo "       --batch_size auto --device cuda"
    echo ""
    echo "  2. If quality is acceptable, run aggressive pruning:"
    echo "     dystrio sculpt --model-id $MODEL_ID --keep-frac 0.65 --outdir ${OUTDIR}_aggressive"
else
    echo ""
    echo "!! RUN FAILED (exit=$RUN_EXIT). Check ${OUTDIR}_run.log"
fi
