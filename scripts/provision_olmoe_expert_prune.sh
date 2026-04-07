#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# A100 80GB — OLMoE-1B-7B Expert Pruning Test
#
# Usage:
#   export HF_TOKEN="hf_..."
#   bash scripts/provision_olmoe_expert_prune.sh
#
# OLMoE-1B-7B: 64 experts/layer, top-8, 16 layers, 7B total / 1B active.
# Same expert/routing ratio as Qwen3.5-35B-A3B (just 4x fewer experts).
#
# Test plan:
#   1. Baseline lm-eval (arc_challenge, hellaswag, winogrande, mmlu, truthfulqa)
#   2. Sculpt with keep_frac=0.80 (64 → 52 experts)
#   3. Post-prune lm-eval on the same tasks
#   4. Compare
###############################################################################

BRANCH="main"
REPO="${SCULPT_REPO:-https://github.com/dystrio-ai/sculpt.git}"
WORKDIR="$HOME/BumbleB"
MODEL_ID="allenai/OLMoE-1B-7B-0924"
OUTDIR="sculpt_out_olmoe_expert_prune"

echo "============================================================"
echo "  Dystrio Sculpt — OLMoE Expert Pruning Test"
echo "  Branch: $BRANCH"
echo "  Model:  $MODEL_ID"
echo "  $(date)"
echo "============================================================"

# ── 0. Preflight ────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found."
    exit 1
fi
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── 1. System deps ──────────────────────────────────────────────
echo ">> Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-pip python3-venv > /dev/null 2>&1
echo "   done."

# ── 2. Clone / update repo ─────────────────────────────────────
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
pip install "lm-eval>=0.4" huggingface_hub -q

echo "   Python: $(python --version)"
echo "   torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA:   $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# ── 4. HuggingFace login ──────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
    echo ">> Logging into HuggingFace..."
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
fi

# ── 5. Quick test ──────────────────────────────────────────────
echo ">> Smoke test..."
python -m pytest tests/test_moe_adapter.py -x -q 2>&1 | tail -3
echo ""

# ── 6. Baseline lm-eval ───────────────────────────────────────
TASKS="arc_challenge,hellaswag,winogrande,truthfulqa_mc2"
BASELINE_DIR="${OUTDIR}/baseline_eval"
mkdir -p "$BASELINE_DIR"

echo "============================================================"
echo "  BASELINE: lm-eval on $MODEL_ID"
echo "  Tasks: $TASKS"
echo "  $(date)"
echo "============================================================"

lm_eval --model hf \
    --model_args "pretrained=${MODEL_ID},dtype=bfloat16" \
    --tasks "$TASKS" \
    --batch_size auto \
    --device cuda \
    --output_path "$BASELINE_DIR" \
    2>&1 | tee "${BASELINE_DIR}/eval.log"

echo ""
echo ">> Baseline eval complete."
echo ""

# ── 7. Sculpt: Expert Pruning ─────────────────────────────────
#
# 64 experts → 52 (keep_frac=0.80)
# Full Physarum pipeline: output covariance → conductance → diversity selection
# Expert merging: dropped experts lerped into most-coupled survivors
#
echo "============================================================"
echo "  SCULPT: $MODEL_ID (keep_frac=0.80, 64→~52 experts)"
echo "  $(date)"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
dystrio sculpt \
    --model-id "$MODEL_ID" \
    --workload general_v2 \
    --frontier 2 \
    --downstream-threshold 0.85 \
    --outdir "$OUTDIR" \
    --no-push-dataset \
    2>&1 | tee "${OUTDIR}/sculpt.log"

SCULPT_EXIT=$?

if [ $SCULPT_EXIT -ne 0 ]; then
    echo "!! SCULPT FAILED (exit=$SCULPT_EXIT). Check ${OUTDIR}/sculpt.log"
    exit $SCULPT_EXIT
fi

# ── 8. Post-prune lm-eval ────────────────────────────────────
PRUNED_DIR=$(ls -d "${OUTDIR}"/frontier_*/model 2>/dev/null | head -1)
if [ -z "$PRUNED_DIR" ]; then
    echo "!! No pruned model found in $OUTDIR"
    exit 1
fi

POST_DIR="${OUTDIR}/post_eval"
mkdir -p "$POST_DIR"

echo "============================================================"
echo "  POST-PRUNE: lm-eval on $PRUNED_DIR"
echo "  Tasks: $TASKS"
echo "  $(date)"
echo "============================================================"

lm_eval --model hf \
    --model_args "pretrained=${PRUNED_DIR},dtype=bfloat16" \
    --tasks "$TASKS" \
    --batch_size auto \
    --device cuda \
    --output_path "$POST_DIR" \
    2>&1 | tee "${POST_DIR}/eval.log"

# ── 9. Summary ────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  COMPLETE"
echo "  $(date)"
echo "============================================================"
echo ""
echo "Baseline results: $BASELINE_DIR"
echo "Pruned model:     $PRUNED_DIR"
echo "Post-prune eval:  $POST_DIR"
echo ""
echo "Compare:"
echo "  diff <(grep acc ${BASELINE_DIR}/eval.log) <(grep acc ${POST_DIR}/eval.log)"
