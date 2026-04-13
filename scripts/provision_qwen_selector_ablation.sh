#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Qwen2.5-3B Selector Ablation — Physarum vs Sensitivity on a High-Coupling Model
#
# Qwen2.5-3B-Instruct has aggregate risk 0.60 (vs Llama-3.1-8B's 0.30).
# This is the minimal experiment to test whether Physarum's conductance-based
# diversity penalty earns its keep on a model with real inter-block coupling.
#
# What it runs:
#   - structural vs sensitivity selector at kf=0.88 and kf=0.70
#   - lm-eval with limit=1000 (tighter error bars than the Llama ablation)
#   - 4 sculpt+eval cells total, ~60-90 min on an A100
#
# Usage (fresh Ubuntu GPU box):
#   curl the script or scp it, then:
#     bash provision_qwen_selector_ablation.sh
#
#   Or from a box that already has the repo:
#     cd ~/sculpt && bash scripts/provision_qwen_selector_ablation.sh
#
# No HF_TOKEN needed — Qwen2.5-3B-Instruct is fully public.
###############################################################################

SCULPT_REPO="${SCULPT_REPO:-https://github.com/dystrio-ai/sculpt.git}"
SCULPT_BRANCH="${SCULPT_BRANCH:-main}"
WORKDIR="${WORKDIR:-$HOME/sculpt}"

echo "============================================================"
echo "  Qwen2.5-3B Selector Ablation"
echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo "  Purpose: Physarum vs sensitivity on a high-coupling model"
echo "  Model:   Qwen/Qwen2.5-3B-Instruct (risk=0.60, public)"
echo "  Cells:   structural x {0.88, 0.70} + sensitivity x {0.88, 0.70}"
echo "  Eval:    lm-eval limit=1000"
echo "============================================================"

# ── Preflight ──────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Use a GPU image."
    exit 1
fi
echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── System deps ────────────────────────────────────────────────
echo ">> apt: git, venv, pip"
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-venv python3-pip > /dev/null 2>&1

# ── Clone / pull ───────────────────────────────────────────────
if [ -d "$WORKDIR/.git" ]; then
    echo ">> repo exists, pulling"
    git -C "$WORKDIR" fetch origin
    git -C "$WORKDIR" checkout "$SCULPT_BRANCH"
    git -C "$WORKDIR" pull origin "$SCULPT_BRANCH"
else
    echo ">> cloning repo"
    git clone --branch "$SCULPT_BRANCH" "$SCULPT_REPO" "$WORKDIR"
fi
cd "$WORKDIR"

# ── Python venv ────────────────────────────────────────────────
echo ">> venv"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

pip install -U pip wheel -q
pip install "setuptools>=68,<82" -q
pip install -e ".[dev,viz]" -q
pip install "lm-eval>=0.4" huggingface_hub -q

echo "   Python: $(python --version)"
echo "   torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA:   $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# ── HF login (optional for Qwen, but needed for push-dataset) ─
if [ -n "${HF_TOKEN:-}" ]; then
    echo ">> HF login"
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
fi

# ── Quick sanity ───────────────────────────────────────────────
echo ">> pytest (quick)"
python -m pytest tests/ -q -m "not slow" --tb=no -x || {
    echo "WARNING: pytest had failures; continuing anyway."
}
echo ""

# ── Run the ablation ───────────────────────────────────────────
echo "============================================================"
echo "  Starting selector ablation: structural vs sensitivity"
echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

MODEL="Qwen/Qwen2.5-3B-Instruct" \
SELECTORS="structural sensitivity" \
KEEP_FRACS="0.88,0.70" \
LMEVAL_LIMIT=1000 \
OUTBASE="qwen_selector_ablation" \
bash scripts/ablation_study.sh

# ── Visualize ──────────────────────────────────────────────────
echo ""
echo ">> Generating charts + summary"
python3 scripts/visualize_ablation.py qwen_selector_ablation/ \
    --model Qwen2.5-3B-Instruct --csv || true

echo ""
echo "============================================================"
echo "  Done.  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "  Results:  $WORKDIR/qwen_selector_ablation/"
echo "  Summary:  $WORKDIR/qwen_selector_ablation/ablation_summary.md"
echo "  CSV:      $WORKDIR/qwen_selector_ablation/ablation_chart_data.csv"
echo ""
echo "  Key comparison:"
echo "    cat qwen_selector_ablation/ablation_summary.md"
echo ""
echo "  Copy results off the box:"
echo "    scp -r <box>:~/sculpt/qwen_selector_ablation/ ."
echo "============================================================"
