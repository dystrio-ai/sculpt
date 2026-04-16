#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Cohesion Selector Proof Run — Llama-3.1-8B-Instruct @ kf=0.85
#
# Minimal GPU run to validate the group-aware cohesion selector against the
# existing ablation data. Targets kf=0.85 — the compression level where the
# old diversity penalty hurt most (GSM8K collapsed from 44.8 → 33.0 vs
# sensitivity baseline).
#
# What it runs:
#   - cohesion selector at kf=0.85 (one sculpt + one lm-eval)
#   - lm-eval with limit=500 (matches existing ablation data)
#   - ~45 min on an A100
#
# Compare results against existing ablation cells:
#   structural kf=0.85:  CS-7=63.6, GSM8K=33.0, PPL=19.57
#   sensitivity kf=0.85: CS-7=63.8, GSM8K=45.0, PPL=19.08
#
# Usage (fresh Ubuntu GPU box):
#   export HF_TOKEN="hf_..."   # needed for Llama-3.1 gated model
#   curl -fsSL <raw_url> | bash
#
#   Or from a box that already has the repo:
#   cd ~/sculpt && bash scripts/provision_cohesion_proof.sh
#
# After run:
#   cat cohesion_proof/ablation_summary.md
#   scp -r <box>:~/sculpt/cohesion_proof/ .
###############################################################################

SCULPT_REPO="${SCULPT_REPO:-https://github.com/dystrio-ai/sculpt.git}"
SCULPT_BRANCH="${SCULPT_BRANCH:-experiment/physarum-cohesion}"
WORKDIR="${WORKDIR:-$HOME/sculpt}"
OUTBASE="cohesion_proof"

echo "============================================================"
echo "  Cohesion Selector Proof Run"
echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo "  Branch:   $SCULPT_BRANCH"
echo "  Model:    meta-llama/Llama-3.1-8B-Instruct"
echo "  Selector: cohesion (group-aware Physarum)"
echo "  KF:       0.85 (15% neurons removed)"
echo "  Eval:     lm-eval limit=500"
echo "  Compare:  structural=63.6 CS-7 / sensitivity=63.8 CS-7"
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

# Detect CUDA driver version and install matching torch
DRIVER_CUDA=$(python3 -c "
import subprocess, re
out = subprocess.check_output(['nvidia-smi'], text=True)
m = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', out)
print(f'{m.group(1)}.{m.group(2)}' if m else 'unknown')
" 2>/dev/null || echo "unknown")

echo "   Driver CUDA: $DRIVER_CUDA"
if python3 -c "v='$DRIVER_CUDA'; major,minor=v.split('.'); exit(0 if int(major)>=12 and int(minor)>=4 else 1)" 2>/dev/null; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
elif python3 -c "v='$DRIVER_CUDA'; major,minor=v.split('.'); exit(0 if int(major)>=12 and int(minor)>=1 else 1)" 2>/dev/null; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
fi
echo "   Torch index: $TORCH_INDEX"
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX" -q

pip install -e ".[dev,viz]" -q
pip install "lm-eval>=0.4" huggingface_hub -q

echo "   Python: $(python --version)"
echo "   torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA:   $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# ── HF login (required for Llama-3.1 gated model) ─────────────
if [ -n "${HF_TOKEN:-}" ]; then
    echo ">> HF login"
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
else
    echo "WARNING: HF_TOKEN not set. Llama-3.1 is gated — this will fail"
    echo "         unless you already have cached credentials."
fi

# ── Quick sanity ───────────────────────────────────────────────
echo ">> pytest (quick)"
python -m pytest tests/ -q -m "not slow" --tb=no -x || {
    echo "WARNING: pytest had failures; continuing anyway."
}
echo ""

# ── Save environment info ──────────────────────────────────────
mkdir -p "$OUTBASE"
nvidia-smi > "$OUTBASE/gpu_info.txt" 2>&1
pip freeze > "$OUTBASE/pip_freeze.txt"
git log --oneline -5 > "$OUTBASE/git_log.txt"
echo "branch: $SCULPT_BRANCH" >> "$OUTBASE/git_log.txt"

# ── Run the proof ──────────────────────────────────────────────
echo "============================================================"
echo "  Starting cohesion proof: kf=0.85"
echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

MODEL="meta-llama/Llama-3.1-8B-Instruct" \
SELECTORS="cohesion" \
KEEP_FRACS="0.85" \
LMEVAL_LIMIT=500 \
OUTBASE="$OUTBASE" \
bash scripts/ablation_study.sh

# ── Summary ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Done.  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "  Results: $WORKDIR/$OUTBASE/"
echo "  Summary: $WORKDIR/$OUTBASE/ablation_summary.md"
echo ""
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │  COMPARE AGAINST EXISTING ABLATION:                 │"
echo "  │                                                     │"
echo "  │  structural kf=0.85:  CS-7=63.6  GSM8K=33.0        │"
echo "  │  sensitivity kf=0.85: CS-7=63.8  GSM8K=45.0        │"
echo "  │                                                     │"
echo "  │  Cohesion wins if:                                  │"
echo "  │    GSM8K > 45.0  (beats sensitivity)                │"
echo "  │    CS-7  > 63.8  (beats sensitivity)                │"
echo "  └─────────────────────────────────────────────────────┘"
echo ""
echo "  Copy results off the box:"
echo "    scp -r <box>:~/sculpt/$OUTBASE/ ."
echo "============================================================"
