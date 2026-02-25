"""CLI tests for the multilayer experiment harness.

Fast tests (no model download):
    pytest tests/test_multilayer_cli.py -m "not slow" -v

Full smoke test (downloads Qwen2-0.5B, ~5 min on GPU):
    pytest tests/test_multilayer_cli.py -m slow -v
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "multilayer_experiment.py"


# ── Fast tests (no model download) ────────────────────────────────────────────


def test_help_exits_zero():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0


def test_help_shows_all_flags():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    for flag in [
        "--smoke", "--phases", "--outdir",
        "--grad-accum-steps", "--curve-every", "--curve-eval-texts",
        "--curve-max-eval-tokens", "--n-ood-texts", "--max-eval-tokens-ood",
        "--skip-ablations", "--enable-vllm",
        "--strike-gold", "--gold-repair-steps", "--gold-ppl-guardrail",
        "--gold-early-stop-patience",
        "--staged", "--stage-size", "--stage-repair-steps", "--stage-guardrail",
        "--final-repair-steps",
        "--selector",
    ]:
        assert flag in result.stdout, f"--help missing flag: {flag}"


def test_phase2_without_phase1_errors():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--phases", "0,2"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "phase 2 requires phase 1" in result.stderr.lower()


def test_phase3_without_phase2_errors():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--phases", "0,1,3"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "phase 3 requires phase 2" in result.stderr.lower()


def test_phase1_without_baseline_errors():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--phases", "1"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "phase 0" in result.stderr.lower()


def test_strike_gold_flag_in_help():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "strike-gold" in result.stdout.lower()
    assert "focused" in result.stdout.lower() or "gold" in result.stdout.lower()


def test_staged_flags_in_help():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--staged" in result.stdout
    assert "--stage-size" in result.stdout
    assert "--stage-repair-steps" in result.stdout
    assert "--stage-guardrail" in result.stdout


def test_selector_flag_in_help():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--selector" in result.stdout
    assert "swiglu_mag" in result.stdout
    assert "structural" in result.stdout


def test_final_repair_steps_in_help():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--final-repair-steps" in result.stdout
    assert "consolidation" in result.stdout.lower() or "global" in result.stdout.lower()


def test_strike_gold_accepts_selector_structural():
    """--strike-gold + --selector structural parses without error (exits at data load)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--strike-gold", "--selector", "structural", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "structural" in result.stdout


def test_phase_dependency_still_works_with_strike_gold_absent():
    """--phases validations still fire when --strike-gold is not used."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--phases", "0,2"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0


# ── Full smoke test (needs model download + inference) ────────────────────────


@pytest.mark.slow
def test_smoke_phase0_produces_summary():
    """Run --smoke --phases 0 and verify outputs are written correctly."""
    outdir = REPO_ROOT / "runs_test_smoke"
    if outdir.exists():
        shutil.rmtree(outdir)

    try:
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--smoke",
                "--phases", "0",
                "--n-ood-texts", "0",
                "--skip-ablations",
                "--outdir", "runs_test_smoke",
            ],
            capture_output=True, text=True, timeout=600,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"Script failed.\nSTDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )

        assert (outdir / "summary.csv").exists(), "summary.csv not created"
        assert (outdir / "summary.json").exists(), "summary.json not created"

        run_dirs = [d for d in outdir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        assert len(run_dirs) >= 1, "No run directories created"

        baseline_dir = run_dirs[0]
        assert (baseline_dir / "config.json").exists()
        assert (baseline_dir / "metrics_post.json").exists()
    finally:
        if outdir.exists():
            shutil.rmtree(outdir)
