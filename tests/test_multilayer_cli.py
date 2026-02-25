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
        "--final-repair-steps", "--final-repair-lr",
        "--final-early-stop-patience", "--final-curve-every",
        "--only-layer-desc", "--only-keep-frac", "--only-grad-accum",
        "--compress-layers", "--skip-last-layers", "--keep-schedule",
        "--model-id",
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


def test_model_id_accepted_with_help():
    """--model-id dummy is accepted by argparse (no download when --help)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--model-id", "dummy/model", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--model-id" in result.stdout


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


# ── Unit-style monkeypatch tests ──────────────────────────────────────────────


def _import_mle():
    """Import multilayer_experiment module (lazy, cached)."""
    import importlib
    _sys = sys
    for p in [str(REPO_ROOT / "experiments"), str(REPO_ROOT / "src")]:
        if p not in _sys.path:
            _sys.path.insert(0, p)
    if "multilayer_experiment" in _sys.modules:
        return _sys.modules["multilayer_experiment"]
    import multilayer_experiment
    return multilayer_experiment


_MLE_GLOBALS = [
    "MODEL_ID", "NUM_LAYERS", "RUNS_DIR",
    "N_TEXTS_CAL", "N_TEXTS_TRAIN", "N_TEXTS_EVAL", "MAX_EVAL_TOKENS",
    "PREFILL_WARMUP_ITERS", "PREFILL_ITERS", "DECODE_WARMUP_ITERS", "DECODE_ITERS",
]


def test_staged_phase1_receives_staged_kwargs(tmp_path):
    """When --staged --phases 0,1 --only-grad-accum 8, run_compressed must
    receive staged=True, stage_repair_steps from CLI, and grad_accum_steps=8."""
    from unittest.mock import patch, MagicMock

    mle = _import_mle()
    saved = {k: getattr(mle, k) for k in _MLE_GLOBALS}

    captured = []

    def spy_run_compressed(**kw):
        captured.append(kw)
        return {
            "run_id": kw["run_id"],
            "phase": kw["phase"],
            "layer_desc": kw["layer_desc"],
            "layers": kw["layers"],
            "keep_frac": kw["keep_frac"],
            "ppl_w2_test_post": 10.0,
            "ppl_w103_valid_post": 10.0,
        }

    baseline_result = {
        "run_id": 0, "phase": 0, "layer_desc": "baseline",
        "layers": [], "keep_frac": 1.0, "repair_steps": 0,
        "actual_repair_steps": 0, "selector_name": "swiglu_mag",
        "grad_accum_steps": 1, "early_stopped": False,
        "guardrail_failed": False, "staged": False, "stages_completed": 0,
        "ppl_w2_test_pre": 10.0, "ppl_w103_valid_pre": 10.0,
        "ppl_w2_test_post": 10.0, "ppl_w103_valid_post": 10.0,
        "ppl_ood_pre": None, "ppl_ood_post": None,
        "prefill_tokens_per_sec_pre": 1000.0,
        "prefill_tokens_per_sec_post": 1000.0,
        "decode_tokens_per_sec_pre": 100.0,
        "decode_tokens_per_sec_post": 100.0,
        "repair_wall_time_seconds": 0.0, "compile_wall_time_seconds": 0.0,
        "time_to_recover_90pct": None,
        "ppl_w103_post_no_repair": 10.0, "ppl_w103_post_random": None,
        "ppl_ood_post_no_repair": None, "ppl_ood_post_random": None,
        "prefill_tps_no_repair": 1000.0, "prefill_tps_random": None,
        "decode_tps_no_repair": 100.0, "decode_tps_random": None,
        "vllm_metrics": None,
    }

    mock_cfg = MagicMock()
    mock_cfg.num_hidden_layers = 24

    try:
        with (
            patch.object(mle, "run_compressed", side_effect=spy_run_compressed),
            patch.object(mle, "run_baseline", return_value=baseline_result),
            patch.object(mle, "load_text_sets", return_value={
                "cal": ["x"] * 5, "train": ["x"] * 10,
                "eval_w2": ["x"] * 50, "eval_w103": ["x"] * 50,
            }),
            patch.object(mle, "load_ood_texts", return_value=[]),
            patch.object(mle, "write_summary"),
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=mock_cfg,
            ),
            patch("sys.argv", [
                "mle",
                "--phases", "0,1",
                "--staged",
                "--stage-repair-steps", "300",
                "--final-repair-steps", "100",
                "--final-repair-lr", "0.0001",
                "--final-early-stop-patience", "2",
                "--final-curve-every", "50",
                "--only-grad-accum", "8",
                "--n-ood-texts", "0",
                "--skip-ablations",
                "--outdir", str(tmp_path / "test_runs"),
            ]),
        ):
            mle.main()
    finally:
        for k, v in saved.items():
            setattr(mle, k, v)

    assert len(captured) == 3, f"Expected 3 Phase 1 layer configs, got {len(captured)}"

    for i, kw in enumerate(captured):
        assert kw["staged"] is True, f"run {i}: staged should be True"
        assert kw["stage_repair_steps"] == 300, (
            f"run {i}: stage_repair_steps={kw['stage_repair_steps']}, expected 300"
        )
        assert kw["final_repair_steps"] == 100, (
            f"run {i}: final_repair_steps={kw['final_repair_steps']}, expected 100"
        )
        assert abs(kw["final_repair_lr"] - 0.0001) < 1e-12, (
            f"run {i}: final_repair_lr={kw['final_repair_lr']}, expected 0.0001"
        )
        assert kw["final_early_stop_patience"] == 2, (
            f"run {i}: final_early_stop_patience={kw['final_early_stop_patience']}"
        )
        assert kw["final_curve_every"] == 50, (
            f"run {i}: final_curve_every={kw['final_curve_every']}"
        )
        assert kw["grad_accum_steps"] == 8, (
            f"run {i}: grad_accum_steps={kw['grad_accum_steps']}, expected 8"
        )
        assert kw["repair_steps"] == 2000, (
            f"run {i}: Phase 1 repair_steps should be 2000 (staged path handles internally)"
        )


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
