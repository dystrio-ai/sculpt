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
import torch

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
        "--staged", "--stage-size", "--stage-repair-steps", "--stage-repair-lr",
        "--stage-guardrail", "--stage-regression-limit",
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


_BASELINE_RESULT = {
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

_FAKE_TEXTS = {
    "cal": ["x"] * 5, "train": ["x"] * 10,
    "eval_w2": ["x"] * 50, "eval_w103": ["x"] * 50,
}


def _spy_factory():
    captured = []

    def spy(**kw):
        captured.append(kw)
        return {
            "run_id": kw["run_id"], "phase": kw["phase"],
            "layer_desc": kw["layer_desc"], "layers": kw["layers"],
            "keep_frac": kw["keep_frac"],
            "ppl_w2_test_post": 10.0, "ppl_w103_valid_post": 10.0,
        }

    return captured, spy


def _run_main_with_args(mle, argv, captured_spy, tmp_path):
    """Run mle.main() with mocked deps and return captured run_compressed calls."""
    from unittest.mock import patch, MagicMock

    saved = {k: getattr(mle, k) for k in _MLE_GLOBALS}
    mock_cfg = MagicMock()
    mock_cfg.num_hidden_layers = 24

    try:
        with (
            patch.object(mle, "run_compressed", side_effect=captured_spy),
            patch.object(mle, "run_baseline", return_value=dict(_BASELINE_RESULT)),
            patch.object(mle, "load_text_sets", return_value=dict(_FAKE_TEXTS)),
            patch.object(mle, "load_ood_texts", return_value=[]),
            patch.object(mle, "write_summary"),
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=mock_cfg,
            ),
            patch("sys.argv", ["mle"] + argv + [
                "--n-ood-texts", "0", "--skip-ablations",
                "--outdir", str(tmp_path / "test_runs"),
            ]),
        ):
            mle.main()
    finally:
        for k, v in saved.items():
            setattr(mle, k, v)


def test_staged_phase1_receives_staged_kwargs(tmp_path):
    """When --staged --phases 0,1 --only-grad-accum 8, run_compressed must
    receive staged=True, stage_repair_steps from CLI, and grad_accum_steps=8."""
    mle = _import_mle()
    captured, spy = _spy_factory()

    _run_main_with_args(mle, [
        "--phases", "0,1",
        "--staged",
        "--stage-repair-steps", "300",
        "--stage-repair-lr", "2e-05",
        "--final-repair-steps", "100",
        "--final-repair-lr", "2e-05",
        "--final-early-stop-patience", "2",
        "--final-curve-every", "50",
        "--only-grad-accum", "8",
    ], spy, tmp_path)

    assert len(captured) == 3, f"Expected 3 Phase 1 layer configs, got {len(captured)}"

    for i, kw in enumerate(captured):
        assert kw["staged"] is True, f"run {i}: staged should be True"
        assert kw["stage_repair_steps"] == 300
        assert abs(kw["stage_repair_lr"] - 2e-05) < 1e-12
        assert kw["final_repair_steps"] == 100
        assert abs(kw["final_repair_lr"] - 2e-05) < 1e-12
        assert kw["final_early_stop_patience"] == 2
        assert kw["final_curve_every"] == 50
        assert kw["grad_accum_steps"] == 8
        assert kw["repair_steps"] == 2000


def test_stage_lr_defaults_to_final_lr(tmp_path):
    """When --stage-repair-lr is NOT provided, it defaults to 0.0 (sentinel),
    and run_compressed receives stage_repair_lr=0.0 so the staged path
    resolves it to final_repair_lr internally."""
    mle = _import_mle()
    captured, spy = _spy_factory()

    _run_main_with_args(mle, [
        "--phases", "0,1",
        "--staged",
        "--final-repair-lr", "2e-05",
    ], spy, tmp_path)

    assert len(captured) == 3
    for kw in captured:
        assert kw["stage_repair_lr"] == 0.0, (
            "sentinel 0.0 should be forwarded; resolution happens inside run_compressed"
        )
        assert abs(kw["final_repair_lr"] - 2e-05) < 1e-12


def test_stage_lr_safety_clamp(tmp_path):
    """When stage_repair_lr >> final_repair_lr, the staged path clamps it
    and emits a warning (not a crash)."""
    import logging
    from unittest.mock import patch, MagicMock

    mle = _import_mle()

    final_lr = 2e-05
    dangerous_stage_lr = 3e-04  # 15x final — well above the 5x clamp threshold

    warnings_captured: list[str] = []
    original_warning = logging.Logger.warning

    def capture_warning(self, msg, *a, **kw):
        warnings_captured.append(msg % a if a else msg)
        return original_warning(self, msg, *a, **kw)

    repair_calls: list[dict] = []

    def spy_repair(**kw):
        repair_calls.append(kw)
        return {"steps": 0.0, "microsteps": 0.0, "curve": [], "early_stopped": False}

    saved = {k: getattr(mle, k) for k in _MLE_GLOBALS}
    mle.RUNS_DIR = tmp_path / "clamp_runs"

    try:
        with (
            patch.object(mle, "repair_layers", side_effect=spy_repair),
            patch.object(mle, "_load_fresh_model", return_value=(
                MagicMock(), MagicMock(),
            )),
            patch.object(mle, "_collect_metrics", return_value={
                "ppl_w2_test": 10.0, "ppl_w103_valid": 10.0,
                "prefill_tokens_per_sec": 1000.0,
                "decode_tokens_per_sec": 100.0,
            }),
            patch.object(mle, "eval_perplexity", return_value=10.0),
            patch.object(mle, "_select_for_layer", return_value=(
                [0, 1], torch.arange(256),
            )),
            patch.object(mle, "compress_mlp_layer_swiglu_inplace", return_value={
                "ffn_kept": 64, "ffn_original": 128,
            }),
            patch.object(mle, "_assert_physical_slicing"),
            patch.object(mle, "_assert_no_masking"),
            patch.object(logging.Logger, "warning", capture_warning),
        ):
            mle.run_compressed(
                run_id=99, phase=1, layer_desc="test",
                layers=[0, 1, 2], keep_frac=0.5, repair_steps=2000,
                texts=dict(_FAKE_TEXTS),
                staged=True, stage_size=6,
                stage_repair_steps=50,
                stage_repair_lr=dangerous_stage_lr,
                final_repair_lr=final_lr,
                final_repair_steps=0,
                grad_accum_steps=1,
                skip_ablations=True,
            )
    finally:
        for k, v in saved.items():
            setattr(mle, k, v)

    clamp_warnings = [
        w for w in warnings_captured
        if "stage_repair_lr" in w and "clamp" in w.lower()
    ]
    assert len(clamp_warnings) >= 1, (
        f"Expected a clamp warning mentioning stage_repair_lr, got: {warnings_captured}"
    )
    assert "3.00e-04" in clamp_warnings[0]

    expected_clamped = final_lr * 5.0
    for rc in repair_calls:
        assert rc["lr"] <= expected_clamped + 1e-12, (
            f"Stage repair lr={rc['lr']:.2e} should be clamped to {expected_clamped:.2e}"
        )


def test_stage_regression_aborts_remaining_stages(tmp_path):
    """If post-repair PPL exceeds pre-repair PPL by >stage_regression_limit,
    remaining stages are aborted and final repair still runs."""
    import logging
    from unittest.mock import patch, MagicMock

    mle = _import_mle()

    warnings_captured: list[str] = []
    original_warning = logging.Logger.warning

    def capture_warning(self, msg, *a, **kw):
        warnings_captured.append(msg % a if a else msg)
        return original_warning(self, msg, *a, **kw)

    repair_calls: list[dict] = []

    def spy_repair(**kw):
        repair_calls.append(kw)
        return {"steps": 10.0, "microsteps": 10.0, "curve": [], "early_stopped": False}

    eval_call_count = [0]

    def mock_eval_ppl(*a, **kw):
        eval_call_count[0] += 1
        n = eval_call_count[0]
        # Stage 1 post-compile: 20.0, post-repair: 30.0 (50% regression > 10% limit)
        if n == 1:
            return 20.0   # post-compile stage 1
        if n == 2:
            return 30.0   # post-repair stage 1 — triggers regression
        return 10.0

    saved = {k: getattr(mle, k) for k in _MLE_GLOBALS}
    mle.RUNS_DIR = tmp_path / "regr_runs"

    try:
        with (
            patch.object(mle, "repair_layers", side_effect=spy_repair),
            patch.object(mle, "_load_fresh_model", return_value=(
                MagicMock(), MagicMock(),
            )),
            patch.object(mle, "_collect_metrics", return_value={
                "ppl_w2_test": 10.0, "ppl_w103_valid": 10.0,
                "prefill_tokens_per_sec": 1000.0,
                "decode_tokens_per_sec": 100.0,
            }),
            patch.object(mle, "eval_perplexity", side_effect=mock_eval_ppl),
            patch.object(mle, "_select_for_layer", return_value=(
                [0, 1], torch.arange(256),
            )),
            patch.object(mle, "compress_mlp_layer_swiglu_inplace", return_value={
                "ffn_kept": 64, "ffn_original": 128,
            }),
            patch.object(mle, "_assert_physical_slicing"),
            patch.object(mle, "_assert_no_masking"),
            patch.object(logging.Logger, "warning", capture_warning),
        ):
            mle.run_compressed(
                run_id=99, phase=1, layer_desc="test",
                layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                keep_frac=0.5, repair_steps=2000,
                texts=dict(_FAKE_TEXTS),
                staged=True, stage_size=6,
                stage_repair_steps=50,
                stage_repair_lr=2e-05,
                stage_guardrail=200.0,
                stage_regression_limit=0.10,
                final_repair_lr=2e-05,
                final_repair_steps=100,
                grad_accum_steps=1,
                skip_ablations=True,
            )
    finally:
        for k, v in saved.items():
            setattr(mle, k, v)

    regression_warnings = [
        w for w in warnings_captured if "STAGE REGRESSION" in w
    ]
    assert len(regression_warnings) == 1, (
        f"Expected 1 regression warning, got {len(regression_warnings)}: {regression_warnings}"
    )
    assert "30.00" in regression_warnings[0]
    assert "20.00" in regression_warnings[0]

    # Stage repair ran for stage 1 only (aborted before stage 2)
    stage_repair_calls = [rc for rc in repair_calls if rc["steps"] == 50]
    assert len(stage_repair_calls) == 1, (
        f"Expected 1 stage repair call (stage 2 aborted), got {len(stage_repair_calls)}"
    )

    # Final global repair still ran (final_repair_steps=100)
    final_repair_calls = [rc for rc in repair_calls if rc["steps"] == 100]
    assert len(final_repair_calls) == 1, (
        f"Expected final global repair to run, got {len(final_repair_calls)}"
    )


def test_phase1_only_layer_desc_filter(tmp_path):
    """--only-layer-desc even filters Phase 1 to a single layer config."""
    mle = _import_mle()
    captured, spy = _spy_factory()

    _run_main_with_args(mle, [
        "--phases", "0,1",
        "--only-layer-desc", "even",
    ], spy, tmp_path)

    assert len(captured) == 1, f"Expected 1 run (even only), got {len(captured)}"
    assert captured[0]["layer_desc"] == "even"
    assert captured[0]["keep_frac"] == 0.50


def test_phase1_only_keep_frac_override(tmp_path):
    """--only-keep-frac 0.70 overrides Phase 1 keep_frac from 0.50 to 0.70."""
    mle = _import_mle()
    captured, spy = _spy_factory()

    _run_main_with_args(mle, [
        "--phases", "0,1",
        "--only-keep-frac", "0.70",
    ], spy, tmp_path)

    assert len(captured) == 3, f"Expected 3 layer configs, got {len(captured)}"
    for kw in captured:
        assert abs(kw["keep_frac"] - 0.70) < 1e-9, (
            f"keep_frac should be 0.70, got {kw['keep_frac']}"
        )


def test_phase1_combined_filters(tmp_path):
    """--only-layer-desc even --only-keep-frac 0.70 produces exactly 1 run."""
    mle = _import_mle()
    captured, spy = _spy_factory()

    _run_main_with_args(mle, [
        "--phases", "0,1",
        "--only-layer-desc", "even",
        "--only-keep-frac", "0.70",
    ], spy, tmp_path)

    assert len(captured) == 1, f"Expected 1 run, got {len(captured)}"
    assert captured[0]["layer_desc"] == "even"
    assert abs(captured[0]["keep_frac"] - 0.70) < 1e-9


def test_phase1_prefix_match(tmp_path):
    """--only-layer-desc 6 matches '6layers' via prefix."""
    mle = _import_mle()
    captured, spy = _spy_factory()

    _run_main_with_args(mle, [
        "--phases", "0,1",
        "--only-layer-desc", "6",
    ], spy, tmp_path)

    assert len(captured) == 1, f"Expected 1 run (6layers via prefix), got {len(captured)}"
    assert captured[0]["layer_desc"] == "6layers"


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
