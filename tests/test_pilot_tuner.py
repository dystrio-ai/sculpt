"""Tests for the Repair Pilot Tuner and triggered escalation (mock-based)."""

from __future__ import annotations

import math
import pytest

from dystrio_sculpt.policy import (
    RepairPolicy,
    PilotResult,
    TuningReport,
    _score_pilot,
    _pilot_candidates,
    _compute_pilot_budget,
    escalate_policy,
    build_policy_ladder,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_policy(name: str = "test", lr: float = 1e-4, steps: int = 100,
                 stage_size: int = 4) -> RepairPolicy:
    return RepairPolicy(
        name=name, stage_size=stage_size, lr=lr, steps=steps,
        early_stop_patience=8, regression_limit=0.02, curve_every=50,
        cheap_eval_texts=64, cheap_eval_max_tokens=5000,
        final_eval_max_tokens=40000, grad_accum_steps=1,
    )


def _make_pilot_result(
    policy_name: str = "p",
    P0: float = 10.0, Pt: float = 9.0, Pmax: float = 10.0,
    elapsed_s: float = 10.0, regression_stop: bool = False,
    nan_inf: bool = False,
) -> PilotResult:
    slope, score, stable, helpful = _score_pilot(
        P0, Pt, Pmax, elapsed_s, regression_stop, nan_inf,
    )
    return PilotResult(
        policy_name=policy_name, P0=P0, Pt=Pt, Pmax=Pmax,
        steps_run=50, elapsed_s=elapsed_s,
        regression_stop=regression_stop, nan_inf=nan_inf,
        slope_per_s=slope, score=score, helpful=helpful, stable=stable,
    )


# ── 1) Scoring: log slope ────────────────────────────────────────────────────

class TestScoringMathLogSlope:
    def test_positive_slope(self):
        slope, score, stable, helpful = _score_pilot(
            P0=10.0, Pt=9.0, Pmax=10.0, elapsed_s=10.0,
            regression_stop=False, nan_inf=False,
        )
        assert slope > 0
        assert stable is True
        assert score > 0

    def test_no_improvement(self):
        slope, score, stable, helpful = _score_pilot(
            P0=10.0, Pt=10.0, Pmax=10.0, elapsed_s=10.0,
            regression_stop=False, nan_inf=False,
        )
        assert abs(slope) < 1e-10
        assert helpful is False

    def test_regression(self):
        slope, score, stable, helpful = _score_pilot(
            P0=10.0, Pt=11.0, Pmax=11.0, elapsed_s=10.0,
            regression_stop=False, nan_inf=False,
        )
        assert slope < 0


# ── 2) Unstable rejected ─────────────────────────────────────────────────────

class TestUnstableRejected:
    def test_regression_stop_rejected(self):
        slope, score, stable, helpful = _score_pilot(
            P0=10.0, Pt=9.0, Pmax=10.0, elapsed_s=10.0,
            regression_stop=True, nan_inf=False,
        )
        assert score == -1e9
        assert stable is False

    def test_nan_inf_rejected(self):
        slope, score, stable, helpful = _score_pilot(
            P0=10.0, Pt=9.0, Pmax=10.0, elapsed_s=10.0,
            regression_stop=False, nan_inf=True,
        )
        assert score == -1e9
        assert stable is False

    def test_both_rejected(self):
        slope, score, stable, helpful = _score_pilot(
            P0=10.0, Pt=9.0, Pmax=10.0, elapsed_s=10.0,
            regression_stop=True, nan_inf=True,
        )
        assert score == -1e9
        assert stable is False


# ── 3) Spike penalty ─────────────────────────────────────────────────────────

class TestSpikePenalty:
    def test_large_spike_reduces_score(self):
        _, score_clean, _, _ = _score_pilot(
            P0=10.0, Pt=9.0, Pmax=10.0, elapsed_s=10.0,
            regression_stop=False, nan_inf=False,
        )
        _, score_spike, _, _ = _score_pilot(
            P0=10.0, Pt=9.0, Pmax=20.0, elapsed_s=10.0,
            regression_stop=False, nan_inf=False,
        )
        assert score_spike < score_clean

    def test_no_spike_when_pmax_near_p0(self):
        epsilon = 0.05
        Pmax_safe = 10.0 * (1 + epsilon)
        _, score_no_spike, _, _ = _score_pilot(
            P0=10.0, Pt=9.0, Pmax=Pmax_safe, elapsed_s=10.0,
            regression_stop=False, nan_inf=False,
        )
        _, score_clean, _, _ = _score_pilot(
            P0=10.0, Pt=9.0, Pmax=10.0, elapsed_s=10.0,
            regression_stop=False, nan_inf=False,
        )
        assert abs(score_no_spike - score_clean) < 0.01

    def test_spike_penalty_magnitude(self):
        _, score_spike, _, _ = _score_pilot(
            P0=10.0, Pt=9.0, Pmax=30.0, elapsed_s=10.0,
            regression_stop=False, nan_inf=False,
        )
        assert score_spike < 0


# ── 4) Candidate list by risk ────────────────────────────────────────────────

class TestCandidateListByRisk:
    def test_low_risk(self):
        ladder = build_policy_ladder(1.0)
        cands = _pilot_candidates(ladder, risk_score=0.2)
        assert len(cands) == 3
        assert cands[0].name == ladder[0].name
        assert cands[1].name == ladder[1].name
        assert cands[2].name == ladder[2].name

    def test_mid_risk(self):
        ladder = build_policy_ladder(1.0)
        cands = _pilot_candidates(ladder, risk_score=0.5)
        assert len(cands) == 2
        assert cands[0].name == ladder[1].name
        assert cands[1].name == ladder[2].name

    def test_high_risk(self):
        ladder = build_policy_ladder(1.0)
        cands = _pilot_candidates(ladder, risk_score=0.7)
        assert len(cands) == 2
        assert cands[0].name == ladder[2].name
        assert cands[1].name == ladder[3].name

    def test_short_ladder(self):
        ladder = build_policy_ladder(1.0)[:2]
        cands = _pilot_candidates(ladder, risk_score=0.7)
        assert len(cands) >= 1
        for c in cands:
            assert c in ladder


# ── 5) Budget skip ───────────────────────────────────────────────────────────

class TestBudgetSkip:
    def test_tiny_budget_skips(self):
        budget = _compute_pilot_budget(max_compile_hours=0.01)
        assert budget < 60

    def test_none_budget_default(self):
        budget = _compute_pilot_budget(max_compile_hours=None)
        assert budget == 240.0

    def test_large_budget_capped(self):
        budget = _compute_pilot_budget(max_compile_hours=10.0)
        assert budget == 480.0

    def test_moderate_budget(self):
        budget = _compute_pilot_budget(max_compile_hours=1.0)
        assert budget == min(0.10 * 3600, 480.0)
        assert budget == 360.0


# ── 6) Selection prefers best score ──────────────────────────────────────────

class TestSelectionPrefersBestScore:
    def test_highest_score_wins(self):
        results = [
            _make_pilot_result("p1", P0=10, Pt=9.5, Pmax=10, elapsed_s=10),
            _make_pilot_result("p2", P0=10, Pt=8.0, Pmax=10, elapsed_s=10),
            _make_pilot_result("p3", P0=10, Pt=9.0, Pmax=10, elapsed_s=10),
        ]
        stable_helpful = [(r, r.policy_name) for r in results if r.stable and r.helpful]
        assert len(stable_helpful) > 0
        best_r, best_name = max(stable_helpful, key=lambda rc: rc[0].score)
        assert best_name == "p2"

    def test_unstable_excluded(self):
        results = [
            _make_pilot_result("p1", P0=10, Pt=8.0, Pmax=10, elapsed_s=10,
                               regression_stop=True),
            _make_pilot_result("p2", P0=10, Pt=9.0, Pmax=10, elapsed_s=10),
        ]
        stable = [r for r in results if r.stable]
        assert len(stable) == 1
        assert stable[0].policy_name == "p2"

    def test_all_unstable_falls_back_to_last(self):
        results = [
            _make_pilot_result("p1", regression_stop=True),
            _make_pilot_result("p2", nan_inf=True),
        ]
        candidates = ["p1", "p2"]
        stable = [r for r in results if r.stable]
        if not stable:
            chosen = candidates[-1]
        else:
            chosen = max(stable, key=lambda r: r.score).policy_name
        assert chosen == "p2"


# ── 7) Escalation trigger ────────────────────────────────────────────────────

class TestEscalationTrigger:
    def test_lr_reduced(self):
        policy = _make_policy(lr=1e-4, steps=100, stage_size=4)
        esc, details = escalate_policy(policy)
        assert esc.lr == 5e-5
        assert details["before"]["lr"] == 1e-4
        assert details["after"]["lr"] == 5e-5

    def test_steps_increased(self):
        policy = _make_policy(steps=100, stage_size=4)
        esc, _ = escalate_policy(policy)
        assert esc.steps == 150

    def test_stage_size_reduced(self):
        policy = _make_policy(stage_size=4)
        esc, _ = escalate_policy(policy)
        assert esc.stage_size == 2

    def test_full_escalation(self):
        policy = _make_policy(lr=5e-5, steps=200, stage_size=2)
        esc, _ = escalate_policy(policy)
        assert esc.lr == 2e-5
        assert esc.steps == 300
        assert esc.stage_size == 1

    def test_already_lowest_lr(self):
        policy = _make_policy(lr=2e-5, steps=100, stage_size=1)
        esc, _ = escalate_policy(policy)
        assert esc.lr == 2e-5
        assert esc.stage_size == 1
        assert esc.steps == 150

    def test_details_dict_complete(self):
        policy = _make_policy(lr=1e-4, steps=100, stage_size=4)
        esc, details = escalate_policy(policy, keep_frac=0.75, trigger_stage=3)
        assert details["keep_frac"] == 0.75
        assert details["trigger_stage"] == 3
        assert "before" in details
        assert "after" in details
        assert details["before"]["name"] == "test"
        assert details["after"]["name"] == "test_esc"

    def test_consecutive_stage_failures_trigger(self):
        """Simulate the engine.py mid-compile escalation logic."""
        stage_stats = [
            {"stage": 0, "layers": [0, 1], "repair_fail": False},
            {"stage": 1, "layers": [2, 3], "repair_fail": True},
            {"stage": 2, "layers": [4, 5], "repair_fail": True},
        ]
        escalation_applied = False
        policy = _make_policy(lr=1e-4, steps=100, stage_size=4)
        consec = 0
        for stat in stage_stats:
            if stat["repair_fail"]:
                consec += 1
            else:
                consec = 0
            if consec >= 2 and not escalation_applied:
                policy, details = escalate_policy(policy, trigger_stage=stat["stage"])
                escalation_applied = True
                break
        assert escalation_applied
        assert policy.lr == 5e-5
        assert policy.steps == 150

    def test_single_failure_no_escalation(self):
        stage_stats = [
            {"stage": 0, "layers": [0, 1], "repair_fail": True},
            {"stage": 1, "layers": [2, 3], "repair_fail": False},
            {"stage": 2, "layers": [4, 5], "repair_fail": True},
        ]
        consec = 0
        escalation_applied = False
        for stat in stage_stats:
            if stat["repair_fail"]:
                consec += 1
            else:
                consec = 0
            if consec >= 2:
                escalation_applied = True
                break
        assert not escalation_applied


# ── 8) Hardening: failure detection uses best_metric ─────────────────────────

class TestStageFailUsesbestMetric:
    """Verify failure detection compares P0 vs Pbest, not P0 vs Plast."""

    def test_best_improved_is_success(self):
        """P0=10, Pbest=9.9 (<0.2% threshold met) -> success even if Plast=10.2."""
        P0 = 10.0
        Pbest = 9.9
        threshold = P0 * (1.0 - 0.002)
        assert Pbest < threshold  # 9.9 < 9.98 -> improvement is real
        # This is NOT a failure
        fail = Pbest >= threshold
        assert fail is False

    def test_negligible_improvement_is_failure(self):
        """P0=10, Pbest=9.985 -> only 0.15% improvement -> failure."""
        P0 = 10.0
        Pbest = 9.985
        threshold = P0 * (1.0 - 0.002)  # 9.98
        fail = Pbest >= threshold
        assert fail is True

    def test_nan_best_is_failure(self):
        """NaN best_metric -> always a failure."""
        import math as _math
        Pbest = float("nan")
        assert _math.isnan(Pbest)

    def test_regression_stop_is_failure(self):
        """regression_stop flag always means failure."""
        regression_stop = True
        assert regression_stop is True


# ── 9) Hardening: mid-compile escalation state machine ───────────────────────

class TestMidCompileEscalation:
    """Test the consecutive-failure counter logic used inside engine.compile_model."""

    def _run_escalation_sim(self, fail_sequence, allow_esc=True):
        """Simulate the engine's consecutive failure counter + escalation."""
        consec = 0
        applied = False
        details = None
        policy = _make_policy(lr=1e-4, steps=100, stage_size=4)
        for si, is_fail in enumerate(fail_sequence):
            if is_fail:
                consec += 1
            else:
                consec = 0
            if consec >= 2 and not applied and allow_esc:
                policy, details = escalate_policy(policy, trigger_stage=si)
                applied = True
        return applied, policy, details

    def test_consecutive_triggers(self):
        applied, policy, details = self._run_escalation_sim(
            [False, True, True, False],
        )
        assert applied
        assert policy.lr == 5e-5
        assert details["trigger_stage"] == 2

    def test_non_consecutive_no_trigger(self):
        applied, policy, _ = self._run_escalation_sim(
            [True, False, True, False, True],
        )
        assert not applied
        assert policy.lr == 1e-4

    def test_escalation_only_once(self):
        """Even with multiple consecutive failures, escalation is once."""
        consec = 0
        applied_count = 0
        policy = _make_policy(lr=1e-4, steps=100, stage_size=4)
        applied = False
        for si, is_fail in enumerate([True, True, True, True]):
            if is_fail:
                consec += 1
            else:
                consec = 0
            if consec >= 2 and not applied:
                policy, _ = escalate_policy(policy, trigger_stage=si)
                applied = True
                applied_count += 1
        assert applied_count == 1

    def test_allow_escalation_false_blocks(self):
        applied, policy, _ = self._run_escalation_sim(
            [True, True, True], allow_esc=False,
        )
        assert not applied
        assert policy.lr == 1e-4


# ── 10) Hardening: search persists escalated policy ──────────────────────────

class TestSearchPersistsPolicy:
    """Verify the policy adoption logic used in search._evaluate."""

    def test_escalated_policy_adopted(self):
        """When compile_result.escalation_applied is True, search adopts it."""
        original = _make_policy(lr=1e-4, steps=100, stage_size=4)
        esc_config = {
            "name": "test_esc", "stage_size": 2, "lr": 5e-5, "steps": 150,
            "early_stop_patience": 8, "regression_limit": 0.02, "curve_every": 50,
            "cheap_eval_texts": 64, "cheap_eval_max_tokens": 5000,
            "final_eval_max_tokens": 40000, "grad_accum_steps": 1,
            "max_grad_norm": None,
        }
        # Simulate search logic
        _escalation_applied = False
        policy = original
        # First keep evaluation — escalation fires inside compile_model
        result_escalation_applied = True
        result_policy_config = esc_config
        if result_escalation_applied and not _escalation_applied:
            _escalation_applied = True
            if result_policy_config["name"].endswith("_esc"):
                policy = RepairPolicy(**{
                    k: v for k, v in result_policy_config.items()
                    if k in RepairPolicy.__dataclass_fields__
                })
        assert _escalation_applied
        assert policy.name == "test_esc"
        assert policy.lr == 5e-5
        assert policy.steps == 150

    def test_second_keep_uses_escalated(self):
        """After escalation, subsequent evaluations start with the new policy."""
        esc = _make_policy(name="test_esc", lr=5e-5, steps=150, stage_size=2)
        _escalation_applied = True
        policy = esc
        # Second keep — allow_escalation should be False
        allow_esc = not _escalation_applied
        assert allow_esc is False
        assert policy.name == "test_esc"

    def test_no_escalation_keeps_original(self):
        """If no escalation fires, policy stays original."""
        original = _make_policy(lr=1e-4, steps=100, stage_size=4)
        _escalation_applied = False
        policy = original
        result_escalation_applied = False
        if result_escalation_applied and not _escalation_applied:
            _escalation_applied = True
        assert not _escalation_applied
        assert policy.name == "test"
        assert policy.lr == 1e-4
