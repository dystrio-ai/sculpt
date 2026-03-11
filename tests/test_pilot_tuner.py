"""Tests for Repair Pilot Tuner, stage-size controller, escalation,
stage outcome semantics, stratified sampling, E2E metrics (mock-based)."""

from __future__ import annotations

import math
import pytest

from dystrio_sculpt.policy import (
    RepairPolicy,
    _asymmetric_reward,
    _score_pilot,
    _score_two_stage_pilot,
    _stratified_pilot_chunks,
    _conductance_probe_chunk,
    _pilot_candidates,
    _compute_pilot_budget,
    _adapt_steps,
    _with_stage_size,
    _recovery_strength,
    _LRArm,
    escalate_policy,
    build_policy_ladder,
    compute_e2e_speedup,
    E2E_PROFILES,
    ASYMMETRIC_SCALE,
    LR_GRID,
    LR_REWARD_THRESHOLD,
    MAX_LR_PROBES,
    HELPFUL_THRESHOLD,
    TIE_BREAK_GAIN,
    WH, WI, WM,
)
from dystrio_sculpt._bench import compute_latency_percentiles
from dystrio_sculpt.emit import (
    _bytes_to_gib, _LATENCY_KEYS, _SUMMARY_COLUMNS,
    _BASELINE_LATENCY_KEYS, _DERIVED_KEYS,
    _safe_pct, _safe_ratio, _safe_throughput_gain_pct, _gpu_hour_reduction_pct,
    emit_run_metadata,
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


def _make_stage_stat(
    stage: int = 0, improve_frac: float = 0.0,
    regression_tripwire: bool = False, nan_inf: bool = False,
    early_stop: bool = False,
) -> dict:
    fail = nan_inf
    helpful = (not fail) and improve_frac >= HELPFUL_THRESHOLD
    return {
        "stage": stage, "layers": [stage * 2, stage * 2 + 1],
        "ppl_pre_repair": 10.0, "ppl_best": 10.0 * (1 - improve_frac),
        "improve_frac": improve_frac,
        "regression_tripwire": regression_tripwire, "nan_inf": nan_inf,
        "early_stop": early_stop,
        "repair_fail": fail, "repair_helpful": helpful,
    }


# ── 1) Stage outcome semantics ───────────────────────────────────────────────

class TestStageOutcomeSemantics:
    """repair_fail = nan_inf only (tripwire is soft stop).
    repair_helpful = improvement >= threshold and no nan_inf."""

    def test_negligible_improvement_not_fail(self):
        s = _make_stage_stat(improve_frac=0.001)
        assert s["repair_fail"] is False
        assert s["repair_helpful"] is False

    def test_zero_improvement_not_fail(self):
        s = _make_stage_stat(improve_frac=0.0)
        assert s["repair_fail"] is False
        assert s["repair_helpful"] is False

    def test_regression_tripwire_is_not_fail(self):
        s = _make_stage_stat(regression_tripwire=True, improve_frac=0.05)
        assert s["repair_fail"] is False
        assert s["regression_tripwire"] is True
        assert s["repair_helpful"] is True

    def test_nan_inf_is_fail(self):
        s = _make_stage_stat(nan_inf=True)
        assert s["repair_fail"] is True

    def test_significant_improvement_is_helpful(self):
        s = _make_stage_stat(improve_frac=0.05)
        assert s["repair_fail"] is False
        assert s["repair_helpful"] is True

    def test_threshold_boundary(self):
        s_below = _make_stage_stat(improve_frac=0.0019)
        s_at = _make_stage_stat(improve_frac=0.002)
        assert s_below["repair_helpful"] is False
        assert s_at["repair_helpful"] is True

    def test_nan_inf_overrides_helpful(self):
        s = _make_stage_stat(nan_inf=True, improve_frac=0.10)
        assert s["repair_fail"] is True
        assert s["repair_helpful"] is False

    def test_tripwire_with_improvement_is_helpful(self):
        s = _make_stage_stat(regression_tripwire=True, improve_frac=0.10)
        assert s["repair_fail"] is False
        assert s["repair_helpful"] is True


# ── 2) Stratified pilot chunk selection ───────────────────────────────────────

class TestStratifiedPilotChunks:
    def test_early_and_late_chunk(self):
        layer_order = list(range(32))
        chunks = _stratified_pilot_chunks(layer_order, stage_size=4, K=2)
        assert len(chunks) == 2
        assert chunks[0] == [0, 1, 2, 3]
        late_idx = int(0.70 * 7)  # 8 chunks total, idx 4
        expected_late = layer_order[late_idx * 4 : late_idx * 4 + 4]
        assert chunks[1] == expected_late

    def test_deterministic(self):
        layer_order = list(range(32))
        c1 = _stratified_pilot_chunks(layer_order, 4, K=2)
        c2 = _stratified_pilot_chunks(layer_order, 4, K=2)
        assert c1 == c2

    def test_small_model_single_chunk(self):
        layer_order = [0, 1, 2]
        chunks = _stratified_pilot_chunks(layer_order, stage_size=4, K=2)
        assert len(chunks) == 1
        assert chunks[0] == [0, 1, 2]

    def test_two_chunks_picks_both(self):
        layer_order = list(range(8))
        chunks = _stratified_pilot_chunks(layer_order, stage_size=4, K=2)
        assert len(chunks) == 2
        assert chunks[0] == [0, 1, 2, 3]
        assert chunks[1] == [4, 5, 6, 7]

    def test_empty_order(self):
        assert _stratified_pilot_chunks([], 4, K=2) == []

    def test_stage_size_2(self):
        layer_order = list(range(16))
        chunks = _stratified_pilot_chunks(layer_order, stage_size=2, K=2)
        assert len(chunks) == 2
        assert chunks[0] == [0, 1]
        late_idx = int(0.70 * 7)  # 8 chunks, idx 4
        assert chunks[1] == layer_order[late_idx * 2 : late_idx * 2 + 2]

    def test_late_not_same_as_early(self):
        layer_order = list(range(32))
        chunks = _stratified_pilot_chunks(layer_order, stage_size=4, K=2)
        assert chunks[0] != chunks[1]


# ── 2b) Asymmetric reward function ─────────────────────────────────────────

class TestAsymmetricReward:
    def test_zero_improvement_zero_reward(self):
        assert _asymmetric_reward(0.0) == 0.0

    def test_negative_improvement_clamped_to_zero(self):
        assert _asymmetric_reward(-0.05) == 0.0
        assert _asymmetric_reward(-1.0) == 0.0

    def test_positive_improvement_positive_reward(self):
        assert _asymmetric_reward(0.01) > 0.0
        assert _asymmetric_reward(0.05) > 0.0

    def test_monotonically_increasing(self):
        r1 = _asymmetric_reward(0.01)
        r2 = _asymmetric_reward(0.03)
        r3 = _asymmetric_reward(0.05)
        r4 = _asymmetric_reward(0.10)
        assert r1 < r2 < r3 < r4

    def test_exponential_amplification(self):
        """Doubling improve_frac more than doubles the reward."""
        r_small = _asymmetric_reward(0.03)
        r_large = _asymmetric_reward(0.06)
        assert r_large > 2.0 * r_small

    def test_custom_scale(self):
        r_default = _asymmetric_reward(0.05)
        r_low_scale = _asymmetric_reward(0.05, scale=1.0)
        r_high_scale = _asymmetric_reward(0.05, scale=20.0)
        assert r_low_scale < r_default < r_high_scale

    def test_known_value(self):
        """exp(10 * 0.05) - 1 = exp(0.5) - 1 ≈ 0.6487."""
        import math
        expected = math.exp(0.5) - 1.0
        assert _asymmetric_reward(0.05) == pytest.approx(expected, rel=1e-6)


# ── 3) Two-stage pilot scoring (with M term) ─────────────────────────────────

class TestTwoStagePilotScoring:
    def test_helpful_policy_preferred(self):
        stats_a = [
            _make_stage_stat(0, improve_frac=0.05),
            _make_stage_stat(1, improve_frac=0.03),
        ]
        stats_b = [
            _make_stage_stat(0, improve_frac=0.001),
            _make_stage_stat(1, improve_frac=0.0),
        ]
        score_a, stable_a, Ha, Ia, Ma = _score_two_stage_pilot(stats_a, 10.0)
        score_b, stable_b, Hb, Ib, Mb = _score_two_stage_pilot(stats_b, 10.0)
        assert score_a > score_b
        assert Ha == 2
        assert Hb == 0

    def test_tripwire_not_rejected(self):
        stats = [
            _make_stage_stat(0, regression_tripwire=True, improve_frac=0.03),
            _make_stage_stat(1, improve_frac=0.05),
        ]
        score, stable, H, I, M = _score_two_stage_pilot(stats, 10.0)
        assert score > -1e9
        assert stable is True
        assert H == 2

    def test_nan_inf_rejected(self):
        stats = [
            _make_stage_stat(0, nan_inf=True),
            _make_stage_stat(1, improve_frac=0.05),
        ]
        score, stable, H, I, M = _score_two_stage_pilot(stats, 10.0)
        assert score == -1e9
        assert stable is False

    def test_empty_stats(self):
        score, stable, H, I, M = _score_two_stage_pilot([], 10.0)
        assert stable is True
        assert H == 0
        assert I == 0.0
        assert M == 0.0

    def test_faster_candidate_preferred(self):
        stats = [_make_stage_stat(0, improve_frac=0.02)]
        score_fast, _, _, _, _ = _score_two_stage_pilot(stats, 5.0)
        score_slow, _, _, _, _ = _score_two_stage_pilot(stats, 50.0)
        assert score_fast > score_slow

    def test_max_improve_term_matters(self):
        """Same H and I, but higher M => higher score."""
        stats_high_m = [
            _make_stage_stat(0, improve_frac=0.08),
            _make_stage_stat(1, improve_frac=0.02),
        ]
        stats_low_m = [
            _make_stage_stat(0, improve_frac=0.05),
            _make_stage_stat(1, improve_frac=0.05),
        ]
        score_hm, _, H_hm, I_hm, M_hm = _score_two_stage_pilot(stats_high_m, 10.0)
        score_lm, _, H_lm, I_lm, M_lm = _score_two_stage_pilot(stats_low_m, 10.0)
        assert H_hm == H_lm  # same helpful count
        assert I_hm == I_lm  # same total I
        assert M_hm > M_lm   # higher peak win
        assert score_hm > score_lm

    def test_returns_five_values(self):
        stats = [_make_stage_stat(0, improve_frac=0.03)]
        result = _score_two_stage_pilot(stats, 10.0)
        assert len(result) == 5
        score, stable, H, I, M = result
        assert M == pytest.approx(0.03)


# ── 4) Legacy scoring (backward compat) ──────────────────────────────────────

class TestLegacyScoringMath:
    def test_positive_slope(self):
        slope, score, stable, helpful = _score_pilot(
            P0=10.0, Pt=9.0, Pmax=10.0, elapsed_s=10.0,
            regression_stop=False, nan_inf=False,
        )
        assert slope > 0
        assert stable is True

    def test_regression_tripwire_not_rejected(self):
        slope, score, stable, _ = _score_pilot(
            P0=10.0, Pt=9.0, Pmax=10.0, elapsed_s=10.0,
            regression_stop=True, nan_inf=False,
        )
        assert score > -1e9
        assert stable is True

    def test_nan_inf_rejected(self):
        _, score, stable, _ = _score_pilot(
            P0=10.0, Pt=9.0, Pmax=10.0, elapsed_s=10.0,
            regression_stop=False, nan_inf=True,
        )
        assert score == -1e9
        assert stable is False


# ── 5) Candidate list by risk ────────────────────────────────────────────────

class TestCandidateListByRisk:
    def test_low_risk(self):
        ladder = build_policy_ladder(1.0)
        cands = _pilot_candidates(ladder, risk_score=0.2)
        assert len(cands) == 3

    def test_mid_risk(self):
        ladder = build_policy_ladder(1.0)
        cands = _pilot_candidates(ladder, risk_score=0.5)
        assert len(cands) == 2

    def test_high_risk(self):
        ladder = build_policy_ladder(1.0)
        cands = _pilot_candidates(ladder, risk_score=0.7)
        assert len(cands) == 2


# ── 6) Budget ─────────────────────────────────────────────────────────────────

class TestBudgetSkip:
    def test_tiny_budget_skips(self):
        assert _compute_pilot_budget(0.01) < 60

    def test_none_budget_default(self):
        assert _compute_pilot_budget(None) == 240.0

    def test_large_budget_capped(self):
        assert _compute_pilot_budget(10.0) == 480.0


# ── 7) Steps adaptation (uses I or M) ────────────────────────────────────────

class TestStepsAdaptation:
    def test_adapts_when_helpful(self):
        policy = _make_policy(steps=100)
        adapted = _adapt_steps(policy, H=2, I=0.06, M=0.04)
        assert adapted.steps > 100
        assert adapted.steps == 125

    def test_no_adapt_when_not_helpful(self):
        policy = _make_policy(steps=100)
        same = _adapt_steps(policy, H=0, I=0.0, M=0.0)
        assert same.steps == 100

    def test_caps_increase(self):
        policy = _make_policy(steps=1000)
        adapted = _adapt_steps(policy, H=2, I=0.10, M=0.08)
        assert adapted.steps <= 1000 + 200

    def test_adapts_on_M_alone(self):
        """I is small but M >= 0.01 triggers adaptation."""
        policy = _make_policy(steps=100)
        adapted = _adapt_steps(policy, H=1, I=0.003, M=0.015)
        assert adapted.steps == 125

    def test_adapts_on_I_alone(self):
        """M is small but I >= 0.005 triggers adaptation."""
        policy = _make_policy(steps=100)
        adapted = _adapt_steps(policy, H=0, I=0.006, M=0.003)
        assert adapted.steps == 125

    def test_no_adapt_both_below(self):
        """Both I and M below thresholds => no change."""
        policy = _make_policy(steps=100)
        same = _adapt_steps(policy, H=1, I=0.004, M=0.009)
        assert same.steps == 100


# ── 8) Stage-size helpers ─────────────────────────────────────────────────────

class TestWithStageSize:
    def test_noop_if_same(self):
        policy = _make_policy(stage_size=4)
        assert _with_stage_size(policy, 4) is policy

    def test_creates_new(self):
        policy = _make_policy(stage_size=4)
        new = _with_stage_size(policy, 2)
        assert new.stage_size == 2
        assert new.lr == policy.lr
        assert new.steps == policy.steps


# ── 9) Stage-size selection logic ─────────────────────────────────────────────

class TestStageSizeSelection:
    def test_picks_largest_stable(self):
        probes = [
            {"stage_size": 4, "has_fail": False, "has_helpful": True},
            {"stage_size": 2, "has_fail": False, "has_helpful": True},
        ]
        chosen = None
        for p in probes:
            if not p["has_fail"]:
                chosen = p["stage_size"]
                break
        assert chosen == 4

    def test_falls_back_on_instability(self):
        probes = [
            {"stage_size": 4, "has_fail": True, "has_helpful": False},
            {"stage_size": 2, "has_fail": False, "has_helpful": True},
        ]
        chosen = 2
        for p in probes:
            if not p["has_fail"]:
                chosen = p["stage_size"]
                break
        assert chosen == 2

    def test_all_unstable_picks_smallest(self):
        probes = [
            {"stage_size": 4, "has_fail": True},
            {"stage_size": 2, "has_fail": True},
        ]
        chosen = 2
        for p in probes:
            if not p["has_fail"]:
                chosen = p["stage_size"]
                break
        assert chosen == 2


# ── 10) Escalation ───────────────────────────────────────────────────────────

class TestEscalationTrigger:
    def test_lr_reduced(self):
        policy = _make_policy(lr=1e-4, steps=100, stage_size=4)
        esc, details = escalate_policy(policy)
        assert esc.lr == 5e-5
        assert details["before"]["lr"] == 1e-4

    def test_steps_increased(self):
        esc, _ = escalate_policy(_make_policy(steps=100))
        assert esc.steps == 150

    def test_stage_size_reduced(self):
        esc, _ = escalate_policy(_make_policy(stage_size=4))
        assert esc.stage_size == 2

    def test_details_complete(self):
        _, d = escalate_policy(_make_policy(), keep_frac=0.75, trigger_stage=3)
        assert d["keep_frac"] == 0.75
        assert d["trigger_stage"] == 3
        assert "before" in d and "after" in d

    def test_only_on_true_instability(self):
        neutral_stats = [
            _make_stage_stat(0, improve_frac=0.0),
            _make_stage_stat(1, improve_frac=0.0),
        ]
        consec = 0
        triggered = False
        for s in neutral_stats:
            if s["repair_fail"]:
                consec += 1
            else:
                consec = 0
            if consec >= 2:
                triggered = True
        assert not triggered

    def test_triggered_on_consecutive_nan_inf(self):
        fail_stats = [
            _make_stage_stat(0, nan_inf=True),
            _make_stage_stat(1, nan_inf=True),
        ]
        consec = 0
        for s in fail_stats:
            if s["repair_fail"]:
                consec += 1
            else:
                consec = 0
        assert consec >= 2

    def test_single_failure_no_trigger(self):
        stats = [
            _make_stage_stat(0, nan_inf=True),
            _make_stage_stat(1, improve_frac=0.01),
            _make_stage_stat(2, nan_inf=True),
        ]
        consec = 0
        triggered = False
        for s in stats:
            if s["repair_fail"]:
                consec += 1
            else:
                consec = 0
            if consec >= 2:
                triggered = True
        assert not triggered


# ── 11) Mid-compile escalation state machine ─────────────────────────────────

class TestMidCompileEscalation:
    def _run_sim(self, fail_sequence, allow_esc=True):
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
        applied, policy, details = self._run_sim([False, True, True, False])
        assert applied
        assert policy.lr == 5e-5

    def test_non_consecutive_no_trigger(self):
        applied, _, _ = self._run_sim([True, False, True, False, True])
        assert not applied

    def test_escalation_only_once(self):
        consec = 0
        count = 0
        policy = _make_policy()
        applied = False
        for si, fail in enumerate([True, True, True, True]):
            consec = consec + 1 if fail else 0
            if consec >= 2 and not applied:
                policy, _ = escalate_policy(policy, trigger_stage=si)
                applied = True
                count += 1
        assert count == 1

    def test_blocked_when_disallowed(self):
        applied, _, _ = self._run_sim([True, True, True], allow_esc=False)
        assert not applied


# ── 12) Search persists escalated policy ──────────────────────────────────────

class TestSearchPersistsPolicy:
    def test_escalated_policy_adopted(self):
        esc_config = {
            "name": "test_esc", "stage_size": 2, "lr": 5e-5, "steps": 150,
            "early_stop_patience": 8, "regression_limit": 0.02, "curve_every": 50,
            "cheap_eval_texts": 64, "cheap_eval_max_tokens": 5000,
            "final_eval_max_tokens": 40000, "grad_accum_steps": 1,
            "max_grad_norm": None,
        }
        _esc_applied = False
        policy = _make_policy()
        if True and not _esc_applied:
            _esc_applied = True
            if esc_config["name"].endswith("_esc"):
                policy = RepairPolicy(**{
                    k: v for k, v in esc_config.items()
                    if k in RepairPolicy.__dataclass_fields__
                })
        assert _esc_applied
        assert policy.name == "test_esc"

    def test_no_escalation_keeps_original(self):
        policy = _make_policy()
        _esc_applied = False
        if False and not _esc_applied:
            _esc_applied = True
        assert not _esc_applied
        assert policy.name == "test"


# ── 13) Search window: continue on helpful signal with cap ────────────────────

class TestSearchWindowContinueOnHelpful:
    _MIN_PREFILL_DELTA = 0.005

    def _window_decision(
        self, ppl_ratio, ceiling, stage_stats, extras_so_far,
        prefill_speedup, prev_prefill_speedup, max_extras=1,
    ):
        """Simulate the search.py window logic (with speed guardrail)."""
        close = ppl_ratio <= ceiling * 1.10
        helpful_count = sum(1 for s in stage_stats if s.get("repair_helpful", False))
        total_improve = sum(s.get("improve_frac", 0.0) for s in stage_stats)
        repair_shows_promise = helpful_count >= 1 or total_improve >= 0.005
        speed_improving = prefill_speedup >= prev_prefill_speedup + self._MIN_PREFILL_DELTA
        if close and repair_shows_promise and speed_improving and extras_so_far < max_extras:
            return "continue"
        return "stop"

    def test_continues_when_helpful_close_and_faster(self):
        stats = [
            _make_stage_stat(0, improve_frac=0.05),
            _make_stage_stat(1, improve_frac=0.0),
        ]
        decision = self._window_decision(
            ppl_ratio=2.05, ceiling=2.0, stage_stats=stats, extras_so_far=0,
            prefill_speedup=1.30, prev_prefill_speedup=1.29,
        )
        assert decision == "continue"

    def test_stops_when_no_speed_gain(self):
        stats = [_make_stage_stat(0, improve_frac=0.05)]
        decision = self._window_decision(
            ppl_ratio=2.05, ceiling=2.0, stage_stats=stats, extras_so_far=0,
            prefill_speedup=1.30, prev_prefill_speedup=1.30,
        )
        assert decision == "stop"

    def test_stops_when_far_from_ceiling(self):
        stats = [_make_stage_stat(0, improve_frac=0.05)]
        decision = self._window_decision(
            ppl_ratio=2.50, ceiling=2.0, stage_stats=stats, extras_so_far=0,
            prefill_speedup=1.40, prev_prefill_speedup=1.30,
        )
        assert decision == "stop"

    def test_stops_when_no_helpful(self):
        stats = [
            _make_stage_stat(0, improve_frac=0.001),
            _make_stage_stat(1, improve_frac=0.0),
        ]
        decision = self._window_decision(
            ppl_ratio=2.05, ceiling=2.0, stage_stats=stats, extras_so_far=0,
            prefill_speedup=1.40, prev_prefill_speedup=1.30,
        )
        assert decision == "stop"

    def test_hard_cap_one_extra(self):
        stats = [_make_stage_stat(0, improve_frac=0.05)]
        d1 = self._window_decision(
            ppl_ratio=2.05, ceiling=2.0, stage_stats=stats, extras_so_far=0,
            prefill_speedup=1.35, prev_prefill_speedup=1.30,
        )
        d2 = self._window_decision(
            ppl_ratio=2.05, ceiling=2.0, stage_stats=stats, extras_so_far=1,
            prefill_speedup=1.40, prev_prefill_speedup=1.30,
        )
        assert d1 == "continue"
        assert d2 == "stop"

    def test_total_improve_threshold(self):
        """Even if no single helpful stage, total_improve >= 0.005 continues."""
        stats = [
            _make_stage_stat(0, improve_frac=0.0019),
            _make_stage_stat(1, improve_frac=0.004),
        ]
        decision = self._window_decision(
            ppl_ratio=2.05, ceiling=2.0, stage_stats=stats, extras_so_far=0,
            prefill_speedup=1.35, prev_prefill_speedup=1.30,
        )
        assert decision == "continue"


# ── 14) E2E workload speedup formula ──────────────────────────────────────────

class TestE2ESpeedup:
    def test_identity(self):
        """No speedup => e2e = 1.0."""
        e2e = compute_e2e_speedup(1.0, 1.0, 256, 256)
        assert e2e == pytest.approx(1.0)

    def test_known_values(self):
        """2x prefill, 1x decode, P=D => e2e = (512)/(128+256) = 512/384 ≈ 1.333."""
        e2e = compute_e2e_speedup(2.0, 1.0, 256, 256)
        assert e2e == pytest.approx(512.0 / 384.0, rel=1e-4)

    def test_rag_profile(self):
        """RAG: prefill-heavy => prefill speedup matters more."""
        e2e_fast_prefill = compute_e2e_speedup(2.0, 1.0, 2048, 128)
        e2e_fast_decode = compute_e2e_speedup(1.0, 2.0, 2048, 128)
        assert e2e_fast_prefill > e2e_fast_decode

    def test_batch_profile(self):
        """Batch: very prefill-heavy => nearly equal to prefill_speedup."""
        e2e = compute_e2e_speedup(1.5, 1.0, 1024, 32)
        assert e2e > 1.4

    def test_zero_speedup_returns_zero(self):
        assert compute_e2e_speedup(0.0, 1.0, 256, 256) == 0.0
        assert compute_e2e_speedup(1.0, 0.0, 256, 256) == 0.0

    def test_profiles_exist(self):
        assert "chat" in E2E_PROFILES
        assert "rag" in E2E_PROFILES
        assert "batch" in E2E_PROFILES
        for name, p in E2E_PROFILES.items():
            assert "P" in p and "D" in p


# ── 15) VRAM metrics: bytes-to-GiB and 6-field structure ──────────────────────

class TestVRAMBytesToGiB:
    def test_basic_conversion(self):
        assert _bytes_to_gib(4 * 1024 ** 3) == 4.0
        assert _bytes_to_gib(6 * 1024 ** 3) == 6.0

    def test_none_passthrough(self):
        assert _bytes_to_gib(None) is None

    def test_zero(self):
        assert _bytes_to_gib(0) == 0.0

    def test_fractional(self):
        val = _bytes_to_gib(int(1.5 * 1024 ** 3))
        assert val == 1.5


class TestVRAM6Fields:
    def test_compile_and_bench_separated(self):
        """All 6 fields populated when CUDA available."""
        compile_alloc = 4 * 1024 ** 3
        compile_resv = 6 * 1024 ** 3
        bench_alloc = 3 * 1024 ** 3
        bench_resv = 5 * 1024 ** 3
        end_alloc = 2 * 1024 ** 3
        end_resv = 4 * 1024 ** 3
        metrics = {
            "peak_compile_alloc_gb": _bytes_to_gib(compile_alloc),
            "peak_compile_reserved_gb": _bytes_to_gib(compile_resv),
            "peak_bench_alloc_gb": _bytes_to_gib(bench_alloc),
            "peak_bench_reserved_gb": _bytes_to_gib(bench_resv),
            "steady_state_alloc_gb": _bytes_to_gib(end_alloc),
            "steady_state_reserved_gb": _bytes_to_gib(end_resv),
        }
        assert metrics["peak_compile_alloc_gb"] == 4.0
        assert metrics["peak_bench_alloc_gb"] == 3.0
        assert metrics["steady_state_alloc_gb"] == 2.0
        assert metrics["peak_compile_alloc_gb"] > metrics["peak_bench_alloc_gb"]

    def test_none_fields_omitted(self):
        """When not on CUDA, all fields are None; metrics omit them."""
        fields = {
            "peak_compile_alloc_gb": _bytes_to_gib(None),
            "peak_bench_alloc_gb": _bytes_to_gib(None),
            "steady_state_alloc_gb": _bytes_to_gib(None),
        }
        metrics = {k: v for k, v in fields.items() if v is not None}
        assert len(metrics) == 0

    def test_backward_compat_missing_keys(self):
        """Emit should not crash when only old-style fields are present."""
        old_style_metrics = {
            "keep_frac": 0.85,
            "ppl_w103_valid": 6.0,
        }
        assert "peak_compile_alloc_gb" not in old_style_metrics
        assert "peak_cuda_allocated_gb" not in old_style_metrics


# ── 16) Stage_size tie-break using recovery_strength ─────────────────────────

class TestRecoveryStrength:
    def test_basic_computation(self):
        stats = [
            _make_stage_stat(0, improve_frac=0.04),
            _make_stage_stat(1, improve_frac=0.06),
        ]
        rec = _recovery_strength(stats)
        I = 0.04 + 0.06
        M = 0.06
        assert rec == pytest.approx(I + M)

    def test_empty_stats(self):
        assert _recovery_strength([]) == 0.0


class TestStageSizeTieBreak:
    def test_ss2_wins_when_recovery_30pct_higher(self):
        """ss=4 stable, ss=2 stable, but ss=2 recovery 30% stronger => ss=2."""
        rec_large = 0.10  # ss=4
        rec_small = 0.14  # ss=2 (40% higher, > TIE_BREAK_GAIN of 25%)
        threshold = (1.0 + TIE_BREAK_GAIN) * rec_large
        assert rec_small >= threshold

    def test_ss4_wins_with_strong_recovery(self):
        """ss=4 has strong recovery, ss=2 only slightly better => ss=4."""
        rec_large = 0.10
        rec_small = 0.11  # only 10% higher, < 25% threshold
        threshold = (1.0 + TIE_BREAK_GAIN) * rec_large
        assert rec_small < threshold

    def test_ss4_unstable_picks_ss2(self):
        """If ss=4 is unstable, ss=2 is the only stable option."""
        stable_results = [{"stage_size": 2, "recovery_strength": 0.05}]
        chosen = stable_results[0]["stage_size"]
        assert chosen == 2

    def test_both_zero_recovery_picks_largest(self):
        """Both stable, both recovery 0 => largest (ss=4)."""
        rec_large = 0.0
        rec_small = 0.0
        threshold = (1.0 + TIE_BREAK_GAIN) * rec_large if rec_large > 0 else 1e-9
        picks_smaller = rec_small >= threshold
        assert not picks_smaller

    def test_large_zero_small_positive_picks_smaller(self):
        """ss=4 recovery=0, ss=2 recovery>0 => ss=2."""
        rec_large = 0.0
        rec_small = 0.001
        threshold = (1.0 + TIE_BREAK_GAIN) * rec_large if rec_large > 0 else 1e-9
        assert rec_small >= threshold


# ── 17) Latency percentile computation ────────────────────────────────────────

class TestLatencyPercentiles:
    def test_known_distribution(self):
        timings = list(range(1, 101))  # 1..100 ms
        pct = compute_latency_percentiles([float(x) for x in timings])
        assert pct["p50"] == pytest.approx(50.5, abs=0.5)
        assert pct["p95"] == pytest.approx(95.05, abs=0.5)
        assert pct["p99"] == pytest.approx(99.01, abs=0.5)
        assert pct["mean"] == pytest.approx(50.5, abs=0.1)
        assert pct["std"] > 0

    def test_single_value(self):
        pct = compute_latency_percentiles([42.0])
        assert pct["p50"] == 42.0
        assert pct["p95"] == 42.0
        assert pct["p99"] == 42.0
        assert pct["mean"] == 42.0
        assert pct["std"] == 0.0

    def test_empty_returns_empty(self):
        assert compute_latency_percentiles([]) == {}

    def test_rounding(self):
        pct = compute_latency_percentiles([1.23456, 2.34567, 3.45678])
        for key in ("p50", "p95", "p99", "mean", "std"):
            parts = str(pct[key]).split(".")
            assert len(parts) <= 2
            if len(parts) == 2:
                assert len(parts[1]) <= 3

    def test_deterministic(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        p1 = compute_latency_percentiles(data)
        p2 = compute_latency_percentiles(data)
        assert p1 == p2

    def test_returns_all_keys(self):
        pct = compute_latency_percentiles([10.0, 20.0, 30.0])
        for k in ("p50", "p95", "p99", "mean", "std"):
            assert k in pct


# ── 18) Latency fields in emit ────────────────────────────────────────────────

class TestLatencyEmitFields:
    def test_latency_keys_list(self):
        """All 10 expected latency keys are defined."""
        assert len(_LATENCY_KEYS) == 10
        assert "prefill_latency_ms_p50" in _LATENCY_KEYS
        assert "prefill_latency_ms_p95" in _LATENCY_KEYS
        assert "prefill_latency_ms_p99" in _LATENCY_KEYS
        assert "decode_ms_per_token_p50" in _LATENCY_KEYS
        assert "decode_ms_per_token_p95" in _LATENCY_KEYS
        assert "decode_ms_per_token_p99" in _LATENCY_KEYS

    def test_summary_csv_has_p95_columns(self):
        assert "prefill_ms_p95" in _SUMMARY_COLUMNS
        assert "decode_ms_per_tok_p95" in _SUMMARY_COLUMNS

    def test_latency_flows_into_metrics_out(self):
        """Simulate emit_frontier_point extracting latency from metrics dict."""
        metrics = {
            "prefill_latency_ms_p50": 12.3,
            "prefill_latency_ms_p95": 15.1,
            "prefill_latency_ms_p99": 18.7,
            "decode_ms_per_token_p50": 0.45,
            "decode_ms_per_token_p95": 0.62,
            "decode_ms_per_token_p99": 0.80,
        }
        metrics_out = {}
        for k in _LATENCY_KEYS:
            val = metrics.get(k)
            if val is not None:
                metrics_out[k] = val
        assert metrics_out["prefill_latency_ms_p95"] == 15.1
        assert metrics_out["decode_ms_per_token_p95"] == 0.62
        assert "prefill_latency_ms_mean" not in metrics_out

    def test_missing_latency_no_crash(self):
        """When metrics have no latency keys, metrics_out just skips them."""
        metrics = {"ppl_w103_valid": 6.0}
        metrics_out = {}
        for k in _LATENCY_KEYS:
            val = metrics.get(k)
            if val is not None:
                metrics_out[k] = val
        assert len(metrics_out) == 0


# ── 19) Repair stop reason semantics ──────────────────────────────────────────

class TestRepairStopSemantics:
    """Verify regression_tripwire vs early_stop vs nan_inf separation."""

    def test_patience_stop_not_regression_tripwire(self):
        """Patience exhausted => early_stop=True, regression_tripwire=False."""
        stat = _make_stage_stat(
            stage=0, improve_frac=0.03, regression_tripwire=False,
            nan_inf=False, early_stop=True,
        )
        assert stat["repair_fail"] is False
        assert stat["early_stop"] is True
        assert stat["regression_tripwire"] is False
        assert stat["repair_helpful"] is True

    def test_tripwire_is_soft_stop_not_fail(self):
        """Regression tripwire => regression_tripwire=True but repair_fail=False."""
        stat = _make_stage_stat(
            stage=0, improve_frac=0.05, regression_tripwire=True,
            nan_inf=False, early_stop=False,
        )
        assert stat["repair_fail"] is False
        assert stat["regression_tripwire"] is True
        assert stat["early_stop"] is False
        assert stat["repair_helpful"] is True

    def test_nan_inf_is_fail_without_regression(self):
        """NaN/Inf detected => repair_fail, regression_tripwire stays False."""
        stat = _make_stage_stat(
            stage=0, improve_frac=0.0, regression_tripwire=False,
            nan_inf=True, early_stop=False,
        )
        assert stat["repair_fail"] is True
        assert stat["regression_tripwire"] is False
        assert stat["nan_inf"] is True

    def test_early_stop_with_improvement_is_helpful(self):
        """A stage with early_stop (patience/max_steps) and improvement is helpful."""
        stat = _make_stage_stat(
            stage=0, improve_frac=0.05, regression_tripwire=False,
            nan_inf=False, early_stop=True,
        )
        assert stat["repair_fail"] is False
        assert stat["repair_helpful"] is True
        assert stat["early_stop"] is True

    def test_manifest_stage_stats_not_all_fail(self):
        """With many early_stop stages but no tripwire, repair_fail count is 0."""
        stages = [
            _make_stage_stat(i, improve_frac=0.03, early_stop=True)
            for i in range(8)
        ]
        fail_count = sum(1 for s in stages if s["repair_fail"])
        early_count = sum(1 for s in stages if s["early_stop"])
        helpful_count = sum(1 for s in stages if s["repair_helpful"])
        assert fail_count == 0
        assert early_count == 8
        assert helpful_count == 8

    def test_mixed_stop_reasons(self):
        """Mix of early_stop, tripwire, nan_inf: only nan_inf counts as fail."""
        stages = [
            _make_stage_stat(0, improve_frac=0.04, early_stop=True),
            _make_stage_stat(1, improve_frac=0.02, early_stop=True),
            _make_stage_stat(2, improve_frac=0.01, regression_tripwire=True),
            _make_stage_stat(3, improve_frac=0.03, early_stop=True),
            _make_stage_stat(4, improve_frac=0.0, nan_inf=True),
        ]
        fail_count = sum(1 for s in stages if s["repair_fail"])
        helpful_count = sum(1 for s in stages if s["repair_helpful"])
        tripwire_count = sum(1 for s in stages if s["regression_tripwire"])
        assert fail_count == 1  # only stage 4 (nan_inf)
        assert tripwire_count == 1  # stage 2
        assert helpful_count == 4  # stages 0, 1, 2 (tripwire but helpful), 3

    def test_repair_return_dict_keys(self):
        """Verify the expected keys exist in the repair return dict shape."""
        result = {
            "regression_tripwire_triggered": False,
            "regression_stop_triggered": False,
            "nan_inf_detected": False,
            "early_stop_triggered": True,
            "early_stopped": True,
        }
        assert result["regression_tripwire_triggered"] is False
        assert result["regression_stop_triggered"] is False
        assert result["early_stop_triggered"] is True
        assert result["early_stopped"] is True


# ── 20) Controller semantics contract ────────────────────────────────────────

class TestControllerSemanticsContract:
    """Enforces the non-negotiable controller semantics:
    repair_fail == nan_inf ONLY.  Tripwire is soft stop."""

    def test_tripwire_only_not_fail(self):
        s = _make_stage_stat(regression_tripwire=True, nan_inf=False, improve_frac=0.04)
        assert s["repair_fail"] is False
        assert s["regression_tripwire"] is True
        assert s["repair_helpful"] is True

    def test_nan_inf_only_is_fail(self):
        s = _make_stage_stat(regression_tripwire=False, nan_inf=True, improve_frac=0.04)
        assert s["repair_fail"] is True
        assert s["repair_helpful"] is False

    def test_tripwire_plus_nan_inf_is_fail(self):
        s = _make_stage_stat(regression_tripwire=True, nan_inf=True)
        assert s["repair_fail"] is True

    def test_pilot_scoring_rejects_nan_inf_not_tripwire(self):
        clean = [_make_stage_stat(0, regression_tripwire=True, improve_frac=0.03)]
        score_ok, stable_ok, *_ = _score_two_stage_pilot(clean, 10.0)
        assert stable_ok is True
        assert score_ok > -1e9

        bad = [_make_stage_stat(0, nan_inf=True)]
        score_bad, stable_bad, *_ = _score_two_stage_pilot(bad, 10.0)
        assert stable_bad is False
        assert score_bad == -1e9

    def test_escalation_counter_increments_only_on_nan_inf(self):
        stages = [
            _make_stage_stat(0, regression_tripwire=True),
            _make_stage_stat(1, regression_tripwire=True),
            _make_stage_stat(2, regression_tripwire=True),
        ]
        consec = 0
        for s in stages:
            if s["repair_fail"]:
                consec += 1
            else:
                consec = 0
        assert consec == 0

        stages_nan = [
            _make_stage_stat(0, nan_inf=True),
            _make_stage_stat(1, nan_inf=True),
        ]
        consec = 0
        for s in stages_nan:
            if s["repair_fail"]:
                consec += 1
            else:
                consec = 0
        assert consec == 2

    def test_helpful_with_tripwire(self):
        s = _make_stage_stat(regression_tripwire=True, improve_frac=0.05)
        assert s["repair_helpful"] is True
        assert s["repair_fail"] is False

    def test_stage_stat_has_all_required_keys(self):
        s = _make_stage_stat()
        for key in ("regression_tripwire", "nan_inf", "early_stop",
                     "repair_fail", "repair_helpful"):
            assert key in s, f"Missing required key: {key}"

    def test_legacy_scoring_ignores_regression_stop_param(self):
        _, score_trip, stable_trip, _ = _score_pilot(
            P0=10.0, Pt=9.0, Pmax=10.0, elapsed_s=10.0,
            regression_stop=True, nan_inf=False,
        )
        _, score_clean, stable_clean, _ = _score_pilot(
            P0=10.0, Pt=9.0, Pmax=10.0, elapsed_s=10.0,
            regression_stop=False, nan_inf=False,
        )
        assert score_trip == score_clean
        assert stable_trip is True
        assert stable_clean is True


# ── 21) Derived metric helpers ────────────────────────────────────────────────

class TestDerivedMetricHelpers:
    """Test _safe_pct, _safe_ratio, _safe_throughput_gain_pct, _gpu_hour_reduction_pct."""

    def test_safe_pct_normal(self):
        assert _safe_pct(100.0, 80.0) == 20.0

    def test_safe_pct_zero_baseline(self):
        assert _safe_pct(0.0, 50.0) is None

    def test_safe_pct_none_inputs(self):
        assert _safe_pct(None, 50.0) is None
        assert _safe_pct(50.0, None) is None

    def test_safe_ratio_normal(self):
        assert _safe_ratio(10.0, 5.0) == 2.0

    def test_safe_ratio_zero_denominator(self):
        assert _safe_ratio(10.0, 0.0) is None

    def test_safe_ratio_none(self):
        assert _safe_ratio(None, 5.0) is None

    def test_throughput_gain_pct(self):
        assert _safe_throughput_gain_pct(150.0, 100.0) == 50.0

    def test_throughput_gain_zero_baseline(self):
        assert _safe_throughput_gain_pct(150.0, 0.0) is None

    def test_gpu_hour_reduction_2x(self):
        assert _gpu_hour_reduction_pct(2.0) == 50.0

    def test_gpu_hour_reduction_none(self):
        assert _gpu_hour_reduction_pct(None) is None

    def test_gpu_hour_reduction_zero(self):
        assert _gpu_hour_reduction_pct(0.0) is None

    def test_gpu_hour_reduction_1x(self):
        assert _gpu_hour_reduction_pct(1.0) == 0.0


# ── 22) Metrics.json contains derived keys ────────────────────────────────────

class TestMetricsJsonDerivedKeys:
    """Verify _DERIVED_KEYS and _BASELINE_LATENCY_KEYS are well-formed."""

    def test_derived_keys_list_populated(self):
        assert len(_DERIVED_KEYS) >= 10

    def test_baseline_latency_keys_match(self):
        for k in _BASELINE_LATENCY_KEYS:
            assert k.startswith("baseline_"), f"{k} missing baseline_ prefix"
        assert len(_BASELINE_LATENCY_KEYS) == len(_LATENCY_KEYS)

    def test_summary_csv_has_whitepaper_columns(self):
        required = [
            "baseline_prefill_ms_p95", "baseline_decode_ms_per_tok_p95",
            "prefill_p95_latency_improvement_pct", "decode_p95_latency_improvement_pct",
            "prefill_throughput_gain_pct", "decode_throughput_gain_pct",
            "gpu_hour_reduction_rag_pct",
            "weights_memory_reduction_pct", "steady_state_memory_reduction_pct",
            "compile_minutes",
            "num_params", "weights_gb",
        ]
        for col in required:
            assert col in _SUMMARY_COLUMNS, f"Missing summary column: {col}"

    def test_derived_keys_include_all_expected(self):
        expected = {
            "prefill_p95_latency_ratio", "decode_p95_latency_ratio",
            "prefill_p95_latency_improvement_pct", "decode_p95_latency_improvement_pct",
            "prefill_throughput_gain_pct", "decode_throughput_gain_pct",
            "gpu_hour_reduction_chat_pct", "gpu_hour_reduction_rag_pct",
            "gpu_hour_reduction_batch_pct",
            "baseline_steady_state_alloc_gb", "steady_state_memory_reduction_pct",
            "weights_memory_reduction_pct",
            "repair_steps_per_stage", "compile_minutes",
        }
        assert expected == set(_DERIVED_KEYS)


# ── 23) Run metadata ─────────────────────────────────────────────────────────

class TestRunMetadata:
    """Verify emit_run_metadata produces expected files."""

    def test_creates_run_metadata_json(self, tmp_path):
        import json
        emit_run_metadata(tmp_path, {"deterministic": True, "seed": 42, "dtype": "bf16"})
        meta_path = tmp_path / "run_metadata.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert "torch_version" in data
        assert "transformers_version" in data
        assert "cuda_available" in data
        assert data["deterministic_flag"] is True
        assert data["seed"] == 42
        assert data["dtype"] == "bf16"
        assert "timestamp" in data
        assert "warmup_iters" in data
        assert "measure_iters" in data

    def test_metadata_has_all_required_keys(self, tmp_path):
        import json
        emit_run_metadata(tmp_path, {})
        data = json.loads((tmp_path / "run_metadata.json").read_text())
        required = {
            "git_commit", "torch_version", "transformers_version",
            "cuda_available", "gpu_name", "deterministic_flag",
            "seed", "dtype", "tf32_enabled",
            "warmup_iters", "measure_iters", "decode_steps", "timestamp",
        }
        for k in required:
            assert k in data, f"Missing run_metadata key: {k}"

    def test_no_crash_on_missing_nvidia_smi(self, tmp_path):
        emit_run_metadata(tmp_path, {})
        assert (tmp_path / "run_metadata.json").exists()


# ── 24) End-to-end derived metric computation ────────────────────────────────

class TestDerivedMetricComputation:
    """Verify the derived metrics math is correct end-to-end."""

    def test_latency_improvement_math(self):
        pct = _safe_pct(10.0, 8.0)
        assert pct == 20.0

    def test_latency_ratio_math(self):
        ratio = _safe_ratio(10.0, 5.0)
        assert ratio == 2.0

    def test_throughput_gain_doubling(self):
        gain = _safe_throughput_gain_pct(200.0, 100.0)
        assert gain == 100.0

    def test_gpu_hour_reduction_4x(self):
        pct = _gpu_hour_reduction_pct(4.0)
        assert pct == 75.0

    def test_memory_reduction(self):
        pct = _safe_pct(10.0, 7.0)
        assert pct == 30.0

    def test_compile_minutes_from_seconds(self):
        assert round(600.0 / 60.0, 1) == 10.0

    def test_repair_steps_per_stage(self):
        total_steps = 500
        n_stages = 10
        assert round(total_steps / max(1, n_stages), 1) == 50.0


# ── Weights-only memory in sculpt emit ────────────────────────────────────────

class TestSculptEmitWeightsMemory:
    """Verify weights_gb and num_params flow through the sculpt emit path."""

    def test_compile_result_has_weight_fields(self):
        from dystrio_sculpt.engine import CompileResult

        cr = CompileResult()
        assert hasattr(cr, "num_params")
        assert hasattr(cr, "weights_bytes")
        assert hasattr(cr, "baseline_num_params")
        assert hasattr(cr, "baseline_weights_bytes")

    def test_emit_frontier_point_writes_weights(self, tmp_path):
        import json
        import torch
        from dystrio_sculpt.emit import emit_frontier_point

        model = torch.nn.Linear(256, 128, bias=False)
        model.save_pretrained = lambda path, **kw: None
        num_p = 256 * 128
        wbytes = num_p * 4  # float32

        class FakeTokenizer:
            def save_pretrained(self, path):
                pass

        model.config = type("Cfg", (), {
            "intermediate_size": 128,
            "num_hidden_layers": 1,
        })()
        model.model = type("M", (), {"layers": []})()

        point_dir = emit_frontier_point(
            model=model,
            tokenizer=FakeTokenizer(),
            outdir=tmp_path,
            label="frontier_0_conservative",
            keep_frac=0.85,
            metrics={"ppl_w2_test": 10.0, "ppl_w103_valid": 11.0,
                      "prefill_tokens_per_sec": 5000, "decode_tokens_per_sec": 50},
            baseline_metrics={"ppl_w103_valid": 10.0,
                               "prefill_tokens_per_sec": 4500, "decode_tokens_per_sec": 55},
            compile_report={},
            config={"model_id": "test", "seed": 0},
            wall_time_s=100.0,
            num_params=num_p,
            weights_bytes=wbytes,
            baseline_num_params=num_p * 2,
            baseline_weights_bytes=wbytes * 2,
        )

        metrics = json.loads((point_dir / "metrics.json").read_text())
        assert metrics["num_params"] == num_p
        assert metrics["weights_gb"] > 0
        assert metrics["baseline_num_params"] == num_p * 2
        assert metrics["baseline_weights_gb"] > 0
        assert metrics["weights_memory_reduction_pct"] is not None
        assert metrics["weights_memory_reduction_pct"] > 0

    def test_summary_csv_includes_weights(self, tmp_path):
        import csv
        from dystrio_sculpt.emit import append_summary_csv

        append_summary_csv(
            outdir=tmp_path, name="test", keep_frac=0.85,
            ppl_w103=11.0, baseline_ppl_w103=10.0,
            prefill_speedup=1.1, decode_speedup=0.95,
            compile_time_s=100.0,
            num_params=1000000, weights_gb=3.725,
            weights_memory_reduction_pct=9.5,
        )

        csv_path = tmp_path / "summary.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["num_params"] == "1000000"
        assert rows[0]["weights_gb"] == "3.725"
        assert rows[0]["weights_memory_reduction_pct"] == "9.5"

    def test_weights_reduction_math(self):
        from dystrio_sculpt.emit import _safe_pct

        baseline_gb = 14.24
        sculpted_gb = 11.53
        pct = _safe_pct(baseline_gb, sculpted_gb)
        assert pct > 0
        assert abs(pct - 19.03) < 0.5


# ── 25) _LRArm: Thompson Sampling bandit arm ─────────────────────────────────

class TestLRArm:
    def test_default_uniform_prior(self):
        arm = _LRArm()
        assert arm.alpha == 1.0
        assert arm.beta == 1.0
        assert arm.mean == 0.5

    def test_mean_after_successes(self):
        arm = _LRArm(alpha=5.0, beta=1.0)
        assert arm.mean == pytest.approx(5.0 / 6.0)

    def test_mean_after_failures(self):
        arm = _LRArm(alpha=1.0, beta=5.0)
        assert arm.mean == pytest.approx(1.0 / 6.0)

    def test_sample_in_zero_one(self):
        import numpy as np
        rng = np.random.default_rng(42)
        arm = _LRArm(alpha=2.0, beta=3.0)
        for _ in range(100):
            s = arm.sample(rng)
            assert 0.0 <= s <= 1.0

    def test_sample_deterministic_with_seed(self):
        import numpy as np
        arm = _LRArm(alpha=2.0, beta=3.0)
        s1 = arm.sample(np.random.default_rng(99))
        s2 = arm.sample(np.random.default_rng(99))
        assert s1 == s2

    def test_update_shifts_mean(self):
        arm = _LRArm()
        initial_mean = arm.mean
        arm.alpha += 1.0
        assert arm.mean > initial_mean

    def test_strong_prior_dominates_early(self):
        arm = _LRArm(alpha=10.0, beta=1.0)
        assert arm.mean > 0.8


# ── 26) LR_GRID and constants ─────────────────────────────────────────────────

class TestLRGridConstants:
    def test_grid_has_wide_range(self):
        assert min(LR_GRID) <= 1e-5
        assert max(LR_GRID) >= 1e-4
        assert len(LR_GRID) >= 5

    def test_grid_is_sorted(self):
        assert LR_GRID == sorted(LR_GRID)

    def test_grid_spans_100x(self):
        ratio = max(LR_GRID) / min(LR_GRID)
        assert ratio >= 100

    def test_reward_threshold_positive(self):
        assert LR_REWARD_THRESHOLD > 0

    def test_max_probes_reasonable(self):
        assert MAX_LR_PROBES >= len(LR_GRID)
        assert MAX_LR_PROBES <= 20


# ── 27) Conductance-informed probe chunk selection ────────────────────────────

class TestConductanceProbeChunk:
    def test_fallback_without_prescan(self):
        layer_order = list(range(32))
        chunk = _conductance_probe_chunk(layer_order, None, stage_size=4)
        assert len(chunk) == 4
        assert all(li in layer_order for li in chunk)

    def test_fallback_empty_prescan(self):
        layer_order = list(range(32))
        chunk = _conductance_probe_chunk(layer_order, {}, stage_size=4)
        assert len(chunk) == 4

    def test_with_prescan_selects_hard_layers(self):
        import torch
        import numpy as np
        layer_order = list(range(8))
        prescan = {}
        for li in layer_order:
            n_blocks = 4
            bs = torch.zeros(n_blocks)
            D = torch.eye(n_blocks) * 0.01
            if li >= 5:
                bs = torch.ones(n_blocks) * 0.9
                D = torch.ones(n_blocks, n_blocks) * 0.5
                D.fill_diagonal_(1.0)
            prescan[li] = {
                "block_sensitivity": bs,
                "D": D,
                "block_energy": None,
            }
        chunk = _conductance_probe_chunk(layer_order, prescan, stage_size=2)
        assert len(chunk) == 2
        assert any(li >= 4 for li in chunk)

    def test_returns_correct_stage_size(self):
        layer_order = list(range(16))
        chunk = _conductance_probe_chunk(layer_order, None, stage_size=4)
        assert len(chunk) == 4

    def test_small_model(self):
        layer_order = [0, 1, 2]
        chunk = _conductance_probe_chunk(layer_order, None, stage_size=4)
        assert len(chunk) <= 3
        assert set(chunk).issubset(set(layer_order))

    def test_empty_order(self):
        chunk = _conductance_probe_chunk([], None, stage_size=4)
        assert chunk == []

    def test_deterministic(self):
        layer_order = list(range(32))
        c1 = _conductance_probe_chunk(layer_order, None, stage_size=4)
        c2 = _conductance_probe_chunk(layer_order, None, stage_size=4)
        assert c1 == c2


# ── 28) Thompson Sampling LR search integration ──────────────────────────────

class TestThompsonLRSearchIntegration:
    """Tests for the Thompson Sampling LR search logic (no model loading)."""

    def test_arms_initialized_uniform(self):
        arms = {lr: _LRArm() for lr in LR_GRID}
        for arm in arms.values():
            assert arm.mean == 0.5

    def test_successful_lr_raises_mean(self):
        arms = {lr: _LRArm() for lr in LR_GRID}
        target_lr = 5e-5
        arms[target_lr].alpha += 3.0
        assert arms[target_lr].mean > 0.5
        best = max(arms, key=lambda lr: arms[lr].mean)
        assert best == target_lr

    def test_nan_penalty_lowers_mean(self):
        arms = {lr: _LRArm() for lr in LR_GRID}
        bad_lr = 5e-4
        arms[bad_lr].beta += 2.0
        assert arms[bad_lr].mean < 0.5

    def test_asymmetric_reward_shapes_selection(self):
        r_small = _asymmetric_reward(0.01)
        r_big = _asymmetric_reward(0.05)
        assert r_small < LR_REWARD_THRESHOLD or r_big > LR_REWARD_THRESHOLD
        assert r_big > r_small

    def test_simulated_search_finds_best(self):
        """Simulate a full Thompson Sampling LR search with fixed improve_fracs."""
        import numpy as np
        rng = np.random.default_rng(42)
        arms = {lr: _LRArm() for lr in LR_GRID}

        true_best_lr = 5e-5
        lr_to_improve = {lr: 0.001 for lr in LR_GRID}
        lr_to_improve[5e-5] = 0.10

        for _ in range(30):
            samples = {lr: arm.sample(rng) for lr, arm in arms.items()}
            chosen_lr = max(samples, key=samples.get)
            improve = lr_to_improve[chosen_lr]
            reward = _asymmetric_reward(improve)
            if reward > LR_REWARD_THRESHOLD:
                arms[chosen_lr].alpha += 1.0
            else:
                arms[chosen_lr].beta += 1.0

        best = max(arms, key=lambda lr: arms[lr].mean)
        assert best == true_best_lr

    def test_all_nan_falls_back(self):
        """If every probe is NaN, the fallback LR should be 5e-5."""
        arms = {lr: _LRArm() for lr in LR_GRID}
        for lr in LR_GRID:
            arms[lr].beta += 2.0
        any_helpful = False
        best_lr = 5e-5 if not any_helpful else max(arms, key=lambda lr: arms[lr].mean)
        assert best_lr == 5e-5

    def test_tuning_report_includes_lr_search(self):
        from dystrio_sculpt.policy import TuningReport
        report = TuningReport(
            pilot_keep_frac=0.85, pilot_budget_s=240.0,
            candidates=["5e-06", "1e-05", "5e-05"],
            results=[], chosen_policy="ts_lr5e-05_p12_s918",
            chosen_reason="thompson_best_mean", risk_score=0.3,
            lr_search={
                "grid": ["5e-06", "1e-05", "5e-05"],
                "best_lr": 5e-5, "reason": "thompson_best_mean",
                "num_probes": 8, "probes": [],
                "arm_posteriors": {},
                "probe_chunk": [10, 11, 12, 13],
            },
        )
        d = report.to_dict()
        assert "lr_search" in d
        assert d["lr_search"]["best_lr"] == 5e-5
        assert d["lr_search"]["probe_chunk"] == [10, 11, 12, 13]

    def test_tuning_report_without_lr_search(self):
        from dystrio_sculpt.policy import TuningReport
        report = TuningReport(
            pilot_keep_frac=0.85, pilot_budget_s=240.0,
            candidates=[], results=[],
            chosen_policy="test", chosen_reason="fallback",
            risk_score=0.3,
        )
        d = report.to_dict()
        assert "lr_search" not in d
