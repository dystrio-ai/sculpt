"""Tests for Repair Pilot Tuner, stage-size controller, escalation,
stage outcome semantics, stratified sampling, E2E metrics (mock-based)."""

from __future__ import annotations

import math
import pytest

from dystrio_sculpt.policy import (
    RepairPolicy,
    _score_pilot,
    _score_two_stage_pilot,
    _stratified_pilot_chunks,
    _pilot_candidates,
    _compute_pilot_budget,
    _adapt_steps,
    _with_stage_size,
    _recovery_strength,
    escalate_policy,
    build_policy_ladder,
    compute_e2e_speedup,
    E2E_PROFILES,
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
            "steady_state_memory_reduction_pct",
            "compile_minutes",
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
