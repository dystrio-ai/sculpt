"""JSONL dataset schema for the Dystrio Efficiency Dataset.

This is the moat. Every factory run deposits a structured record with
data nobody else captures: per-layer structural risk, compression
decisions, repair dynamics, calibration geometry, and cross-tier
quality curves. The dataset gets more valuable with every model
we process.

Schema is append-only and forward-compatible — new fields can be added
without breaking old records (from_dict ignores unknown keys).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"


# ── Per-layer insights (the stuff nobody else has) ────────────────────────────

@dataclass
class LayerInsight:
    """What happened to a single layer during compression.

    Captures the structural decision (what was kept), the risk profile
    (how compressible this layer was), and the repair outcome.
    """
    layer_idx: int = 0
    original_ffn: int = 0
    ffn_kept: int = 0
    keep_frac: float = 0.0
    kept_blocks: int = 0
    risk_score: float = 0.0
    sensitivity_mean: float = 0.0
    sensitivity_max: float = 0.0
    geometry_condition_number: float = 0.0
    repair_helped: bool = False
    ppl_before_repair: float = 0.0
    ppl_after_repair: float = 0.0


# ── Tier-level results ────────────────────────────────────────────────────────

@dataclass
class TierRecord:
    """Full record for one compression tier."""
    name: str
    keep_frac: float
    ppl_ratio: float

    # Performance
    prefill_tps: float = 0.0
    ttft_p95_ms: float = 0.0
    decode_tps: float = 0.0
    weights_gb: float = 0.0
    num_params: int = 0

    # Speedups relative to baseline
    prefill_speedup: float = 0.0
    decode_speedup: float = 0.0

    # VRAM
    peak_vram_gb: float = 0.0
    steady_state_vram_gb: float = 0.0

    # Compilation
    wall_time_s: float = 0.0
    total_repair_steps: int = 0
    policy_name: str = ""
    layers_compressed: int = 0
    guardrail_triggered: bool = False
    early_stopped: bool = False

    # Per-layer structural data — THE UNIQUE INSIGHT
    layer_insights: List[LayerInsight] = field(default_factory=list)

    # Risk
    risk_score: float = 0.0

    # Publishing
    artifact_url: str = ""


# ── Baseline ──────────────────────────────────────────────────────────────────

@dataclass
class BaselineInfo:
    ppl_wikitext: float = 0.0
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    ttft_p95_ms: float = 0.0
    weights_gb: float = 0.0
    num_params: int = 0
    vram_gb: float = 0.0


# ── Environment ───────────────────────────────────────────────────────────────

@dataclass
class EnvironmentInfo:
    gpu: str = ""
    gpu_count: int = 1
    dtype: str = "bf16"
    torch_version: str = ""
    cuda_version: str = ""
    transformers_version: str = ""
    dystrio_version: str = ""
    git_sha: str = ""


# ── Risk profile (model-level structural analysis) ───────────────────────────

@dataclass
class RiskProfile:
    """Model-level risk summary from prescan."""
    aggregate_risk: float = 0.0
    layer_risks: List[float] = field(default_factory=list)
    compressibility_order: List[int] = field(default_factory=list)
    high_risk_layers: List[int] = field(default_factory=list)
    low_risk_layers: List[int] = field(default_factory=list)


# ── Policy trace ──────────────────────────────────────────────────────────────

@dataclass
class PolicyTrace:
    """How the repair policy was selected and whether it escalated."""
    initial_policy: str = ""
    final_policy: str = ""
    escalation_applied: bool = False
    escalation_details: str = ""
    pilot_keep_frac: float = 0.0
    pilot_ppl_ratio: float = 0.0
    total_repair_steps: int = 0
    stage_size: int = 0


# ── Top-level record ─────────────────────────────────────────────────────────

@dataclass
class DatasetRecord:
    """One row in the Dystrio Efficiency Dataset.

    Contains everything needed to understand what happened during an
    optimization run and why. This is the data that trains future
    optimization decisions — the more models we process, the better
    we get at predicting which layers compress well, which policies
    to use, and which architectures respond to which techniques.
    """

    # Identity
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = SCHEMA_VERSION
    source: str = "factory"  # "factory" | "backfill"
    model_id: str = ""
    source_repo: str = ""
    error_category: Optional[str] = None

    # Architecture fingerprint
    architecture: Dict[str, Any] = field(default_factory=dict)

    # What we configured
    optimization_config: Dict[str, Any] = field(default_factory=dict)

    # Where we ran
    environment: EnvironmentInfo = field(default_factory=EnvironmentInfo)

    # Structural risk profile (from prescan)
    risk_profile: RiskProfile = field(default_factory=RiskProfile)

    # How the policy was chosen
    policy_trace: PolicyTrace = field(default_factory=PolicyTrace)

    # Per-tier results (each with per-layer insights)
    tiers: List[TierRecord] = field(default_factory=list)

    # Baseline measurements
    baseline: BaselineInfo = field(default_factory=BaselineInfo)

    # Raw decision trace (compile_report JSON for deep analysis)
    decision_trace: str = ""

    # ── Scoring ────────────────────────────────────────────────────────────

    def completeness_score(self) -> Dict[str, float]:
        """Score how complete this record is, per section.

        Returns scores from 0.0 to 1.0 for each section plus an overall.
        Context-aware: backfill records are not penalized for fields that
        only factory runs can produce.
        """
        is_factory = self.source == "factory"
        scores: Dict[str, float] = {}

        # Descriptor
        arch = self.architecture
        desc_checks = [
            bool(arch.get("family")),
            bool(arch.get("num_layers")),
            bool(arch.get("hidden_size")),
            bool(arch.get("mlp_type")),
            bool(arch.get("support_state")),
        ]
        scores["descriptor"] = sum(desc_checks) / len(desc_checks)

        # Tiers
        if self.tiers:
            tier_scores = []
            for t in self.tiers:
                checks = [
                    t.keep_frac > 0,
                    t.ppl_ratio > 0,
                    t.prefill_tps > 0,
                    t.decode_tps > 0,
                    t.weights_gb > 0,
                ]
                if is_factory:
                    checks.append(len(t.layer_insights) > 0)
                tier_scores.append(sum(checks) / len(checks))
            scores["tiers"] = round(sum(tier_scores) / len(tier_scores), 3)
        else:
            scores["tiers"] = 0.0

        # Risk profile (only expected from factory)
        if is_factory:
            rp = self.risk_profile
            rp_checks = [
                rp.aggregate_risk > 0,
                len(rp.layer_risks) > 0,
                len(rp.compressibility_order) > 0,
            ]
            scores["risk_profile"] = round(sum(rp_checks) / len(rp_checks), 3)
        else:
            scores["risk_profile"] = 1.0  # not expected, don't penalize

        # Policy trace (only expected from factory)
        if is_factory:
            pt = self.policy_trace
            pt_checks = [
                bool(pt.initial_policy),
                bool(pt.final_policy),
                pt.total_repair_steps > 0,
            ]
            scores["policy_trace"] = round(sum(pt_checks) / len(pt_checks), 3)
        else:
            scores["policy_trace"] = 1.0

        # Decision trace
        scores["decision_trace"] = 1.0 if self.decision_trace else (0.0 if is_factory else 1.0)

        scores["overall"] = round(sum(scores.values()) / len(scores), 3)
        return scores

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tiers"] = [asdict(t) for t in self.tiers]
        d["baseline"] = asdict(self.baseline)
        d["environment"] = asdict(self.environment)
        d["risk_profile"] = asdict(self.risk_profile)
        d["policy_trace"] = asdict(self.policy_trace)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DatasetRecord:
        rec = cls(
            run_id=d.get("run_id", str(uuid.uuid4())),
            timestamp=d.get("timestamp", ""),
            schema_version=d.get("schema_version", "0.0"),
            source=d.get("source", "backfill"),
            model_id=d.get("model_id", ""),
            source_repo=d.get("source_repo", ""),
            error_category=d.get("error_category"),
            architecture=d.get("architecture", {}),
            optimization_config=d.get("optimization_config", {}),
            decision_trace=d.get("decision_trace", ""),
        )

        def _safe_build(cls_, data):
            if not isinstance(data, dict):
                return cls_()
            return cls_(**{k: v for k, v in data.items() if k in cls_.__dataclass_fields__})

        if "environment" in d:
            rec.environment = _safe_build(EnvironmentInfo, d["environment"])
        if "baseline" in d:
            rec.baseline = _safe_build(BaselineInfo, d["baseline"])
        if "risk_profile" in d and isinstance(d["risk_profile"], dict):
            rec.risk_profile = _safe_build(RiskProfile, d["risk_profile"])
        if "policy_trace" in d and isinstance(d["policy_trace"], dict):
            rec.policy_trace = _safe_build(PolicyTrace, d["policy_trace"])

        if "tiers" in d and isinstance(d["tiers"], list):
            for t in d["tiers"]:
                tr = _safe_build(TierRecord, t)
                if "layer_insights" in t and isinstance(t["layer_insights"], list):
                    tr.layer_insights = [
                        _safe_build(LayerInsight, li) for li in t["layer_insights"]
                    ]
                rec.tiers.append(tr)

        return rec

    def validate(self) -> List[str]:
        """Validate the record. Returns list of error strings (empty = valid).

        For source="factory" records, enforces the minimum viable rich-record
        contract. For source="backfill", only checks basic structure.
        """
        issues: List[str] = []

        # ── Universal checks (all sources) ────────────────────────────────
        if not self.model_id:
            issues.append("model_id is required")
        if not self.run_id:
            issues.append("run_id is required")
        if not self.tiers:
            issues.append("at least one tier is required")
        for i, t in enumerate(self.tiers):
            if not t.name:
                issues.append(f"tier[{i}].name is required")
            if t.keep_frac <= 0 or t.keep_frac > 1:
                issues.append(f"tier[{i}].keep_frac must be in (0, 1]")

        if self.source != "factory":
            return issues

        # ── Factory-only: minimum viable rich-record contract ─────────────
        if not self.schema_version:
            issues.append("schema_version is required for factory runs")
        if not self.timestamp:
            issues.append("timestamp is required for factory runs")

        # Descriptor
        arch = self.architecture
        for key in ("family", "num_layers", "hidden_size", "mlp_type", "support_state"):
            if not arch.get(key):
                issues.append(f"architecture.{key} is required for factory runs")

        # Tier metrics + layer insights
        for i, t in enumerate(self.tiers):
            if t.ppl_ratio <= 0:
                issues.append(f"tier[{i}].ppl_ratio must be > 0")
            if t.prefill_tps <= 0:
                issues.append(f"tier[{i}].prefill_tps must be > 0")
            if t.decode_tps <= 0:
                issues.append(f"tier[{i}].decode_tps must be > 0")
            if t.weights_gb <= 0:
                issues.append(f"tier[{i}].weights_gb must be > 0")
            if not t.layer_insights:
                issues.append(f"tier[{i}].layer_insights must be non-empty for factory runs")

        # Optimization trace: need at least policy_trace OR decision_trace
        has_policy = bool(self.policy_trace.initial_policy or self.policy_trace.final_policy)
        has_decision = bool(self.decision_trace)
        if not has_policy and not has_decision:
            issues.append("policy_trace or decision_trace must be present for factory runs")

        # Risk profile must have layer_risks
        if not self.risk_profile.layer_risks:
            issues.append("risk_profile.layer_risks must be non-empty for factory runs")

        return issues
