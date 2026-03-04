"""Bench audit: validate publishability assumptions for benchmark results."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_log = logging.getLogger(__name__)

# Tolerance for ppl_ratio check: |computed - recorded| / recorded
PPL_RATIO_TOLERANCE = 0.01
MIN_EFFECTIVE_PROMPT_PCT = 0.95


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _float_or_none(val) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ── individual checks ─────────────────────────────────────────────────────────

def _check_promptpack_provenance(
    results_dir: Path,
) -> List[Dict[str, Any]]:
    """Ensure all models used the same prompt pack hash per workload."""
    findings: List[Dict[str, Any]] = []
    workload_hashes: Dict[str, Dict[str, str]] = {}

    for meta_path in sorted(results_dir.rglob("run_metadata.json")):
        try:
            meta = _read_json(meta_path)
        except Exception:
            continue
        wl = meta.get("workload")
        ph = meta.get("promptpack_hash")
        mid = meta.get("model_id", "unknown")
        if wl is None or ph is None:
            continue
        workload_hashes.setdefault(wl, {})[mid] = ph

    for wl, model_hashes in workload_hashes.items():
        unique = set(model_hashes.values())
        if len(unique) > 1:
            findings.append({
                "check": "promptpack_provenance",
                "status": "FAIL",
                "workload": wl,
                "detail": f"Multiple prompt pack hashes across models: {model_hashes}",
            })
        elif len(unique) == 1:
            findings.append({
                "check": "promptpack_provenance",
                "status": "PASS",
                "workload": wl,
                "detail": f"All models used hash {unique.pop()}",
            })

    if not workload_hashes:
        findings.append({
            "check": "promptpack_provenance",
            "status": "WARN",
            "detail": "No promptpack_hash found in any run_metadata.json",
        })
    return findings


def _check_baseline_anchoring(
    bench_out: Path,
) -> List[Dict[str, Any]]:
    """Ensure baseline model and wikitext ppl are present and ppl_ratio is consistent."""
    findings: List[Dict[str, Any]] = []

    root_meta_path = bench_out / "run_metadata.json"
    baseline_model_id = None
    if root_meta_path.exists():
        try:
            root_meta = _read_json(root_meta_path)
            baseline_model_id = root_meta.get("baseline_model_id")
        except Exception:
            pass

    if baseline_model_id is None:
        findings.append({
            "check": "baseline_anchoring",
            "status": "WARN",
            "detail": "No baseline_model_id recorded in root run_metadata.json",
        })

    csv_path = bench_out / "benchmarks.csv"
    if not csv_path.exists():
        findings.append({
            "check": "baseline_anchoring",
            "status": "FAIL",
            "detail": "benchmarks.csv not found",
        })
        return findings

    rows = _read_csv(csv_path)
    model_ppls: Dict[str, Optional[float]] = {}
    for r in rows:
        if r.get("workload") == "wikitext":
            model_ppls[r["model_id"]] = _float_or_none(r.get("ppl_wikitext"))

    if baseline_model_id and baseline_model_id not in model_ppls:
        findings.append({
            "check": "baseline_anchoring",
            "status": "FAIL",
            "detail": f"Baseline model '{baseline_model_id}' has no wikitext row in benchmarks.csv",
        })
        return findings

    baseline_ppl = model_ppls.get(baseline_model_id or "")
    if baseline_ppl is None and baseline_model_id:
        findings.append({
            "check": "baseline_anchoring",
            "status": "FAIL",
            "detail": f"Baseline model '{baseline_model_id}' has null ppl_wikitext",
        })
        return findings

    if baseline_ppl is not None and baseline_ppl > 0:
        for r in rows:
            recorded_ratio = _float_or_none(r.get("ppl_ratio"))
            mid = r["model_id"]
            mppl = model_ppls.get(mid)
            if recorded_ratio is not None and mppl is not None:
                expected = mppl / baseline_ppl
                if abs(expected - recorded_ratio) / max(recorded_ratio, 1e-9) > PPL_RATIO_TOLERANCE:
                    findings.append({
                        "check": "baseline_anchoring",
                        "status": "FAIL",
                        "detail": (
                            f"ppl_ratio mismatch for {mid}: "
                            f"expected {expected:.4f} got {recorded_ratio:.4f}"
                        ),
                    })
        if not any(f["status"] == "FAIL" for f in findings if f["check"] == "baseline_anchoring"):
            findings.append({
                "check": "baseline_anchoring",
                "status": "PASS",
                "detail": f"Baseline={baseline_model_id}, ppl={baseline_ppl:.4f}; ratios consistent",
            })
    return findings


def _check_environment_parity(
    results_dir: Path,
) -> List[Dict[str, Any]]:
    """Confirm deterministic flag, seed, dtype, versions are consistent across runs."""
    findings: List[Dict[str, Any]] = []
    fields_to_check = ["dtype", "torch_version", "transformers_version", "deterministic", "seed"]
    field_vals: Dict[str, Set[str]] = {f: set() for f in fields_to_check}

    gpu_names: Set[str] = set()
    meta_count = 0

    for meta_path in sorted(results_dir.rglob("run_metadata.json")):
        try:
            meta = _read_json(meta_path)
        except Exception:
            continue
        meta_count += 1
        for f in fields_to_check:
            val = meta.get(f)
            if val is not None:
                field_vals[f].add(str(val))
        gn = meta.get("gpu_name")
        if gn:
            gpu_names.add(str(gn))

    if meta_count == 0:
        findings.append({
            "check": "environment_parity",
            "status": "WARN",
            "detail": "No run_metadata.json found in results",
        })
        return findings

    for f in fields_to_check:
        vals = field_vals[f]
        if len(vals) > 1:
            findings.append({
                "check": "environment_parity",
                "status": "WARN",
                "field": f,
                "detail": f"Inconsistent {f} across runs: {sorted(vals)}",
            })

    if not any(f_item["status"] == "WARN" for f_item in findings if f_item["check"] == "environment_parity"):
        findings.append({
            "check": "environment_parity",
            "status": "PASS",
            "detail": f"Environment consistent across {meta_count} metadata files",
        })

    if gpu_names:
        findings.append({
            "check": "environment_parity",
            "status": "INFO",
            "detail": f"GPU(s): {', '.join(sorted(gpu_names))}",
        })

    return findings


def _check_prompt_id_parity(
    results_dir: Path,
) -> List[Dict[str, Any]]:
    """Ensure all models used the same prompt IDs per workload."""
    findings: List[Dict[str, Any]] = []
    wl_model_ids: Dict[str, Dict[str, List[str]]] = {}

    for pp_path in sorted(results_dir.rglob("per_prompt.csv")):
        parts = pp_path.relative_to(results_dir).parts
        if len(parts) < 3:
            continue
        model_dir = parts[0]
        workload = parts[1]
        try:
            rows = _read_csv(pp_path)
        except Exception:
            continue
        ids = [r.get("id", "") for r in rows]
        wl_model_ids.setdefault(workload, {})[model_dir] = ids

    for wl, model_id_lists in wl_model_ids.items():
        models = list(model_id_lists.keys())
        if len(models) < 2:
            continue
        ref = model_id_lists[models[0]]
        for m in models[1:]:
            other = model_id_lists[m]
            if other != ref:
                findings.append({
                    "check": "prompt_id_parity",
                    "status": "FAIL",
                    "workload": wl,
                    "detail": f"Prompt IDs differ between {models[0]} and {m}",
                })
                break
        else:
            findings.append({
                "check": "prompt_id_parity",
                "status": "PASS",
                "workload": wl,
                "detail": f"All {len(models)} models share identical prompt IDs",
            })

    return findings


def _check_error_rates(
    bench_out: Path,
) -> List[Dict[str, Any]]:
    """Warn if errors_skipped_prompts > 0 or effective prompts < 95%."""
    findings: List[Dict[str, Any]] = []
    csv_path = bench_out / "benchmarks.csv"
    if not csv_path.exists():
        return findings

    for r in _read_csv(csv_path):
        errors = _float_or_none(r.get("errors_skipped_prompts"))
        num_prompts = _float_or_none(r.get("num_prompts"))
        mid = r.get("model_id", "?")
        wl = r.get("workload", "?")

        if errors is not None and errors > 0:
            effective_pct = 1.0
            if num_prompts and num_prompts > 0:
                effective_pct = (num_prompts - errors) / num_prompts
            status = "FAIL" if effective_pct < MIN_EFFECTIVE_PROMPT_PCT else "WARN"
            findings.append({
                "check": "error_rates",
                "status": status,
                "detail": (
                    f"{mid}/{wl}: {int(errors)} errors, "
                    f"{effective_pct:.1%} effective prompts"
                ),
            })

    if not findings:
        findings.append({
            "check": "error_rates",
            "status": "PASS",
            "detail": "No skipped prompts across all workloads",
        })
    return findings


def _check_memory_claims(
    bench_out: Path,
) -> List[Dict[str, Any]]:
    """Warn if memory_reduction_pct is positive while memory actually increased."""
    findings: List[Dict[str, Any]] = []
    results_dir = bench_out / "results"
    if not results_dir.exists():
        return findings

    for mj in sorted(results_dir.rglob("metrics.json")):
        try:
            data = _read_json(mj)
        except Exception:
            continue
        red_pct = data.get("steady_state_memory_reduction_pct")
        ss_gb = data.get("steady_state_alloc_gb")
        base_gb = data.get("baseline_steady_state_alloc_gb")
        if red_pct is not None and ss_gb is not None and base_gb is not None:
            if red_pct > 0 and ss_gb > base_gb:
                findings.append({
                    "check": "memory_claims",
                    "status": "WARN",
                    "detail": (
                        f"{mj.relative_to(bench_out)}: "
                        f"reduction_pct={red_pct} but ss={ss_gb:.2f} > baseline={base_gb:.2f}"
                    ),
                })

    if not findings:
        findings.append({
            "check": "memory_claims",
            "status": "PASS",
            "detail": "No contradictory memory reduction claims",
        })
    return findings


def _check_weights_consistency(
    bench_out: Path,
) -> List[Dict[str, Any]]:
    """weights_gb must be identical across workloads for the same model."""
    findings: List[Dict[str, Any]] = []
    csv_path = bench_out / "benchmarks.csv"
    if not csv_path.exists():
        return findings

    model_weights: Dict[str, Set[str]] = {}
    for r in _read_csv(csv_path):
        mid = r.get("model_id", "")
        wgb = r.get("weights_gb", "")
        if wgb and wgb != "":
            model_weights.setdefault(mid, set()).add(wgb)

    if not model_weights:
        findings.append({
            "check": "weights_consistency",
            "status": "WARN",
            "detail": "No weights_gb values found in benchmarks.csv",
        })
        return findings

    for mid, vals in model_weights.items():
        if len(vals) > 1:
            findings.append({
                "check": "weights_consistency",
                "status": "FAIL",
                "detail": f"{mid}: weights_gb varies across workloads: {sorted(vals)}",
            })
        else:
            findings.append({
                "check": "weights_consistency",
                "status": "PASS",
                "detail": f"{mid}: weights_gb consistent ({vals.pop()})",
            })

    return findings


def _check_cold_alloc_consistency(
    bench_out: Path,
) -> List[Dict[str, Any]]:
    """cold_alloc_gb should be consistent across workloads for the same model."""
    findings: List[Dict[str, Any]] = []
    csv_path = bench_out / "benchmarks.csv"
    if not csv_path.exists():
        return findings

    model_cold: Dict[str, Set[str]] = {}
    for r in _read_csv(csv_path):
        mid = r.get("model_id", "")
        ca = r.get("cold_alloc_gb", "")
        if ca and ca != "":
            model_cold.setdefault(mid, set()).add(ca)

    if not model_cold:
        return findings

    for mid, vals in model_cold.items():
        if len(vals) > 1:
            findings.append({
                "check": "cold_alloc_consistency",
                "status": "WARN",
                "detail": f"{mid}: cold_alloc_gb varies across workloads: {sorted(vals)}",
            })

    if not findings:
        findings.append({
            "check": "cold_alloc_consistency",
            "status": "PASS",
            "detail": "cold_alloc_gb consistent across workloads for all models",
        })
    return findings


def _check_steady_state_advisory(
    bench_out: Path,
) -> List[Dict[str, Any]]:
    """Warn that steady_state_alloc_gb is workload-dependent."""
    findings: List[Dict[str, Any]] = []
    csv_path = bench_out / "benchmarks.csv"
    if not csv_path.exists():
        return findings

    model_ss: Dict[str, Set[str]] = {}
    for r in _read_csv(csv_path):
        mid = r.get("model_id", "")
        ss = r.get("steady_state_alloc_gb", "")
        if ss and ss != "":
            model_ss.setdefault(mid, set()).add(ss)

    for mid, vals in model_ss.items():
        if len(vals) > 1:
            findings.append({
                "check": "steady_state_advisory",
                "status": "WARN",
                "detail": (
                    f"{mid}: steady_state_alloc_gb varies by workload ({sorted(vals)}); "
                    "not suitable for headline memory claims — use weights_gb instead"
                ),
            })

    if not findings:
        findings.append({
            "check": "steady_state_advisory",
            "status": "INFO",
            "detail": "steady_state_alloc_gb is workload-dependent; prefer weights_gb for headline claims",
        })
    return findings


# ── main entry ────────────────────────────────────────────────────────────────

def run_audit(bench_out: Path) -> Dict[str, Any]:
    """Run all audit checks and write audit.txt + audit.json under bench_out/report/.

    Returns the full audit result dict.
    """
    results_dir = bench_out / "results"
    report_dir = bench_out / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    all_findings: List[Dict[str, Any]] = []
    all_findings.extend(_check_promptpack_provenance(results_dir))
    all_findings.extend(_check_baseline_anchoring(bench_out))
    all_findings.extend(_check_environment_parity(results_dir))
    all_findings.extend(_check_prompt_id_parity(results_dir))
    all_findings.extend(_check_error_rates(bench_out))
    all_findings.extend(_check_memory_claims(bench_out))
    all_findings.extend(_check_weights_consistency(bench_out))
    all_findings.extend(_check_cold_alloc_consistency(bench_out))
    all_findings.extend(_check_steady_state_advisory(bench_out))

    n_fail = sum(1 for f in all_findings if f["status"] == "FAIL")
    n_warn = sum(1 for f in all_findings if f["status"] == "WARN")
    n_pass = sum(1 for f in all_findings if f["status"] == "PASS")
    overall = "FAIL" if n_fail > 0 else ("WARN" if n_warn > 0 else "PASS")

    audit_result = {
        "overall": overall,
        "summary": {"pass": n_pass, "warn": n_warn, "fail": n_fail},
        "findings": all_findings,
    }

    # Write JSON
    json_path = report_dir / "audit.json"
    with open(json_path, "w") as f:
        json.dump(audit_result, f, indent=2, default=str)

    # Write human-readable text
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append(f"  BENCH AUDIT REPORT — {overall}")
    lines.append(f"  pass={n_pass}  warn={n_warn}  fail={n_fail}")
    lines.append("=" * 60)
    lines.append("")
    for finding in all_findings:
        tag = f"[{finding['status']:>4}]"
        check = finding.get("check", "")
        detail = finding.get("detail", "")
        lines.append(f"{tag} {check}: {detail}")
    lines.append("")
    lines.append("=" * 60)
    txt = "\n".join(lines)

    txt_path = report_dir / "audit.txt"
    txt_path.write_text(txt)

    _log.info("audit complete: %s — %s", overall, txt_path)
    return audit_result
