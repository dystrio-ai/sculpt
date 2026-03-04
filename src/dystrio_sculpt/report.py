"""Benchmark report: aggregate benchmarks.csv, generate plots and model card."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .bench_runner import model_shortname

_log = logging.getLogger(__name__)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _float_or_none(val: str) -> Optional[float]:
    if val == "" or val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _load_benchmarks(results_dir: Path) -> List[Dict[str, Any]]:
    for candidate in [results_dir.parent / "benchmarks.csv", results_dir / "benchmarks.csv"]:
        if candidate.exists():
            rows = _read_csv(candidate)
            _log.info("loaded %d rows from %s", len(rows), candidate)
            return rows
    raise FileNotFoundError("benchmarks.csv not found")


def _load_per_prompt_ttft(results_dir: Path, model_id: str, workload: str) -> List[float]:
    from .bench_runner import sanitize_model_id
    safe = sanitize_model_id(model_id)
    candidates = [
        results_dir / safe / workload / "per_prompt.csv",
        results_dir.parent / "results" / safe / workload / "per_prompt.csv",
    ]
    for pp in candidates:
        if pp.exists():
            vals: List[float] = []
            for row in _read_csv(pp):
                warmup = row.get("is_warmup", "")
                if warmup == "True":
                    continue
                v = _float_or_none(row.get("ttft_ms", ""))
                if v is not None:
                    vals.append(v)
            return vals
    return []


# ── Plotting ──────────────────────────────────────────────────────────────────

def generate_report(
    results_dir: Path,
    report_dir: Path,
    bench_out: Optional[Path] = None,
) -> None:
    """Read benchmark results and generate plots + model card snippet."""
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_benchmarks(results_dir)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _log.error("matplotlib not installed — cannot generate plots")
        return

    _plot_frontier_rag_ttft(rows, report_dir, plt)
    _plot_frontier_chat_decode(rows, report_dir, plt)
    _plot_p95_latency_bars(rows, report_dir, plt)
    _plot_throughput_bars(rows, report_dir, plt)
    _plot_rag_ttft_cdf(rows, results_dir, report_dir, plt)
    _plot_memory_vs_quality(rows, report_dir, plt)
    _plot_memory_vs_quality_weights(rows, report_dir, plt)
    _plot_memory_vs_quality_cold_alloc(rows, report_dir, plt)

    _write_model_card_snippet(rows, report_dir, bench_out)

    _log.info("[report] plots + model card written to %s", report_dir)


def _labels_for(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    return {r["model_id"]: model_shortname(r["model_id"]) for r in rows}


# ── frontier: rag TTFT p95 vs ppl_ratio ───────────────────────────────────────

def _plot_frontier_rag_ttft(
    rows: List[Dict[str, Any]], report_dir: Path, plt,
) -> None:
    pts = [(r, _float_or_none(r.get("ppl_ratio", "")), _float_or_none(r.get("ttft_ms_p95", "")))
           for r in rows if r.get("workload") == "rag"]
    pts = [(r, x, y) for r, x, y in pts if x is not None and y is not None]
    if not pts:
        _log.warning("no rag rows with ppl_ratio+ttft — skipping frontier plot")
        return
    labels = _labels_for([r for r, _, _ in pts])

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [x for _, x, _ in pts]
    ys = [y for _, _, y in pts]
    ax.scatter(xs, ys, s=60, zorder=5)
    for r, x, y in pts:
        ax.annotate(labels[r["model_id"]], (x, y), textcoords="offset points",
                     xytext=(6, 6), fontsize=8)
    ax.set_xlabel("PPL Ratio (vs baseline)")
    ax.set_ylabel("TTFT incl. prefill p95 (ms)")
    ax.set_title("Frontier: RAG TTFT p95 vs Quality")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "frontier_rag_ttft_p95_vs_pplratio.png", dpi=150)
    plt.close(fig)


# ── frontier: chat decode p95 vs ppl_ratio ────────────────────────────────────

def _plot_frontier_chat_decode(
    rows: List[Dict[str, Any]], report_dir: Path, plt,
) -> None:
    pts = [(r, _float_or_none(r.get("ppl_ratio", "")),
            _float_or_none(r.get("decode_ms_per_tok_p95", "")))
           for r in rows if r.get("workload") == "chat"]
    pts = [(r, x, y) for r, x, y in pts if x is not None and y is not None]
    if not pts:
        _log.warning("no chat rows with ppl_ratio+decode — skipping frontier plot")
        return
    labels = _labels_for([r for r, _, _ in pts])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([x for _, x, _ in pts], [y for _, _, y in pts], s=60, zorder=5)
    for r, x, y in pts:
        ax.annotate(labels[r["model_id"]], (x, y), textcoords="offset points",
                     xytext=(6, 6), fontsize=8)
    ax.set_xlabel("PPL Ratio (vs baseline)")
    ax.set_ylabel("Chat Decode ms/tok p95")
    ax.set_title("Frontier: Chat Decode p95 vs Quality")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "frontier_chat_decode_p95_vs_pplratio.png", dpi=150)
    plt.close(fig)


# ── bar: p95 latency by workload ──────────────────────────────────────────────

def _plot_p95_latency_bars(
    rows: List[Dict[str, Any]], report_dir: Path, plt,
) -> None:
    model_ids = list(dict.fromkeys(r["model_id"] for r in rows))
    labels = {m: model_shortname(m) for m in model_ids}
    workloads = [
        ("rag", "ttft_ms_p95", "RAG TTFT p95"),
        ("chat", "first_decode_step_ms_p95", "Chat 1st-decode p95"),
        ("code", "first_decode_step_ms_p95", "Code 1st-decode p95"),
    ]

    groups: List[str] = []
    data: Dict[str, List[Optional[float]]] = {m: [] for m in model_ids}
    for wl, key, lbl in workloads:
        groups.append(lbl)
        for m in model_ids:
            val = None
            for r in rows:
                if r["model_id"] == m and r["workload"] == wl:
                    val = _float_or_none(r.get(key, ""))
                    break
            data[m].append(val)

    if not groups:
        return

    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(groups))
    w = 0.8 / max(1, len(model_ids))
    for i, m in enumerate(model_ids):
        vals = [v if v is not None else 0 for v in data[m]]
        ax.bar(x + i * w, vals, w, label=labels[m])
    ax.set_xticks(x + w * len(model_ids) / 2)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("p95 Latency by Workload")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "p95_latency_by_workload.png", dpi=150)
    plt.close(fig)


# ── bar: throughput by workload ───────────────────────────────────────────────

def _plot_throughput_bars(
    rows: List[Dict[str, Any]], report_dir: Path, plt,
) -> None:
    model_ids = list(dict.fromkeys(r["model_id"] for r in rows))
    labels = {m: model_shortname(m) for m in model_ids}
    workloads = [
        ("rag", "prefill_tokens_per_sec", "RAG Prefill TPS"),
        ("rag", "decode_tokens_per_sec", "RAG Decode TPS"),
        ("chat", "decode_tokens_per_sec", "Chat Decode TPS"),
        ("code", "decode_tokens_per_sec", "Code Decode TPS"),
    ]

    groups: List[str] = []
    data: Dict[str, List[Optional[float]]] = {m: [] for m in model_ids}
    for wl, key, lbl in workloads:
        groups.append(lbl)
        for m in model_ids:
            val = None
            for r in rows:
                if r["model_id"] == m and r["workload"] == wl:
                    val = _float_or_none(r.get(key, ""))
                    break
            data[m].append(val)

    if not groups:
        return

    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(groups))
    w = 0.8 / max(1, len(model_ids))
    for i, m in enumerate(model_ids):
        vals = [v if v is not None else 0 for v in data[m]]
        ax.bar(x + i * w, vals, w, label=labels[m])
    ax.set_xticks(x + w * len(model_ids) / 2)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("Tokens / sec")
    ax.set_title("Throughput by Workload")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "throughput_by_workload.png", dpi=150)
    plt.close(fig)


# ── CDF: rag TTFT distribution ───────────────────────────────────────────────

def _plot_rag_ttft_cdf(
    rows: List[Dict[str, Any]], results_dir: Path, report_dir: Path, plt,
) -> None:
    model_ids = list(dict.fromkeys(r["model_id"] for r in rows if r.get("workload") == "rag"))
    if not model_ids:
        return

    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in model_ids:
        vals = _load_per_prompt_ttft(results_dir, m, "rag")
        if not vals:
            continue
        sorted_v = np.sort(vals)
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.plot(sorted_v, cdf, label=model_shortname(m), linewidth=1.5)

    ax.set_xlabel("TTFT incl. prefill (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("RAG TTFT Distribution (request-level)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "rag_ttft_cdf.png", dpi=150)
    plt.close(fig)


# ── scatter: memory vs quality ─────────────────────────────────────────────────

def _plot_memory_vs_quality(
    rows: List[Dict[str, Any]], report_dir: Path, plt,
) -> None:
    """Scatter plot of steady-state VRAM vs PPL ratio + markdown table."""
    # Deduplicate: one point per model (pick first row with both values)
    seen: Dict[str, tuple] = {}
    for r in rows:
        mid = r.get("model_id", "")
        if mid in seen:
            continue
        ppl = _float_or_none(r.get("ppl_ratio", ""))
        ss = _float_or_none(r.get("steady_state_alloc_gb", ""))
        if ppl is not None and ss is not None:
            seen[mid] = (ppl, ss)

    if not seen:
        _log.warning("no rows with ppl_ratio + steady_state_alloc_gb — skipping memory_vs_quality")
        return

    labels = {mid: model_shortname(mid) for mid in seen}

    fig, ax = plt.subplots(figsize=(8, 5))
    for mid, (ppl, ss) in seen.items():
        short = labels[mid]
        is_baseline = "baseline" in short.lower()
        marker = "*" if is_baseline else "o"
        color = "black" if is_baseline else "#1f77b4"
        ax.scatter(ppl, ss, s=120, marker=marker, color=color, zorder=5)
        ax.annotate(short, (ppl, ss), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)

    ax.set_xlabel("Quality Drift (Perplexity Ratio vs Baseline)")
    ax.set_ylabel("Steady-State VRAM (GB)")
    ax.set_title("Memory vs Quality Tradeoff")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "memory_vs_quality.png", dpi=150)
    plt.close(fig)

    # Markdown table
    md_lines: List[str] = []
    md_lines.append("## Memory vs Quality\n")
    md_lines.append("| Model | PPL Ratio | VRAM (GB) |")
    md_lines.append("|-------|-----------|-----------|")
    for mid in seen:
        ppl, ss = seen[mid]
        md_lines.append(f"| {labels[mid]} | {ppl:.3f} | {ss:.3f} |")
    md_lines.append("")
    (report_dir / "memory_vs_quality.md").write_text("\n".join(md_lines))
    _log.info("[report] memory_vs_quality.png + .md written")


# ── scatter: weights memory vs quality ────────────────────────────────────────

def _plot_memory_vs_quality_weights(
    rows: List[Dict[str, Any]], report_dir: Path, plt,
) -> None:
    """Scatter of weights-only GiB vs PPL ratio (headline VRAM plot)."""
    seen: Dict[str, tuple] = {}
    for r in rows:
        mid = r.get("model_id", "")
        if mid in seen:
            continue
        ppl = _float_or_none(r.get("ppl_ratio", ""))
        wgb = _float_or_none(r.get("weights_gb", ""))
        if ppl is not None and wgb is not None:
            seen[mid] = (ppl, wgb)

    if not seen:
        _log.warning("no rows with ppl_ratio + weights_gb — skipping memory_vs_quality_weights")
        return

    labels = {mid: model_shortname(mid) for mid in seen}
    fig, ax = plt.subplots(figsize=(8, 5))
    for mid, (ppl, wgb) in seen.items():
        short = labels[mid]
        is_baseline = "baseline" in short.lower()
        marker = "*" if is_baseline else "o"
        color = "black" if is_baseline else "#1f77b4"
        ax.scatter(ppl, wgb, s=120, marker=marker, color=color, zorder=5)
        ax.annotate(short, (ppl, wgb), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)

    ax.set_xlabel("Quality Drift (Perplexity Ratio vs Baseline)")
    ax.set_ylabel("Weights Memory (GiB)")
    ax.set_title("Weights Memory vs Quality Drift")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "memory_vs_quality_weights.png", dpi=150)
    plt.close(fig)
    _log.info("[report] memory_vs_quality_weights.png written")


# ── scatter: cold alloc vs quality ────────────────────────────────────────────

def _plot_memory_vs_quality_cold_alloc(
    rows: List[Dict[str, Any]], report_dir: Path, plt,
) -> None:
    """Scatter of post-load allocated VRAM vs PPL ratio."""
    seen: Dict[str, tuple] = {}
    for r in rows:
        mid = r.get("model_id", "")
        if mid in seen:
            continue
        ppl = _float_or_none(r.get("ppl_ratio", ""))
        ca = _float_or_none(r.get("cold_alloc_gb", ""))
        if ppl is not None and ca is not None:
            seen[mid] = (ppl, ca)

    if not seen:
        _log.warning("no rows with ppl_ratio + cold_alloc_gb — skipping memory_vs_quality_cold_alloc")
        return

    labels = {mid: model_shortname(mid) for mid in seen}
    fig, ax = plt.subplots(figsize=(8, 5))
    for mid, (ppl, ca) in seen.items():
        short = labels[mid]
        is_baseline = "baseline" in short.lower()
        marker = "*" if is_baseline else "o"
        color = "black" if is_baseline else "#2ca02c"
        ax.scatter(ppl, ca, s=120, marker=marker, color=color, zorder=5)
        ax.annotate(short, (ppl, ca), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)

    ax.set_xlabel("Quality Drift (Perplexity Ratio vs Baseline)")
    ax.set_ylabel("Post-load Allocated VRAM (GiB)")
    ax.set_title("Post-load Allocated VRAM vs Quality Drift")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "memory_vs_quality_cold_alloc.png", dpi=150)
    plt.close(fig)
    _log.info("[report] memory_vs_quality_cold_alloc.png written")


# ── Model card snippet ────────────────────────────────────────────────────────

def _write_model_card_snippet(
    rows: List[Dict[str, Any]],
    report_dir: Path,
    bench_out: Optional[Path] = None,
) -> None:
    """Write a copy/paste model-card markdown table + environment footnote."""
    model_ids = list(dict.fromkeys(r["model_id"] for r in rows))
    if not model_ids:
        return

    # Gather one row per model for the "hero" workloads
    def _get(mid: str, wl: str, key: str) -> str:
        for r in rows:
            if r["model_id"] == mid and r.get("workload") == wl:
                v = r.get(key, "")
                if v != "":
                    return str(v)
        return "—"

    lines: List[str] = []
    lines.append("## Benchmark Results\n")
    lines.append(
        "| Model | PPL | PPL Ratio | RAG TTFT p95 (ms) | Chat Decode p95 (ms/tok) "
        "| Prefill TPS | Decode TPS | Weights (GiB) | Post-load (GiB) "
        "| End-of-bench (GiB) | Peak (GiB) |"
    )
    lines.append(
        "|-------|-----|-----------|-------------------|-------------------------|"
        "------------|------------|---------------|----------------"
        "|--------------------|------------|"
    )
    for mid in model_ids:
        short = model_shortname(mid)
        ppl = _get(mid, "wikitext", "ppl_wikitext")
        ratio = _get(mid, "wikitext", "ppl_ratio")
        if ratio == "—":
            for r in rows:
                if r["model_id"] == mid and r.get("ppl_ratio", "") != "":
                    ratio = r["ppl_ratio"]
                    break
        rag_ttft = _get(mid, "rag", "ttft_ms_p95")
        chat_decode = _get(mid, "chat", "first_decode_step_ms_p95")
        if chat_decode == "—":
            chat_decode = _get(mid, "chat", "decode_ms_per_tok_p95")
        pf_tps = _get(mid, "rag", "prefill_tokens_per_sec")
        dc_tps = _get(mid, "chat", "decode_tokens_per_sec")

        def _get_any(mid: str, key: str) -> str:
            for wl_pref in ("rag", "chat", "wikitext", "code"):
                v = _get(mid, wl_pref, key)
                if v != "\u2014":
                    return v
            return "\u2014"

        wts_gb = _get_any(mid, "weights_gb")
        cold_gb = _get_any(mid, "cold_alloc_gb")
        ss_gb = _get(mid, "rag", "steady_state_alloc_gb")
        if ss_gb == "\u2014":
            ss_gb = _get(mid, "chat", "steady_state_alloc_gb")
        peak_gb = _get(mid, "rag", "peak_alloc_gb")
        if peak_gb == "\u2014":
            peak_gb = _get(mid, "chat", "peak_alloc_gb")

        lines.append(
            f"| {short} | {ppl} | {ratio} | {rag_ttft} | {chat_decode} "
            f"| {pf_tps} | {dc_tps} | {wts_gb} | {cold_gb} "
            f"| {ss_gb} | {peak_gb} |"
        )

    # Environment footnote
    lines.append("")
    lines.append("### Benchmark Environment\n")

    env_info = _load_env_footnote(bench_out)
    for k, v in env_info.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("- Single-GPU, Hugging Face Transformers, no custom kernels.")
    lines.append("")

    # Metric definitions
    lines.append("### Metric Definitions\n")
    lines.append(
        "- **TTFT incl. prefill**: Wall time from prompt submission to first generated token "
        "(prefill forward + first decode step). Per-prompt request-level measurement."
    )
    lines.append(
        "- **First decode step**: Wall time of the first decode forward call only (post-prefill). "
        "Per-prompt request-level measurement."
    )
    lines.append(
        "- **Prefill/Decode TPS**: Throughput from batched microbenchmark iterations "
        "(not request-level; used for throughput comparison only)."
    )
    lines.append(
        "- **Weights (GiB)**: Model parameter memory only "
        "(sum of numel * element_size for all parameters). "
        "Deterministic and runtime-independent."
    )
    lines.append(
        "- **Post-load (GiB)**: `torch.cuda.memory_allocated()` immediately after "
        "`model.eval()` + `torch.cuda.empty_cache()`. "
        "Captures weights + framework overhead before any inference."
    )
    lines.append(
        "- **End-of-bench (GiB)**: `torch.cuda.memory_allocated()` at end of "
        "benchmark workload. Includes KV-cache and activations still held."
    )
    lines.append(
        "- **Peak (GiB)**: `torch.cuda.max_memory_allocated()` during benchmark. "
        "High-water mark for planning GPU headroom."
    )
    lines.append("")

    md = "\n".join(lines)
    out_path = report_dir / "model_card_snippet.md"
    out_path.write_text(md)
    _log.info("[report] model_card_snippet.md written")


def _load_env_footnote(bench_out: Optional[Path]) -> Dict[str, str]:
    """Load environment info from run_metadata.json for footnote."""
    info: Dict[str, str] = {}
    if bench_out is None:
        return info
    meta_path = bench_out / "run_metadata.json"
    if not meta_path.exists():
        return info
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return info
    mapping = [
        ("GPU", "gpu_name"),
        ("dtype", "dtype"),
        ("Torch", "torch_version"),
        ("Transformers", "transformers_version"),
        ("Deterministic", "deterministic_flag"),
        ("Seed", "seed"),
    ]
    for label, key in mapping:
        v = meta.get(key)
        if v is not None:
            info[label] = str(v)
    return info
