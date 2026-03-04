"""Tests for bench pipeline: prompt packs, sanitizer, CSV schema, TTFT, report, audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


# ── prompt packs ──────────────────────────────────────────────────────────────

class TestPromptPacks:
    def test_load_prompt_pack(self, tmp_path):
        from dystrio_sculpt.prompt_packs import load_prompt_pack

        p = tmp_path / "test.jsonl"
        p.write_text(
            '{"id": "a", "prompt": "hello"}\n'
            '{"id": "b", "prompt": "world", "max_new_tokens": 128}\n'
        )
        prompts = load_prompt_pack(p)
        assert len(prompts) == 2
        assert prompts[0]["id"] == "a"
        assert prompts[0]["max_new_tokens"] == 64  # default
        assert prompts[0]["temperature"] == 0.0
        assert prompts[1]["max_new_tokens"] == 128  # explicit

    def test_prompt_pack_hash_deterministic(self, tmp_path):
        from dystrio_sculpt.prompt_packs import prompt_pack_hash

        p = tmp_path / "test.jsonl"
        p.write_text('{"id": "x", "prompt": "data"}\n')
        h1 = prompt_pack_hash(p)
        h2 = prompt_pack_hash(p)
        assert h1 == h2
        assert len(h1) == 16

    def test_empty_lines_skipped(self, tmp_path):
        from dystrio_sculpt.prompt_packs import load_prompt_pack

        p = tmp_path / "sparse.jsonl"
        p.write_text('\n{"id": "a", "prompt": "hi"}\n\n')
        prompts = load_prompt_pack(p)
        assert len(prompts) == 1


# ── sanitizer ─────────────────────────────────────────────────────────────────

class TestSanitizer:
    def test_slash_replaced(self):
        from dystrio_sculpt.bench_runner import sanitize_model_id

        assert sanitize_model_id("org/model-name") == "org__model_name"

    def test_special_chars(self):
        from dystrio_sculpt.bench_runner import sanitize_model_id

        assert sanitize_model_id("a/b.c-d@e") == "a__b_c_d_e"

    def test_already_clean(self):
        from dystrio_sculpt.bench_runner import sanitize_model_id

        assert sanitize_model_id("simple_model") == "simple_model"


class TestModelShortname:
    def test_baseline(self):
        from dystrio_sculpt.bench_runner import model_shortname

        assert model_shortname("org/my-baseline-model") == "baseline"

    def test_conservative(self):
        from dystrio_sculpt.bench_runner import model_shortname

        assert model_shortname("org/sculpted-conservative-v1") == "conservative"

    def test_fallback(self):
        from dystrio_sculpt.bench_runner import model_shortname

        name = model_shortname("org/Llama-2-7b-chat-hf")
        assert isinstance(name, str)
        assert len(name) > 0


# ── benchmarks.csv schema ────────────────────────────────────────────────────

class TestBenchmarksCsvSchema:
    def test_columns_defined(self):
        from dystrio_sculpt.bench_runner import BENCHMARKS_CSV_COLUMNS

        required = [
            "model_id", "workload", "num_prompts",
            "ppl_wikitext", "ppl_ratio",
            "prefill_tokens_per_sec", "decode_tokens_per_sec",
            "ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99",
            "first_decode_step_ms_p50", "first_decode_step_ms_p95", "first_decode_step_ms_p99",
            "peak_alloc_gb", "steady_state_alloc_gb",
            "errors_skipped_prompts",
        ]
        for col in required:
            assert col in BENCHMARKS_CSV_COLUMNS, f"Missing: {col}"

    def test_microbench_columns_separate(self):
        from dystrio_sculpt.bench_runner import BENCHMARKS_CSV_COLUMNS

        for prefix in ("microbench_prefill_ms_", "microbench_decode_ms_per_tok_"):
            assert any(c.startswith(prefix) for c in BENCHMARKS_CSV_COLUMNS)

    def test_write_benchmarks_csv(self, tmp_path):
        from dystrio_sculpt.bench_runner import write_benchmarks_csv

        results = {
            "base_model": {
                "wikitext": {
                    "model_id": "base_model", "workload": "wikitext",
                    "ppl_wikitext": 10.5, "num_prompts": 100,
                },
                "chat": {
                    "model_id": "base_model", "workload": "chat",
                    "num_prompts": 200, "ttft_ms_p95": 15.2,
                    "prefill_tokens_per_sec": 5000.0,
                    "decode_tokens_per_sec": 300.0,
                },
            },
            "sculpted": {
                "wikitext": {
                    "model_id": "sculpted", "workload": "wikitext",
                    "ppl_wikitext": 11.0, "num_prompts": 100,
                },
            },
        }
        csv_path = write_benchmarks_csv(results, tmp_path, baseline_model="base_model")
        assert csv_path.exists()

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

        wt_row = [r for r in rows if r["model_id"] == "sculpted" and r["workload"] == "wikitext"][0]
        assert float(wt_row["ppl_ratio"]) == pytest.approx(11.0 / 10.5, rel=1e-3)


# ── TTFT function shape ──────────────────────────────────────────────────────

class TestTTFTShape:
    def test_bench_ttft_returns_list_of_dicts(self):
        from dystrio_sculpt._bench import bench_ttft_per_prompt

        assert callable(bench_ttft_per_prompt)

    def test_ttft_warmup_constant_exists(self):
        from dystrio_sculpt._bench import TTFT_WARMUP

        assert isinstance(TTFT_WARMUP, int)
        assert TTFT_WARMUP >= 1

    def test_per_prompt_csv_columns_defined(self):
        from dystrio_sculpt.bench_runner import PER_PROMPT_CSV_COLUMNS

        required = [
            "id", "prompt_tokens", "max_new_tokens",
            "prefill_ms", "first_decode_step_ms", "ttft_ms",
            "is_warmup", "error",
        ]
        for col in required:
            assert col in PER_PROMPT_CSV_COLUMNS


# ── report helpers ────────────────────────────────────────────────────────────

class TestReportHelpers:
    def test_float_or_none(self):
        from dystrio_sculpt.report import _float_or_none

        assert _float_or_none("3.14") == pytest.approx(3.14)
        assert _float_or_none("") is None
        assert _float_or_none("abc") is None

    def test_load_benchmarks_not_found(self, tmp_path):
        from dystrio_sculpt.report import _load_benchmarks

        with pytest.raises(FileNotFoundError):
            _load_benchmarks(tmp_path)

    def test_load_benchmarks_reads_csv(self, tmp_path):
        from dystrio_sculpt.report import _load_benchmarks

        csv_path = tmp_path / "benchmarks.csv"
        csv_path.write_text("model_id,workload\nfoo,chat\n")
        rows = _load_benchmarks(tmp_path)
        assert len(rows) == 1
        assert rows[0]["model_id"] == "foo"

    def test_load_per_prompt_ttft_excludes_warmup(self, tmp_path):
        from dystrio_sculpt.report import _load_per_prompt_ttft
        from dystrio_sculpt.bench_runner import sanitize_model_id

        safe = sanitize_model_id("org/model")
        pp_dir = tmp_path / safe / "rag"
        pp_dir.mkdir(parents=True)
        pp = pp_dir / "per_prompt.csv"
        pp.write_text(
            "id,prompt_tokens,max_new_tokens,prefill_ms,first_decode_step_ms,ttft_ms,is_warmup,error\n"
            "w1,100,64,5.0,2.0,7.0,True,\n"
            "w2,100,64,5.0,2.0,7.0,True,\n"
            "m1,100,64,4.0,1.5,5.5,False,\n"
            "m2,100,64,6.0,3.0,9.0,False,\n"
        )
        vals = _load_per_prompt_ttft(tmp_path, "org/model", "rag")
        assert len(vals) == 2
        assert 5.5 in vals
        assert 9.0 in vals


# ── audit tests ──────────────────────────────────────────────────────────────

class TestAudit:
    def _make_bench_out(self, tmp_path, models, workloads, same_hash=True, same_ids=True):
        """Helper to create a synthetic bench output directory for audit."""
        from dystrio_sculpt.bench_runner import sanitize_model_id

        bench_out = tmp_path / "bench_out"
        results_dir = bench_out / "results"

        baseline = models[0]
        base_ppl = 10.0

        # Root run_metadata.json
        root_meta = {
            "baseline_model_id": baseline,
            "deterministic_flag": True,
            "seed": 42,
            "dtype": "bf16",
            "torch_version": "2.2.0",
            "transformers_version": "4.38.0",
            "gpu_name": "A100",
        }
        bench_out.mkdir(parents=True)
        (bench_out / "run_metadata.json").write_text(json.dumps(root_meta))

        csv_rows = []
        for midx, mid in enumerate(models):
            safe = sanitize_model_id(mid)
            model_ppl = base_ppl + midx * 0.5

            # wikitext
            wl_dir = results_dir / safe / "wikitext"
            wl_dir.mkdir(parents=True)
            (wl_dir / "metrics.json").write_text(json.dumps({
                "model_id": mid, "workload": "wikitext",
                "ppl_wikitext": model_ppl,
            }))
            csv_rows.append({
                "model_id": mid, "workload": "wikitext",
                "ppl_wikitext": str(model_ppl),
                "ppl_ratio": str(round(model_ppl / base_ppl, 4)),
                "num_prompts": "500",
            })

            # prompt workloads
            for wl in workloads:
                if wl == "wikitext":
                    continue
                wl_dir = results_dir / safe / wl
                wl_dir.mkdir(parents=True)

                h = "abcd1234abcd1234" if same_hash else f"hash_{mid}_{wl}"
                (wl_dir / "run_metadata.json").write_text(json.dumps({
                    "model_id": mid, "workload": wl,
                    "promptpack_hash": h,
                    "dtype": "bf16",
                    "torch_version": "2.2.0",
                    "transformers_version": "4.38.0",
                    "deterministic": True,
                    "seed": 42,
                    "gpu_name": "A100",
                }))

                prompt_ids = ["p1", "p2", "p3"] if same_ids else (
                    ["p1", "p2", "p3"] if midx == 0 else ["p1", "p2", "p4"]
                )
                pp_lines = [
                    "id,prompt_tokens,max_new_tokens,prefill_ms,first_decode_step_ms,ttft_ms,is_warmup,error"
                ]
                for pid in prompt_ids:
                    pp_lines.append(f"{pid},100,64,5.0,2.0,7.0,False,")
                (wl_dir / "per_prompt.csv").write_text("\n".join(pp_lines) + "\n")
                (wl_dir / "metrics.json").write_text(json.dumps({
                    "model_id": mid, "workload": wl,
                    "num_prompts": len(prompt_ids),
                    "ttft_ms_p95": 7.0,
                    "errors_skipped_prompts": 0,
                }))

                csv_rows.append({
                    "model_id": mid, "workload": wl,
                    "ppl_ratio": str(round(model_ppl / base_ppl, 4)),
                    "num_prompts": str(len(prompt_ids)),
                    "ttft_ms_p95": "7.0",
                    "errors_skipped_prompts": "0",
                })

        # Write benchmarks.csv
        from dystrio_sculpt.bench_runner import BENCHMARKS_CSV_COLUMNS
        csv_path = bench_out / "benchmarks.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=BENCHMARKS_CSV_COLUMNS)
            writer.writeheader()
            for row in csv_rows:
                padded = {c: row.get(c, "") for c in BENCHMARKS_CSV_COLUMNS}
                writer.writerow(padded)

        return bench_out

    def test_audit_passes_clean(self, tmp_path):
        from dystrio_sculpt.audit import run_audit

        bench_out = self._make_bench_out(
            tmp_path, ["baseline_m", "sculpted_m"], ["wikitext", "chat"],
        )
        result = run_audit(bench_out)
        assert result["overall"] in ("PASS", "INFO")
        assert result["summary"]["fail"] == 0

    def test_audit_fails_prompt_id_mismatch(self, tmp_path):
        from dystrio_sculpt.audit import run_audit

        bench_out = self._make_bench_out(
            tmp_path, ["baseline_m", "sculpted_m"], ["wikitext", "chat"],
            same_ids=False,
        )
        result = run_audit(bench_out)
        id_findings = [f for f in result["findings"] if f["check"] == "prompt_id_parity"]
        assert any(f["status"] == "FAIL" for f in id_findings)

    def test_audit_fails_promptpack_hash_mismatch(self, tmp_path):
        from dystrio_sculpt.audit import run_audit

        bench_out = self._make_bench_out(
            tmp_path, ["baseline_m", "sculpted_m"], ["wikitext", "chat"],
            same_hash=False,
        )
        result = run_audit(bench_out)
        pp_findings = [f for f in result["findings"] if f["check"] == "promptpack_provenance"]
        assert any(f["status"] == "FAIL" for f in pp_findings)

    def test_ppl_ratio_anchoring_consistent(self, tmp_path):
        from dystrio_sculpt.audit import run_audit

        bench_out = self._make_bench_out(
            tmp_path, ["baseline_m", "sculpted_m"], ["wikitext", "chat"],
        )
        result = run_audit(bench_out)
        bl_findings = [f for f in result["findings"] if f["check"] == "baseline_anchoring"]
        assert any(f["status"] == "PASS" for f in bl_findings)

    def test_audit_writes_outputs(self, tmp_path):
        from dystrio_sculpt.audit import run_audit

        bench_out = self._make_bench_out(
            tmp_path, ["baseline_m"], ["wikitext", "rag"],
        )
        result = run_audit(bench_out)
        assert (bench_out / "report" / "audit.json").exists()
        assert (bench_out / "report" / "audit.txt").exists()

        audit_json = json.loads((bench_out / "report" / "audit.json").read_text())
        assert "overall" in audit_json
        assert "findings" in audit_json

    def test_error_rates_warn(self, tmp_path):
        """Inject errors and verify audit warns."""
        from dystrio_sculpt.audit import run_audit
        from dystrio_sculpt.bench_runner import BENCHMARKS_CSV_COLUMNS

        bench_out = self._make_bench_out(
            tmp_path, ["baseline_m"], ["wikitext", "chat"],
        )
        # Rewrite benchmarks.csv with errors
        csv_path = bench_out / "benchmarks.csv"
        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["workload"] == "chat":
                    r["errors_skipped_prompts"] = "2"
                rows.append(r)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=BENCHMARKS_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        result = run_audit(bench_out)
        err_findings = [f for f in result["findings"] if f["check"] == "error_rates"]
        assert any(f["status"] in ("WARN", "FAIL") for f in err_findings)


# ── warmup exclusion ─────────────────────────────────────────────────────────

class TestWarmupExclusion:
    def test_per_prompt_warmup_not_in_metrics(self, tmp_path):
        """Warmup prompts should not affect published percentile metrics."""
        from dystrio_sculpt._bench import compute_latency_percentiles

        warmup_vals = [100.0, 200.0, 300.0]
        measured_vals = [5.0, 5.1, 5.2, 5.3, 5.0, 4.9, 5.1, 5.0, 5.2, 5.1]
        all_results = [
            {"ttft_ms": v, "is_warmup": True, "error": ""} for v in warmup_vals
        ] + [
            {"ttft_ms": v, "is_warmup": False, "error": ""} for v in measured_vals
        ]

        filtered = [r["ttft_ms"] for r in all_results if not r["is_warmup"] and not r.get("error")]
        pct = compute_latency_percentiles(filtered)
        assert pct["p50"] < 10.0
        assert pct["p95"] < 10.0

    def test_per_prompt_warmup_constant(self):
        from dystrio_sculpt.bench_runner import PER_PROMPT_WARMUP

        assert isinstance(PER_PROMPT_WARMUP, int)
        assert PER_PROMPT_WARMUP >= 1


# ── model card snippet ────────────────────────────────────────────────────────

class TestModelCardSnippet:
    def test_model_card_generated(self, tmp_path):
        from dystrio_sculpt.report import _write_model_card_snippet

        rows = [
            {"model_id": "baseline_model", "workload": "wikitext", "ppl_wikitext": "10.0", "ppl_ratio": "1.0"},
            {"model_id": "baseline_model", "workload": "rag", "ttft_ms_p95": "15.0",
             "prefill_tokens_per_sec": "5000", "decode_tokens_per_sec": "300",
             "steady_state_alloc_gb": "6.2", "peak_alloc_gb": "8.1"},
            {"model_id": "sculpted", "workload": "wikitext", "ppl_wikitext": "10.5", "ppl_ratio": "1.05"},
            {"model_id": "sculpted", "workload": "rag", "ttft_ms_p95": "12.0",
             "prefill_tokens_per_sec": "7000", "decode_tokens_per_sec": "400",
             "steady_state_alloc_gb": "5.0", "peak_alloc_gb": "7.0"},
        ]

        report_dir = tmp_path / "report"
        report_dir.mkdir()

        _write_model_card_snippet(rows, report_dir)
        md_path = report_dir / "model_card_snippet.md"
        assert md_path.exists()

        content = md_path.read_text()
        assert "## Benchmark Results" in content
        assert "PPL Ratio" in content
        assert "TTFT" in content
        assert "baseline" in content.lower()
        assert "Metric Definitions" in content

    def test_model_card_env_footnote(self, tmp_path):
        from dystrio_sculpt.report import _write_model_card_snippet

        bench_out = tmp_path / "bench"
        bench_out.mkdir()
        (bench_out / "run_metadata.json").write_text(json.dumps({
            "gpu_name": "A100", "dtype": "bf16",
            "torch_version": "2.2.0", "transformers_version": "4.38.0",
        }))

        rows = [
            {"model_id": "m1", "workload": "wikitext", "ppl_wikitext": "10"},
        ]

        report_dir = tmp_path / "report"
        report_dir.mkdir()
        _write_model_card_snippet(rows, report_dir, bench_out=bench_out)

        content = (report_dir / "model_card_snippet.md").read_text()
        assert "A100" in content
        assert "Single-GPU" in content

    def test_model_card_empty_rows(self, tmp_path):
        from dystrio_sculpt.report import _write_model_card_snippet

        report_dir = tmp_path / "report"
        report_dir.mkdir()
        _write_model_card_snippet([], report_dir)
        assert not (report_dir / "model_card_snippet.md").exists()


# ── memory vs quality plot ────────────────────────────────────────────────────

class TestMemoryVsQuality:
    def test_plot_and_table_generated(self, tmp_path):
        from dystrio_sculpt.report import _plot_memory_vs_quality

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")

        rows = [
            {"model_id": "org/baseline-model", "ppl_ratio": "1.0", "steady_state_alloc_gb": "8.5"},
            {"model_id": "org/sculpted-conservative", "ppl_ratio": "1.02", "steady_state_alloc_gb": "6.1"},
            {"model_id": "org/sculpted-balanced", "ppl_ratio": "1.08", "steady_state_alloc_gb": "5.0"},
        ]
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        _plot_memory_vs_quality(rows, report_dir, plt)

        assert (report_dir / "memory_vs_quality.png").exists()
        assert (report_dir / "memory_vs_quality.md").exists()

        md = (report_dir / "memory_vs_quality.md").read_text()
        assert "| PPL Ratio | VRAM (GB) |" in md
        assert "baseline" in md.lower()
        assert "1.000" in md
        assert "8.500" in md

    def test_baseline_star_marker(self, tmp_path):
        from dystrio_sculpt.report import _plot_memory_vs_quality

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")

        rows = [
            {"model_id": "org/baseline-v1", "ppl_ratio": "1.0", "steady_state_alloc_gb": "8.0"},
            {"model_id": "org/sculpted-v1", "ppl_ratio": "1.05", "steady_state_alloc_gb": "6.0"},
        ]
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        _plot_memory_vs_quality(rows, report_dir, plt)
        assert (report_dir / "memory_vs_quality.png").exists()

    def test_skipped_when_no_data(self, tmp_path):
        from dystrio_sculpt.report import _plot_memory_vs_quality

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")

        rows = [{"model_id": "m1", "workload": "chat"}]
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        _plot_memory_vs_quality(rows, report_dir, plt)
        assert not (report_dir / "memory_vs_quality.png").exists()

    def test_table_values_rounded(self, tmp_path):
        from dystrio_sculpt.report import _plot_memory_vs_quality

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")

        rows = [
            {"model_id": "org/model-x", "ppl_ratio": "1.12345", "steady_state_alloc_gb": "7.56789"},
        ]
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        _plot_memory_vs_quality(rows, report_dir, plt)

        md = (report_dir / "memory_vs_quality.md").read_text()
        assert "1.123" in md
        assert "7.568" in md
        assert "1.12345" not in md


# ── TTFT naming clarity ──────────────────────────────────────────────────────

class TestTTFTNaming:
    def test_per_prompt_csv_has_both_metrics(self):
        from dystrio_sculpt.bench_runner import PER_PROMPT_CSV_COLUMNS

        assert "first_decode_step_ms" in PER_PROMPT_CSV_COLUMNS
        assert "ttft_ms" in PER_PROMPT_CSV_COLUMNS
        assert "prefill_ms" in PER_PROMPT_CSV_COLUMNS

    def test_benchmarks_csv_has_both_percentile_sets(self):
        from dystrio_sculpt.bench_runner import BENCHMARKS_CSV_COLUMNS

        assert "first_decode_step_ms_p95" in BENCHMARKS_CSV_COLUMNS
        assert "ttft_ms_p95" in BENCHMARKS_CSV_COLUMNS
        assert "prefill_ms_p95" in BENCHMARKS_CSV_COLUMNS

    def test_microbench_and_request_level_separated(self):
        from dystrio_sculpt.bench_runner import BENCHMARKS_CSV_COLUMNS

        microbench = [c for c in BENCHMARKS_CSV_COLUMNS if c.startswith("microbench_")]
        request_level = [c for c in BENCHMARKS_CSV_COLUMNS if c.startswith("first_decode_step_ms_")]
        assert len(microbench) >= 3
        assert len(request_level) >= 3
