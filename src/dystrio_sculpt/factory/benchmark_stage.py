"""Benchmark stage: wraps bench_runner for factory pipeline.

Runs all configured workloads against compiled tier models and the baseline.
Auto-generates prompt packs if they don't exist.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

_log = logging.getLogger(__name__)

DEFAULT_PROMPTS_DIR = Path("prompts")


def _ensure_prompt_packs(prompts_dir: Path) -> Path:
    """Generate prompt packs if they don't exist yet."""
    chat_files = list(prompts_dir.glob("chat*.jsonl"))
    if chat_files:
        _log.info("prompt packs found in %s", prompts_dir)
        return prompts_dir

    _log.info("generating prompt packs in %s ...", prompts_dir)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-v1", split="validation")
        texts = [t.strip() for t in ds["text"] if len(t.strip()) > 100]

        import json, random
        rng = random.Random(42)

        def _write(path, rows):
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            _log.info("  wrote %d prompts -> %s", len(rows), path)

        # Chat prompts
        templates = [
            "Summarize the following passage in two sentences:\n\n{text}",
            "What are the key facts mentioned in this passage?\n\n{text}",
            "Explain the following in simple terms:\n\n{text}",
            "Rewrite the following passage as a bulleted list:\n\n{text}",
        ]
        chat = []
        for i, t in enumerate(rng.sample(texts, min(200, len(texts)))):
            short = " ".join(t.split()[:80])
            chat.append({"id": f"chat_{i:03d}", "prompt": templates[i % 4].format(text=short),
                         "max_new_tokens": 128, "temperature": 0.0, "top_p": 1.0})
        _write(prompts_dir / "chat.jsonl", chat)

        # RAG prompts
        rag = []
        for i in range(100):
            paras = rng.sample(texts, min(rng.randint(5, 8), len(texts)))
            ctx = "\n\n".join(paras)
            rag.append({"id": f"rag_{i:03d}", "prompt": f"Based on the following context, provide a comprehensive summary of the main topics discussed.\n\nContext:\n{ctx}\n\nSummary:",
                        "max_new_tokens": 256, "temperature": 0.0, "top_p": 1.0})
        _write(prompts_dir / "rag.jsonl", rag)

        # Code prompts
        code_tpls = [
            "Write a Python function that extracts all proper nouns from the following text:\n\n{text}\n\ndef extract_proper_nouns(text: str) -> list[str]:",
            "Write a Python function that counts the word frequency in the following text:\n\n{text}\n\ndef word_frequency(text: str) -> dict[str, int]:",
        ]
        code = []
        for i, t in enumerate(rng.sample(texts, min(100, len(texts)))):
            short = " ".join(t.split()[:60])
            code.append({"id": f"code_{i:03d}", "prompt": code_tpls[i % 2].format(text=short),
                         "max_new_tokens": 256, "temperature": 0.0, "top_p": 1.0})
        _write(prompts_dir / "code.jsonl", code)

        _log.info("prompt packs generated successfully")

    except Exception as exc:
        _log.warning("failed to generate prompt packs: %s", exc)

    return prompts_dir


def run_benchmark_stage(
    model_dirs: List[str],
    baseline_model_id: str,
    outdir: Path,
    *,
    workloads: Optional[List[str]] = None,
    prompts_dir: Optional[Path] = None,
    device: str = "cuda",
    dtype_str: str = "bf16",
    seed: int = 0,
    deterministic: bool = True,
) -> Optional[Path]:
    """Run benchmarks for all tier model dirs + baseline.

    Returns path to the generated benchmarks.csv, or None on failure.
    """
    if workloads is None:
        workloads = ["wikitext", "chat", "rag", "code"]

    if prompts_dir is None:
        prompts_dir = DEFAULT_PROMPTS_DIR

    needs_prompts = any(w in workloads for w in ("chat", "rag", "code"))
    if needs_prompts:
        prompts_dir = _ensure_prompt_packs(prompts_dir)

    models = [baseline_model_id] + model_dirs

    outdir = Path(outdir)
    bench_out = outdir / "bench"
    bench_out.mkdir(parents=True, exist_ok=True)

    _log.info(
        "benchmark stage: %d models x %d workloads (prompts_dir=%s)",
        len(models), len(workloads), prompts_dir,
    )

    try:
        from ..bench_runner import run_bench

        csv_path = run_bench(
            models=models,
            workloads=workloads,
            prompts_dir=prompts_dir,
            outdir=bench_out,
            device=device,
            dtype_str=dtype_str,
            seed=seed,
            deterministic=deterministic,
            baseline_model=baseline_model_id,
        )
        _log.info("benchmarks saved to %s", csv_path)
        return csv_path

    except Exception as exc:
        _log.error("benchmark stage failed: %s", exc, exc_info=True)
        return None
