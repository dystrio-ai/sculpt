"""CLI entrypoint: `pnn` — powered by Typer."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="pnn",
    help="PNN Compiler — demand-aware FFN block compression + repair for Qwen2-style models.",
    add_completion=False,
    rich_markup_mode="rich",
)

# ── Shared options ───────────────────────────────────────────────────────────

ConfigOpt = Annotated[
    Optional[Path],
    typer.Option("--config", "-c", help="Path to YAML config file."),
]
OverrideOpt = Annotated[
    Optional[list[str]],
    typer.Option("--set", "-s", help="Override config values: key=value"),
]


def _cfg(config: Path | None, overrides: list[str] | None):
    from pnn_compiler.config import load_config
    return load_config(config, overrides or [])


def _setup(cfg):
    """Seed + TF32 init."""
    import numpy as np
    import torch
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _load(cfg):
    """Resolve dtype and load model + tokenizer."""
    from pnn_compiler.config import resolve_dtype
    from pnn_compiler.model import load_model_and_tokenizer
    dtype = resolve_dtype(cfg.dtype)
    return load_model_and_tokenizer(cfg.model_id, cfg.device, dtype), dtype


# ── Commands ─────────────────────────────────────────────────────────────────

@app.command()
def calibrate(
    config: ConfigOpt = None,
    set_: OverrideOpt = None,
) -> None:
    """Collect per-neuron SwiGLU activation importance scores."""
    cfg = _cfg(config, set_)
    _setup(cfg)
    from pnn_compiler.data import load_text_sets
    from pnn_compiler.calibrate import collect_ffn_importance_swiglu

    (model, tok), dtype = _load(cfg)
    texts = load_text_sets(cfg.n_texts_cal, cfg.n_texts_train, cfg.n_texts_eval)
    for li in cfg.layers:
        imp = collect_ffn_importance_swiglu(model, tok, li, texts["cal"], cfg.max_len, cfg.device)
        print(f"Layer {li}: scored {imp.shape[0]} neurons")


@app.command()
def compile(
    config: ConfigOpt = None,
    set_: OverrideOpt = None,
) -> None:
    """Calibrate + select blocks + physically slice FFN weights."""
    cfg = _cfg(config, set_)
    _setup(cfg)
    from pnn_compiler.data import load_text_sets
    from pnn_compiler.calibrate import collect_ffn_importance_swiglu
    from pnn_compiler.compile import select_blocks, compress_mlp_layer_swiglu_inplace

    (model, tok), dtype = _load(cfg)
    texts = load_text_sets(cfg.n_texts_cal, cfg.n_texts_train, cfg.n_texts_eval)
    for li in cfg.layers:
        imp = collect_ffn_importance_swiglu(model, tok, li, texts["cal"], cfg.max_len, cfg.device)
        kept_blocks, kept_idx = select_blocks(imp, cfg.block_size, cfg.keep_frac)
        rep = compress_mlp_layer_swiglu_inplace(model, li, kept_idx, dtype, cfg.device)
        print(f"Layer {li}: {len(kept_blocks)} blocks, {rep['ffn_kept']}/{imp.numel()} neurons")


@app.command()
def repair(
    config: ConfigOpt = None,
    set_: OverrideOpt = None,
) -> None:
    """Calibrate + compile + fine-tune compressed MLP layers."""
    cfg = _cfg(config, set_)
    _setup(cfg)
    from pnn_compiler.data import load_text_sets
    from pnn_compiler.calibrate import collect_ffn_importance_swiglu
    from pnn_compiler.compile import select_blocks, compress_mlp_layer_swiglu_inplace
    from pnn_compiler.repair import repair_layers

    (model, tok), dtype = _load(cfg)
    texts = load_text_sets(cfg.n_texts_cal, cfg.n_texts_train, cfg.n_texts_eval)
    for li in cfg.layers:
        imp = collect_ffn_importance_swiglu(model, tok, li, texts["cal"], cfg.max_len, cfg.device)
        kept_blocks, kept_idx = select_blocks(imp, cfg.block_size, cfg.keep_frac)
        compress_mlp_layer_swiglu_inplace(model, li, kept_idx, dtype, cfg.device)

    repair_layers(
        model=model, tokenizer=tok, texts_train=texts["train"],
        layers=cfg.layers, steps=cfg.repair_steps, lr=cfg.lr,
        warmup=cfg.warmup, weight_decay=cfg.weight_decay,
        max_len=cfg.max_len, device=cfg.device,
    )


@app.command("eval")
def eval_(
    config: ConfigOpt = None,
    set_: OverrideOpt = None,
) -> None:
    """Evaluate perplexity on wikitext-2 and wikitext-103."""
    cfg = _cfg(config, set_)
    _setup(cfg)
    from pnn_compiler.data import load_text_sets
    from pnn_compiler.eval import eval_perplexity

    (model, tok), dtype = _load(cfg)
    texts = load_text_sets(cfg.n_texts_cal, cfg.n_texts_train, cfg.n_texts_eval)
    ppl_w2 = eval_perplexity(model, tok, texts["eval_w2"], cfg.max_len, cfg.device, cfg.max_eval_tokens)
    ppl_w103 = eval_perplexity(model, tok, texts["eval_w103"], cfg.max_len, cfg.device, cfg.max_eval_tokens)
    print(f"PPL wikitext-2={ppl_w2:.2f}  wikitext-103={ppl_w103:.2f}")


@app.command()
def bench(
    config: ConfigOpt = None,
    set_: OverrideOpt = None,
) -> None:
    """Benchmark prefill throughput (tokens/sec)."""
    cfg = _cfg(config, set_)
    _setup(cfg)
    from pnn_compiler.data import load_text_sets
    from pnn_compiler.bench import bench_prefill_tokens_per_sec

    (model, tok), dtype = _load(cfg)
    texts = load_text_sets(cfg.n_texts_cal, cfg.n_texts_train, cfg.n_texts_eval)
    tps = bench_prefill_tokens_per_sec(
        model, tok, texts["eval_w2"][: cfg.bench_texts],
        cfg.max_len, cfg.device, cfg.bench_warmup_iters, cfg.bench_iters,
    )
    print(f"{tps:,.0f} tokens/sec")


@app.command("run-all")
def run_all(
    config: ConfigOpt = None,
    set_: OverrideOpt = None,
) -> None:
    """Full pipeline: baseline → calibrate → compile → eval → repair → eval."""
    cfg = _cfg(config, set_)
    _setup(cfg)
    from pnn_compiler.pipeline import run_pipeline

    results = run_pipeline(cfg)

    for stage, vals in results.items():
        print(f"\n{'=' * 50}")
        print(f"  {stage}")
        print(f"{'=' * 50}")
        if isinstance(vals, dict):
            for k, v in vals.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    app()
