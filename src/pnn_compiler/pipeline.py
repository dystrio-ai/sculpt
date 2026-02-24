"""Full pipeline: baseline → calibrate → compile → eval/bench → repair → eval/bench."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from .config import EngineConfig, resolve_dtype
from .data import load_text_sets
from .model import load_model_and_tokenizer
from .calibrate import collect_ffn_importance_swiglu
from .compile import select_blocks, compress_mlp_layer_swiglu_inplace
from .repair import repair_layers
from .eval import eval_perplexity
from .bench import bench_prefill_tokens_per_sec


def run_pipeline(cfg: EngineConfig) -> Dict[str, Any]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype = resolve_dtype(cfg.dtype)
    model, tok = load_model_and_tokenizer(cfg.model_id, cfg.device, dtype)
    texts = load_text_sets(cfg.n_texts_cal, cfg.n_texts_train, cfg.n_texts_eval)

    eval_w2 = texts["eval_w2"]
    eval_w103 = texts["eval_w103"]
    bench_texts = eval_w2[: cfg.bench_texts]

    out: Dict[str, Any] = {}

    # baseline
    out["baseline"] = {
        "ppl_w2": eval_perplexity(
            model, tok, eval_w2, cfg.max_len, cfg.device, cfg.max_eval_tokens,
        ),
        "ppl_w103": eval_perplexity(
            model, tok, eval_w103, cfg.max_len, cfg.device, cfg.max_eval_tokens,
        ),
        "tps": bench_prefill_tokens_per_sec(
            model, tok, bench_texts, cfg.max_len, cfg.device,
            cfg.bench_warmup_iters, cfg.bench_iters,
        ),
    }

    # compile layers
    out["compile"] = {}
    for li in cfg.layers:
        imp = collect_ffn_importance_swiglu(
            model, tok, li, texts["cal"], cfg.max_len, cfg.device,
        )
        kept_blocks, kept_idx = select_blocks(imp, cfg.block_size, cfg.keep_frac)
        rep = compress_mlp_layer_swiglu_inplace(model, li, kept_idx, dtype, cfg.device)
        out["compile"][str(li)] = {"kept_blocks": len(kept_blocks), **rep}

    # pre-repair
    out["compressed_pre_repair"] = {
        "ppl_w2": eval_perplexity(
            model, tok, eval_w2, cfg.max_len, cfg.device, cfg.max_eval_tokens,
        ),
        "ppl_w103": eval_perplexity(
            model, tok, eval_w103, cfg.max_len, cfg.device, cfg.max_eval_tokens,
        ),
        "tps": bench_prefill_tokens_per_sec(
            model, tok, bench_texts, cfg.max_len, cfg.device,
            cfg.bench_warmup_iters, cfg.bench_iters,
        ),
    }

    # repair
    repair_layers(
        model=model,
        tokenizer=tok,
        texts_train=texts["train"],
        layers=cfg.layers,
        steps=cfg.repair_steps,
        lr=cfg.lr,
        warmup=cfg.warmup,
        weight_decay=cfg.weight_decay,
        max_len=cfg.max_len,
        device=cfg.device,
        log_every=500,
    )

    # post-repair
    out["compressed_post_repair"] = {
        "ppl_w2": eval_perplexity(
            model, tok, eval_w2, cfg.max_len, cfg.device, cfg.max_eval_tokens,
        ),
        "ppl_w103": eval_perplexity(
            model, tok, eval_w103, cfg.max_len, cfg.device, cfg.max_eval_tokens,
        ),
        "tps": bench_prefill_tokens_per_sec(
            model, tok, bench_texts, cfg.max_len, cfg.device,
            cfg.bench_warmup_iters, cfg.bench_iters,
        ),
    }

    return out
