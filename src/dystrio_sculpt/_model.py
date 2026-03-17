"""Model and tokenizer loading, plus architecture-agnostic layer access."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

_log = logging.getLogger(__name__)

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def resolve_dtype(s: str) -> torch.dtype:
    d = DTYPE_MAP.get(s.lower())
    if d is None:
        raise ValueError(f"Unknown dtype: {s!r}. Use one of {list(DTYPE_MAP)}")
    return d


def _find_layers(model: nn.Module) -> nn.ModuleList:
    """Resolve the decoder layer list for any supported architecture.

    Probes known paths in priority order:
      1. model.model.layers        (Llama, Mistral, Gemma, Qwen2.5, …)
      2. model.language_model.layers (Qwen3.5, multimodal wrappers)
      3. model.transformer.h        (GPT-2 / Phi-2 style)
    """
    candidates = [
        ("model.model.layers", lambda m: m.model.layers),
        ("model.language_model.layers", lambda m: m.language_model.layers),
        ("model.transformer.h", lambda m: m.transformer.h),
    ]
    for path, accessor in candidates:
        try:
            layers = accessor(model)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                _log.debug("layer path resolved: %s (%d layers)", path, len(layers))
                return layers
        except AttributeError:
            continue
    raise RuntimeError(
        "Unable to locate decoder layers. Supported paths: "
        + ", ".join(p for p, _ in candidates)
    )


def get_layers(model: nn.Module) -> nn.ModuleList:
    """Return the decoder layer list (cached on the model instance)."""
    if not hasattr(model, "_sculpt_layers"):
        model._sculpt_layers = _find_layers(model)
    return model._sculpt_layers


def get_mlp(model: nn.Module, layer_idx: int) -> Any:
    """Return the MLP sub-module for a given layer index."""
    return get_layers(model)[layer_idx].mlp


def get_text_config(model: nn.Module):
    """Return the text/LM config, handling nested multimodal configs."""
    cfg = model.config
    if hasattr(cfg, "text_config") and cfg.text_config is not None:
        return cfg.text_config
    return cfg


def load_model_and_tokenizer(
    model_id: str, device: str, dtype: torch.dtype,
):
    from transformers import AutoConfig

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True,
        ).to(device)
    except AttributeError:
        # Multimodal wrappers (e.g. Qwen3.5) expose a composite config
        # that lacks top-level text attributes like vocab_size.
        # Load the full model and verify it has a CausalLM head.
        from transformers import AutoModel
        _log.warning(
            "AutoModelForCausalLM failed for %s — falling back to AutoModel",
            model_id,
        )
        model = AutoModel.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True,
        ).to(device)
        if not hasattr(model, "lm_head") and not hasattr(model, "output"):
            _log.warning(
                "loaded model has no lm_head — perplexity evaluation "
                "will rely on model(**inp) returning loss"
            )
    return model, tok
