"""Auto-detect model architecture from HuggingFace config.

Uses AutoConfig to inspect the model without downloading weights, then
maps config fields to an ArchitectureDescriptor with a confidence score.
"""

from __future__ import annotations

import logging
from typing import Optional

from .descriptor import ArchitectureDescriptor, MlpType, SupportState

_log = logging.getLogger(__name__)

# Maps HF config model_type -> (family, mlp_type, gating, activation)
_KNOWN_ARCHITECTURES = {
    "llama":    ("llama",    MlpType.SWIGLU, True,  "silu"),
    "mistral":  ("mistral",  MlpType.SWIGLU, True,  "silu"),
    "qwen2":    ("qwen",     MlpType.SWIGLU, True,  "silu"),
    "qwen3_5":  ("qwen",     MlpType.SWIGLU, True,  "silu"),
    "qwen3_5_text": ("qwen", MlpType.SWIGLU, True,  "silu"),
    "phi3":     ("phi",      MlpType.SWIGLU, True,  "silu"),
    "phi":      ("phi",      MlpType.SWIGLU, True,  "silu"),
    "gemma":    ("gemma",    MlpType.SWIGLU, True,  "gelu_pytorch_tanh"),
    "gemma2":   ("gemma",    MlpType.SWIGLU, True,  "gelu_pytorch_tanh"),
    "starcoder2": ("starcoder", MlpType.PLAIN, False, "gelu_pytorch_tanh"),
    # MoE variants
    "mixtral":  ("mixtral",  MlpType.SWIGLU, True,  "silu"),
    "deepseek": ("deepseek", MlpType.SWIGLU, True,  "silu"),
    "olmoe":    ("mixtral",  MlpType.SWIGLU, True,  "silu"),
    "qwen2_moe": ("qwen_moe", MlpType.SWIGLU, True, "silu"),
    "qwen3_5_moe": ("qwen_moe", MlpType.SWIGLU, True, "silu"),
    "qwen3_5_moe_text": ("qwen_moe", MlpType.SWIGLU, True, "silu"),
    # Multimodal wrappers with dense SwiGLU LLM backbone
    "minicpmo": ("minicpm", MlpType.SWIGLU, True, "silu"),
    "minicpmv": ("minicpm", MlpType.SWIGLU, True, "silu"),
    # Plain MLP (ungated)
    "gpt2":     ("gpt2",     MlpType.PLAIN,  False, "gelu_new"),
    "gptj":     ("gptj",     MlpType.PLAIN,  False, "gelu_new"),
    "gpt_neox": ("gpt_neox", MlpType.PLAIN,  False, "gelu"),
    "opt":      ("opt",      MlpType.PLAIN,  False, "relu"),
    "bloom":    ("bloom",    MlpType.PLAIN,  False, "gelu"),
}

# Families where the SwiGLU dense adapter works today
_DENSE_SWIGLU_FAMILIES = {"llama", "mistral", "qwen", "phi", "gemma", "minicpm"}

_DENSE_PLAIN_FAMILIES = {"starcoder"}

_MULTIMODAL_FAMILIES = {"minicpm"}


def _extract_num_params(config) -> Optional[int]:
    """Best-effort parameter count from config alone."""
    try:
        cfg = config
        if hasattr(config, "text_config") and config.text_config is not None:
            tc = config.text_config
            if getattr(tc, "hidden_size", 0) > 0:
                cfg = tc
        h = cfg.hidden_size
        L = cfg.num_hidden_layers
        V = cfg.vocab_size
        I = getattr(cfg, "intermediate_size", 4 * h)
        gated = getattr(cfg, "hidden_act", "") in ("silu", "swish")
        mlp_factor = 3 if gated else 2
        return 2 * V * h + L * (4 * h * h + mlp_factor * h * I)
    except Exception:
        return None


def fingerprint(model_id: str) -> ArchitectureDescriptor:
    """Fingerprint a model by inspecting its HuggingFace config.

    Does NOT download weights — only fetches config.json.
    Returns an ArchitectureDescriptor with confidence and support_state.
    """
    from transformers import AutoConfig

    _log.info("fingerprinting %s", model_id)
    config = None
    config_dict: Optional[dict] = None

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    except (ValueError, KeyError):
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        except (ValueError, KeyError, OSError):
            pass

    if config is None:
        # AutoConfig can't handle this model_type (newer than installed
        # transformers). Fall back to reading config.json as raw JSON.
        _log.info("AutoConfig failed — falling back to raw config.json")
        import json
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(model_id, "config.json")
        with open(cfg_path) as f:
            config_dict = json.load(f)

    if config_dict is not None:
        # Raw dict path: extract fields manually, supporting nested text_config
        tc = config_dict.get("text_config", config_dict)
        if tc.get("hidden_size", 0) > 0:
            text_d = tc
        else:
            text_d = config_dict
        model_type = text_d.get("model_type") or config_dict.get("model_type", "")
        config_class = "raw_json"
        hidden_size = text_d.get("hidden_size", 0)
        num_layers = text_d.get("num_hidden_layers", 0)
        intermediate_size = text_d.get("intermediate_size", 0)
        num_attention_heads = text_d.get("num_attention_heads", 0)
        num_kv_heads = text_d.get("num_key_value_heads")
        vocab_size = text_d.get("vocab_size", 0)
        activation = text_d.get("hidden_act", "unknown")
        tie_embeddings = config_dict.get("tie_word_embeddings", False)
        context_length = (
            text_d.get("max_position_embeddings")
            or text_d.get("max_sequence_length")
        )
        num_experts = text_d.get("num_local_experts") or text_d.get("num_experts")
        num_experts_per_tok = text_d.get("num_experts_per_tok")
    else:
        model_type = getattr(config, "model_type", None) or ""
        config_class = type(config).__name__

        # Multimodal models (e.g. Qwen3.5) nest text attributes inside text_config.
        text_cfg = config
        if hasattr(config, "text_config") and config.text_config is not None:
            tc = config.text_config
            if getattr(tc, "hidden_size", 0) > 0:
                text_cfg = tc
                tc_model_type = getattr(tc, "model_type", None)
                if tc_model_type:
                    model_type = tc_model_type
                _log.info("using nested text_config (model_type=%s)", model_type)

        hidden_size = getattr(text_cfg, "hidden_size", 0)
        num_layers = getattr(text_cfg, "num_hidden_layers", 0)
        intermediate_size = getattr(text_cfg, "intermediate_size", 0)
        num_attention_heads = getattr(text_cfg, "num_attention_heads", 0)
        num_kv_heads = getattr(text_cfg, "num_key_value_heads", None)
        vocab_size = getattr(text_cfg, "vocab_size", 0)
        activation = getattr(text_cfg, "hidden_act", "unknown")
        tie_embeddings = getattr(config, "tie_word_embeddings", False)

    if config is not None:
        context_length = (
            getattr(config, "max_position_embeddings", None)
            or getattr(config, "max_sequence_length", None)
            or getattr(config, "n_positions", None)
        )
        num_experts = getattr(config, "num_local_experts", None) or getattr(config, "num_experts", None)
        num_experts_per_tok = getattr(config, "num_experts_per_tok", None)

    is_moe = num_experts is not None and num_experts > 1

    # Lookup known architecture
    known = _KNOWN_ARCHITECTURES.get(model_type)

    if known is not None:
        family, mlp_type, gating, _act = known
        activation = activation or _act
        confidence = 1.0
    else:
        family = model_type or "unknown"
        gating = activation in ("silu", "swish")
        mlp_type = MlpType.SWIGLU if gating else MlpType.PLAIN
        confidence = 0.3
        _log.warning(
            "unknown model_type %r for %s — confidence=%.1f",
            model_type, model_id, confidence,
        )

    # Determine support state
    if is_moe:
        _moe_families = {"mixtral", "deepseek", "qwen_moe"}
        if family in _moe_families and confidence >= 0.8:
            support_state = SupportState.SUPPORTED
        else:
            support_state = SupportState.PARTIALLY_SUPPORTED
    elif family in _DENSE_SWIGLU_FAMILIES and confidence >= 0.8:
        support_state = SupportState.SUPPORTED
    elif family in _DENSE_PLAIN_FAMILIES and confidence >= 0.8:
        support_state = SupportState.SUPPORTED
    elif mlp_type == MlpType.PLAIN:
        support_state = SupportState.NEEDS_ADAPTER
    elif confidence < 0.5:
        support_state = SupportState.UNSUPPORTED
    else:
        support_state = SupportState.NEEDS_ADAPTER

    desc = ArchitectureDescriptor(
        family=family,
        model_type=model_type,
        config_class=config_class,
        hidden_size=hidden_size,
        num_layers=num_layers,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        context_length=context_length,
        num_params=_extract_num_params(config),
        mlp_type=mlp_type,
        activation_type=activation,
        gating=gating,
        moe=is_moe,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        is_multimodal=family in _MULTIMODAL_FAMILIES,
        tie_word_embeddings=tie_embeddings,
        confidence=confidence,
        support_state=support_state,
    )

    _log.info(
        "fingerprint: family=%s mlp=%s moe=%s confidence=%.1f support=%s",
        desc.family, desc.mlp_type, desc.moe, desc.confidence, desc.support_state,
    )
    return desc
