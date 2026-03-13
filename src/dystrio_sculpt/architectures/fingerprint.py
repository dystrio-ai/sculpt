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
    "phi3":     ("phi",      MlpType.SWIGLU, True,  "silu"),
    "phi":      ("phi",      MlpType.SWIGLU, True,  "silu"),
    "gemma":    ("gemma",    MlpType.SWIGLU, True,  "gelu_pytorch_tanh"),
    "gemma2":   ("gemma",    MlpType.SWIGLU, True,  "gelu_pytorch_tanh"),
    "starcoder2": ("starcoder", MlpType.SWIGLU, True, "gelu_pytorch_tanh"),
    # MoE variants
    "mixtral":  ("mixtral",  MlpType.SWIGLU, True,  "silu"),
    "deepseek": ("deepseek", MlpType.SWIGLU, True,  "silu"),
    "qwen2_moe": ("qwen_moe", MlpType.SWIGLU, True, "silu"),
    # Plain MLP (ungated)
    "gpt2":     ("gpt2",     MlpType.PLAIN,  False, "gelu_new"),
    "gptj":     ("gptj",     MlpType.PLAIN,  False, "gelu_new"),
    "gpt_neox": ("gpt_neox", MlpType.PLAIN,  False, "gelu"),
    "opt":      ("opt",      MlpType.PLAIN,  False, "relu"),
    "bloom":    ("bloom",    MlpType.PLAIN,  False, "gelu"),
}

# Families where the SwiGLU dense adapter works today
_DENSE_SWIGLU_FAMILIES = {"llama", "mistral", "qwen", "phi", "gemma", "starcoder"}


def _extract_num_params(config) -> Optional[int]:
    """Best-effort parameter count from config alone."""
    try:
        h = config.hidden_size
        L = config.num_hidden_layers
        V = config.vocab_size
        I = getattr(config, "intermediate_size", 4 * h)
        # Rough: 2*V*h (embed+lm_head) + L*(4*h*h + 3*h*I) for SwiGLU
        gated = getattr(config, "hidden_act", "") in ("silu", "swish")
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
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)

    model_type = getattr(config, "model_type", None) or ""
    config_class = type(config).__name__

    hidden_size = getattr(config, "hidden_size", 0)
    num_layers = getattr(config, "num_hidden_layers", 0)
    intermediate_size = getattr(config, "intermediate_size", 0)
    num_attention_heads = getattr(config, "num_attention_heads", 0)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    vocab_size = getattr(config, "vocab_size", 0)
    activation = getattr(config, "hidden_act", "unknown")
    tie_embeddings = getattr(config, "tie_word_embeddings", False)

    context_length = (
        getattr(config, "max_position_embeddings", None)
        or getattr(config, "max_sequence_length", None)
        or getattr(config, "n_positions", None)
    )

    # MoE detection
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
        support_state = SupportState.PARTIALLY_SUPPORTED
    elif family in _DENSE_SWIGLU_FAMILIES and confidence >= 0.8:
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
        tie_word_embeddings=tie_embeddings,
        confidence=confidence,
        support_state=support_state,
    )

    _log.info(
        "fingerprint: family=%s mlp=%s moe=%s confidence=%.1f support=%s",
        desc.family, desc.mlp_type, desc.moe, desc.confidence, desc.support_state,
    )
    return desc
