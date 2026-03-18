"""Architecture registry: maps model families to adapters.

Usage::

    from dystrio_sculpt.architectures import fingerprint, get_adapter

    desc = fingerprint(model_id)
    adapter = get_adapter(desc)
    adapter.compress_layer(model, layer_idx, kept_idx, dtype, device)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Type

from .base import ArchitectureAdapter
from .descriptor import ArchitectureDescriptor, SupportState
from .fingerprint import fingerprint

_log = logging.getLogger(__name__)

_REGISTRY: Dict[str, Type[ArchitectureAdapter]] = {}


def register(family: str, adapter_cls: Type[ArchitectureAdapter]) -> None:
    """Register an adapter class for a model family."""
    _REGISTRY[family] = adapter_cls
    _log.debug("registered adapter %s for family %r", adapter_cls.__name__, family)


def get_adapter(desc: ArchitectureDescriptor) -> ArchitectureAdapter:
    """Look up and instantiate the adapter for a given architecture descriptor.

    Raises ValueError if the architecture is unsupported or no adapter is registered.
    """
    if desc.support_state == SupportState.UNSUPPORTED:
        raise ValueError(
            f"Architecture {desc.family!r} (model_type={desc.model_type!r}) "
            f"is unsupported (confidence={desc.confidence:.1f})"
        )

    adapter_cls = _REGISTRY.get(desc.family)
    if adapter_cls is None:
        raise ValueError(
            f"No adapter registered for family {desc.family!r}. "
            f"Registered families: {sorted(_REGISTRY.keys())}"
        )
    return adapter_cls()


def get_adapter_for_model(model_id: str) -> tuple[ArchitectureDescriptor, ArchitectureAdapter]:
    """Convenience: fingerprint + adapter lookup in one call."""
    desc = fingerprint(model_id)
    adapter = get_adapter(desc)
    return desc, adapter


def _auto_register() -> None:
    """Register built-in adapters. Called on import."""
    from .swiglu_dense import SwiGLUDenseAdapter
    from .plain_mlp import PlainMLPAdapter
    from .swiglu_moe import SwiGLUMoEAdapter

    for family in ("llama", "mistral", "qwen", "phi", "gemma", "starcoder"):
        register(family, SwiGLUDenseAdapter)

    for family in ("gpt2", "gptj", "gpt_neox", "opt", "bloom"):
        register(family, PlainMLPAdapter)

    for family in ("mixtral", "deepseek", "qwen_moe"):
        register(family, SwiGLUMoEAdapter)


_auto_register()

__all__ = [
    "fingerprint",
    "get_adapter",
    "get_adapter_for_model",
    "register",
    "ArchitectureAdapter",
    "ArchitectureDescriptor",
    "SupportState",
]
