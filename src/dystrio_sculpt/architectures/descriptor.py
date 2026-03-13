"""Architecture descriptor: structured representation of a model's architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SupportState(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    NEEDS_ADAPTER = "NEEDS_ADAPTER"


class MlpType(str, Enum):
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    REGLU = "reglu"
    GELU_GATED = "gelu_gated"
    PLAIN = "plain"
    UNKNOWN = "unknown"


class OptimizationTarget(str, Enum):
    """Types of structural optimization that can be applied."""
    MLP_BLOCK = "mlp_block"
    ATTENTION_HEAD = "attention_head"
    FULL_LAYER = "full_layer"
    EXPERT_BLOCK = "expert_block"


@dataclass
class ArchitectureDescriptor:
    """Full description of a model's architecture for optimization dispatch.

    Designed to be future-proof: includes fields for MoE, attention,
    KV cache, and multimodal traits even though v1 only uses MLP fields.
    """

    # Identity
    family: str
    model_type: Optional[str] = None
    config_class: Optional[str] = None

    # Core dimensions
    hidden_size: int = 0
    num_layers: int = 0
    intermediate_size: int = 0
    num_attention_heads: int = 0
    num_kv_heads: Optional[int] = None
    vocab_size: int = 0
    context_length: Optional[int] = None
    num_params: Optional[int] = None

    # MLP structure
    mlp_type: str = MlpType.UNKNOWN
    activation_type: str = "silu"
    gating: bool = False

    # MoE
    moe: bool = False
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None

    # Attention traits (future optimization surface)
    has_cross_attention: bool = False
    rope: bool = True

    # Multimodal
    is_multimodal: bool = False

    # Embedding
    tie_word_embeddings: bool = False

    # Dispatch metadata
    confidence: float = 1.0
    support_state: str = SupportState.UNSUPPORTED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ArchitectureDescriptor:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})
