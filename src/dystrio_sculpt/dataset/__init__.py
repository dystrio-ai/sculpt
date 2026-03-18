"""Dystrio Efficiency Dataset — structured JSONL logging of optimization runs."""

from .schema import (
    SCHEMA_VERSION,
    DatasetRecord,
    TierRecord,
    EnvironmentInfo,
    BaselineInfo,
    LayerInsight,
    RiskProfile,
    PolicyTrace,
)
from .logger import DatasetLogger

__all__ = [
    "SCHEMA_VERSION",
    "DatasetRecord",
    "TierRecord",
    "EnvironmentInfo",
    "BaselineInfo",
    "LayerInsight",
    "RiskProfile",
    "PolicyTrace",
    "DatasetLogger",
]
