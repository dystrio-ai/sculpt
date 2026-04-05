"""Abstract base class for architecture adapters.

An adapter tells the engine how to access, compress, calibrate, and repair
a specific model architecture.  The interface is deliberately broader than
MLP compression so that future optimization targets (attention heads, full
layers, expert blocks) can be added without changing the adapter contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .descriptor import ArchitectureDescriptor, OptimizationTarget


class ArchitectureAdapter(ABC):
    """Interface that architecture-specific adapters must implement."""

    @abstractmethod
    def supported_targets(self) -> List[OptimizationTarget]:
        """Which optimization targets this adapter can handle."""
        ...

    # ── Layer access ──────────────────────────────────────────────────────

    @abstractmethod
    def get_num_layers(self, model) -> int:
        ...

    @abstractmethod
    def get_mlp(self, model, layer_idx: int):
        """Return the MLP module for a given layer."""
        ...

    @abstractmethod
    def get_ffn_size(self, model, layer_idx: int) -> int:
        """Return the intermediate (FFN) dimension for a given layer."""
        ...

    # ── Compression ───────────────────────────────────────────────────────

    @abstractmethod
    def compress_layer(
        self, model, layer_idx: int, kept_idx: torch.Tensor,
        dtype: torch.dtype, device: str, **kwargs: Any,
    ) -> Dict[str, int]:
        """Physically compress a layer in-place, return info dict."""
        ...

    # ── Calibration ───────────────────────────────────────────────────────

    @abstractmethod
    def collect_block_geometry(
        self, model, tokenizer, layer_idx: int, texts: Sequence[str],
        max_len: int, device: str, block_size: int, max_tokens: int = 30_000,
    ) -> Dict[str, Any]:
        """Block-level covariance / geometry for structural selection."""
        ...

    @abstractmethod
    def collect_block_sensitivity(
        self, model, tokenizer, layer_idx: int, texts: Sequence[str],
        max_len: int, device: str, block_size: int, max_tokens: int = 30_000,
    ) -> Dict[str, Any]:
        """Operator fidelity: how much zeroing each block changes output."""
        ...

    @abstractmethod
    def collect_importance(
        self, model, tokenizer, layer_idx: int, texts: Sequence[str],
        max_len: int, device: str,
    ) -> torch.Tensor:
        """Per-neuron magnitude importance for magnitude-based selection."""
        ...

    # ── Repair support ────────────────────────────────────────────────────

    @abstractmethod
    def snapshot_trainable(
        self, model, layers: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        """Save a CPU copy of trainable parameters for the given layers."""
        ...

    @abstractmethod
    def restore_trainable(
        self, model, layers: Sequence[int],
        snap: Dict[str, torch.Tensor],
    ) -> None:
        """Restore trainable parameters from a snapshot."""
        ...

    @abstractmethod
    def get_trainable_params(
        self, model, layers: Sequence[int],
    ) -> List[torch.nn.Parameter]:
        """Return trainable parameters for repair optimisation."""
        ...

    # ── Model routing ─────────────────────────────────────────────────────

    def get_eval_model(self, model):
        """Return the sub-model to use for inference (perplexity, repair, benchmarks).

        For standalone models this is the model itself.  For multimodal
        wrappers (e.g. MiniCPM-o) this returns the inner LLM backbone so
        that text-only forward passes work correctly.

        The returned model shares layers with *model*, so in-place
        compression is visible through both references.
        """
        return model
