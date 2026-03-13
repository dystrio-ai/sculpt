"""SwiGLU MoE adapter stub for Mixtral, DeepSeek-MoE, Qwen-MoE.

MoE architectures use sparse expert selection with SwiGLU FFN blocks per
expert.  Optimization here means compressing individual expert FFN blocks
while preserving the routing/gating mechanism.

This stub defines the interface but raises NotImplementedError — implementing
MoE compression requires:
  1. Expert-level calibration (importance per expert, not per layer)
  2. Expert-aware compression (keep routing intact)
  3. Possibly expert merging or dropping as additional optimization targets
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

from .base import ArchitectureAdapter
from .descriptor import OptimizationTarget


class SwiGLUMoEAdapter(ArchitectureAdapter):
    """Stub adapter for Mixture-of-Experts with SwiGLU FFN blocks.

    Status: PARTIALLY_SUPPORTED — fingerprinting works, optimization is experimental.
    """

    def supported_targets(self) -> List[OptimizationTarget]:
        return [OptimizationTarget.MLP_BLOCK, OptimizationTarget.EXPERT_BLOCK]

    def get_num_layers(self, model) -> int:
        return model.config.num_hidden_layers

    def get_mlp(self, model, layer_idx: int):
        # Mixtral: model.model.layers[i].block_sparse_moe
        layer = model.model.layers[layer_idx]
        if hasattr(layer, "block_sparse_moe"):
            return layer.block_sparse_moe
        raise NotImplementedError(
            f"Cannot locate MoE module for {type(model).__name__}"
        )

    def get_ffn_size(self, model, layer_idx: int) -> int:
        moe = self.get_mlp(model, layer_idx)
        if hasattr(moe, "experts") and len(moe.experts) > 0:
            expert = moe.experts[0]
            if hasattr(expert, "w1"):
                return expert.w1.out_features
            if hasattr(expert, "gate_proj"):
                return expert.gate_proj.out_features
        raise NotImplementedError("Cannot determine FFN size for MoE expert")

    def compress_layer(self, model, layer_idx, kept_idx, dtype, device):
        raise NotImplementedError(
            "MoE compression not yet implemented. "
            "Requires per-expert compression with routing preservation."
        )

    def collect_block_geometry(self, model, tokenizer, layer_idx, texts, max_len, device,
                               block_size=128, max_tokens=30_000):
        raise NotImplementedError("MoE calibration not yet implemented.")

    def collect_block_sensitivity(self, model, tokenizer, layer_idx, texts, max_len, device,
                                   block_size=128, max_tokens=30_000):
        raise NotImplementedError("MoE sensitivity not yet implemented.")

    def collect_importance(self, model, tokenizer, layer_idx, texts, max_len, device):
        raise NotImplementedError("MoE importance not yet implemented.")

    def snapshot_trainable(self, model, layers):
        raise NotImplementedError("MoE snapshot not yet implemented.")

    def restore_trainable(self, model, layers, snap):
        raise NotImplementedError("MoE restore not yet implemented.")

    def get_trainable_params(self, model, layers):
        raise NotImplementedError("MoE trainable params not yet implemented.")
