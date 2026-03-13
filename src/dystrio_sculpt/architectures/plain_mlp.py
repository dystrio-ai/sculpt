"""Plain MLP adapter stub for GPT-2, OPT, Bloom, GPT-J, GPT-NeoX.

These architectures use a simple 2-matrix MLP (up_proj -> act -> down_proj)
without gating. This stub defines the interface but raises NotImplementedError
for methods that need architecture-specific calibration and compression logic.

Implementing this adapter is a future milestone — it requires different
importance scoring (no gate/up split) and different compression math.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

from .base import ArchitectureAdapter
from .descriptor import OptimizationTarget


class PlainMLPAdapter(ArchitectureAdapter):
    """Stub adapter for plain (ungated) MLP architectures.

    Status: NEEDS_ADAPTER — fingerprinting works but optimization is not yet implemented.
    """

    def supported_targets(self) -> List[OptimizationTarget]:
        return [OptimizationTarget.MLP_BLOCK]

    def get_num_layers(self, model) -> int:
        config = model.config
        return getattr(config, "num_hidden_layers", 0) or getattr(config, "n_layer", 0)

    def get_mlp(self, model, layer_idx: int):
        # GPT-2 style: model.transformer.h[i].mlp
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[layer_idx].mlp
        # OPT style: model.model.decoder.layers[i].fc1/fc2
        if hasattr(model, "model") and hasattr(model.model, "decoder"):
            return model.model.decoder.layers[layer_idx]
        raise NotImplementedError(
            f"Cannot locate MLP for {type(model).__name__}. "
            "PlainMLPAdapter needs extension for this architecture."
        )

    def get_ffn_size(self, model, layer_idx: int) -> int:
        mlp = self.get_mlp(model, layer_idx)
        for name in ("c_fc", "fc1", "dense_h_to_4h", "up_proj"):
            if hasattr(mlp, name):
                proj = getattr(mlp, name)
                return proj.out_features if hasattr(proj, "out_features") else proj.nf
        raise NotImplementedError("Cannot determine FFN size for plain MLP")

    def compress_layer(self, model, layer_idx, kept_idx, dtype, device):
        raise NotImplementedError(
            "Plain MLP compression not yet implemented. "
            "Requires different slicing logic (no gate_proj)."
        )

    def collect_block_geometry(self, model, tokenizer, layer_idx, texts, max_len, device,
                               block_size=128, max_tokens=30_000):
        raise NotImplementedError("Plain MLP calibration not yet implemented.")

    def collect_block_sensitivity(self, model, tokenizer, layer_idx, texts, max_len, device,
                                   block_size=128, max_tokens=30_000):
        raise NotImplementedError("Plain MLP sensitivity not yet implemented.")

    def collect_importance(self, model, tokenizer, layer_idx, texts, max_len, device):
        raise NotImplementedError("Plain MLP importance not yet implemented.")

    def snapshot_trainable(self, model, layers):
        raise NotImplementedError("Plain MLP snapshot not yet implemented.")

    def restore_trainable(self, model, layers, snap):
        raise NotImplementedError("Plain MLP restore not yet implemented.")

    def get_trainable_params(self, model, layers):
        raise NotImplementedError("Plain MLP trainable params not yet implemented.")
