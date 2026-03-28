"""StarCoder2 adapter for plain (2-projection) MLP models.

StarCoder2 uses c_fc -> gelu -> c_proj (no gate/up split like SwiGLU).
This adapter handles calibration, compression, and repair for that pattern.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

from .base import ArchitectureAdapter
from .descriptor import OptimizationTarget


class Starcoder2Adapter(ArchitectureAdapter):
    """Adapter for StarCoder2 (and similar plain-MLP models with c_fc/c_proj)."""

    UP_NAME = "c_fc"
    DOWN_NAME = "c_proj"

    def supported_targets(self) -> List[OptimizationTarget]:
        return [OptimizationTarget.MLP_BLOCK]

    # ── Layer access ──────────────────────────────────────────────────────

    def get_num_layers(self, model) -> int:
        return model.config.num_hidden_layers

    def get_mlp(self, model, layer_idx: int):
        return model.model.layers[layer_idx].mlp

    def get_ffn_size(self, model, layer_idx: int) -> int:
        mlp = self.get_mlp(model, layer_idx)
        return getattr(mlp, self.UP_NAME).out_features

    # ── Compression ───────────────────────────────────────────────────────

    def compress_layer(
        self, model, layer_idx: int, kept_idx: torch.Tensor,
        dtype: torch.dtype, device: str,
    ) -> Dict[str, int]:
        from .._compile import compress_mlp_layer_plain_inplace
        return compress_mlp_layer_plain_inplace(
            model, layer_idx, kept_idx, dtype, device,
            up_name=self.UP_NAME, down_name=self.DOWN_NAME,
        )

    # ── Calibration ───────────────────────────────────────────────────────

    def collect_block_geometry(
        self, model, tokenizer, layer_idx: int, texts: Sequence[str],
        max_len: int, device: str, block_size: int, max_tokens: int = 30_000,
    ) -> Dict[str, Any]:
        from .._calibrate import collect_block_geometry_plain
        return collect_block_geometry_plain(
            model, tokenizer, layer_idx, texts, max_len, device,
            block_size=block_size, max_tokens=max_tokens,
        )

    def collect_block_sensitivity(
        self, model, tokenizer, layer_idx: int, texts: Sequence[str],
        max_len: int, device: str, block_size: int, max_tokens: int = 30_000,
    ) -> Dict[str, Any]:
        from .._calibrate import collect_block_operator_sensitivity_plain
        return collect_block_operator_sensitivity_plain(
            model, tokenizer, layer_idx, texts, max_len, device,
            block_size=block_size, max_tokens=max_tokens,
        )

    def collect_importance(
        self, model, tokenizer, layer_idx: int, texts: Sequence[str],
        max_len: int, device: str,
    ) -> torch.Tensor:
        from .._calibrate import collect_ffn_importance_plain
        return collect_ffn_importance_plain(
            model, tokenizer, layer_idx, texts, max_len, device,
        )

    # ── Repair support ────────────────────────────────────────────────────

    def snapshot_trainable(
        self, model, layers: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        snap: Dict[str, torch.Tensor] = {}
        for li in layers:
            mlp = model.model.layers[li].mlp
            for name, p in mlp.named_parameters():
                key = f"layers.{li}.mlp.{name}"
                snap[key] = p.data.detach().cpu().clone()
        return snap

    def restore_trainable(
        self, model, layers: Sequence[int],
        snap: Dict[str, torch.Tensor],
    ) -> None:
        for li in layers:
            mlp = model.model.layers[li].mlp
            for name, p in mlp.named_parameters():
                key = f"layers.{li}.mlp.{name}"
                if key in snap:
                    p.data.copy_(snap[key].to(p.device))

    def get_trainable_params(
        self, model, layers: Sequence[int],
    ) -> list[torch.nn.Parameter]:
        params = []
        for li in layers:
            for p in model.model.layers[li].mlp.parameters():
                params.append(p)
        return params
