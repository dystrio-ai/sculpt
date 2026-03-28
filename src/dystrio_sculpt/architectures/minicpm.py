"""MiniCPM-o/V adapter: prunes the LLM backbone, leaves encoders untouched.

MiniCPM-o wraps a Qwen3-8B LLM (SwiGLU dense) at ``model.llm``, with
transformer layers at ``model.llm.model.layers``.  Vision (SigLip2),
audio (Whisper-medium), and TTS (CosyVoice2) modules sit beside the
LLM and are *not* touched by structural pruning — only the MLP
intermediate dimension of the LLM backbone is compressed.

Because the multimodal forward path expects images/audio inputs that
we don't have during text-only calibration, all calibration and repair
calls are routed through ``model.llm`` directly (a standard
``Qwen3ForCausalLM`` that accepts plain tokenizer output).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

from .base import ArchitectureAdapter
from .descriptor import OptimizationTarget


def _resolve_llm(model: nn.Module) -> nn.Module:
    """Return the LLM backbone from a MiniCPM-o/V wrapper.

    Falls back to the model itself if ``model.llm`` doesn't exist (e.g.
    when the adapter is tested against a standalone Qwen model).
    """
    if hasattr(model, "llm"):
        return model.llm
    return model


def _llm_layers(model: nn.Module) -> nn.ModuleList:
    """Return the transformer layer list from the LLM backbone."""
    llm = _resolve_llm(model)
    return llm.model.layers


class MiniCPMAdapter(ArchitectureAdapter):
    """Adapter for MiniCPM-o / MiniCPM-V multimodal models.

    Only the Qwen3-8B LLM backbone is pruned (MLP blocks).
    Vision / audio encoders and decoders are left untouched.
    """

    def supported_targets(self) -> List[OptimizationTarget]:
        return [OptimizationTarget.MLP_BLOCK]

    # ── Layer access ──────────────────────────────────────────────────────

    def get_num_layers(self, model) -> int:
        return len(_llm_layers(model))

    def get_mlp(self, model, layer_idx: int):
        return _llm_layers(model)[layer_idx].mlp

    def get_ffn_size(self, model, layer_idx: int) -> int:
        return _llm_layers(model)[layer_idx].mlp.gate_proj.out_features

    # ── Compression ───────────────────────────────────────────────────────

    @torch.no_grad()
    def compress_layer(
        self, model, layer_idx: int, kept_idx: torch.Tensor,
        dtype: torch.dtype, device: str,
    ) -> Dict[str, int]:
        mlp = self.get_mlp(model, layer_idx)
        old_gate, old_up, old_down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
        hidden = old_gate.in_features
        ffn_kept = int(kept_idx.numel())
        kept = kept_idx.to(device=device)

        new_gate = nn.Linear(
            hidden, ffn_kept, bias=(old_gate.bias is not None),
            device=device, dtype=dtype,
        )
        new_up = nn.Linear(
            hidden, ffn_kept, bias=(old_up.bias is not None),
            device=device, dtype=dtype,
        )
        new_down = nn.Linear(
            ffn_kept, hidden, bias=(old_down.bias is not None),
            device=device, dtype=dtype,
        )

        new_gate.weight.copy_(old_gate.weight[kept].to(dtype))
        if old_gate.bias is not None:
            new_gate.bias.copy_(old_gate.bias[kept].to(dtype))
        new_up.weight.copy_(old_up.weight[kept].to(dtype))
        if old_up.bias is not None:
            new_up.bias.copy_(old_up.bias[kept].to(dtype))
        new_down.weight.copy_(old_down.weight[:, kept].to(dtype))
        if old_down.bias is not None:
            new_down.bias.copy_(old_down.bias.to(dtype))

        mlp.gate_proj = new_gate
        mlp.up_proj = new_up
        mlp.down_proj = new_down
        return {"hidden": hidden, "ffn_kept": ffn_kept}

    # ── Calibration ───────────────────────────────────────────────────────
    # Route through the LLM backbone so that model(**inp) is a standard
    # causal-LM forward pass that fires MLP hooks correctly.

    def collect_block_geometry(
        self, model, tokenizer, layer_idx: int, texts: Sequence[str],
        max_len: int, device: str, block_size: int, max_tokens: int = 30_000,
    ) -> Dict[str, Any]:
        from .._calibrate import collect_block_geometry_swiglu
        return collect_block_geometry_swiglu(
            _resolve_llm(model), tokenizer, layer_idx, texts,
            max_len, device, block_size=block_size, max_tokens=max_tokens,
        )

    def collect_block_sensitivity(
        self, model, tokenizer, layer_idx: int, texts: Sequence[str],
        max_len: int, device: str, block_size: int, max_tokens: int = 30_000,
    ) -> Dict[str, Any]:
        from .._calibrate import collect_block_operator_sensitivity_swiglu
        return collect_block_operator_sensitivity_swiglu(
            _resolve_llm(model), tokenizer, layer_idx, texts,
            max_len, device, block_size=block_size, max_tokens=max_tokens,
        )

    def collect_importance(
        self, model, tokenizer, layer_idx: int, texts: Sequence[str],
        max_len: int, device: str,
    ) -> torch.Tensor:
        from .._calibrate import collect_ffn_importance_swiglu
        return collect_ffn_importance_swiglu(
            _resolve_llm(model), tokenizer, layer_idx, texts, max_len, device,
        )

    # ── Repair support ────────────────────────────────────────────────────

    def snapshot_trainable(
        self, model, layers: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        snap: Dict[str, torch.Tensor] = {}
        ll = _llm_layers(model)
        for li in layers:
            mlp = ll[li].mlp
            for name, p in mlp.named_parameters():
                key = f"layers.{li}.mlp.{name}"
                snap[key] = p.data.detach().cpu().clone()
        return snap

    def restore_trainable(
        self, model, layers: Sequence[int],
        snap: Dict[str, torch.Tensor],
    ) -> None:
        ll = _llm_layers(model)
        for li in layers:
            mlp = ll[li].mlp
            for name, p in mlp.named_parameters():
                key = f"layers.{li}.mlp.{name}"
                if key in snap:
                    p.data.copy_(snap[key].to(p.device))

    def get_trainable_params(
        self, model, layers: Sequence[int],
    ) -> list[torch.nn.Parameter]:
        params = []
        ll = _llm_layers(model)
        for li in layers:
            for p in ll[li].mlp.parameters():
                params.append(p)
        return params

    # ── Model routing ─────────────────────────────────────────────────────

    def get_eval_model(self, model):
        """Return the LLM backbone for text-only inference.

        The multimodal forward path requires image/audio inputs we don't
        have during sculpt.  The LLM backbone (Qwen3ForCausalLM) accepts
        standard tokenizer output and returns CausalLMOutput with logits.
        """
        return _resolve_llm(model)
