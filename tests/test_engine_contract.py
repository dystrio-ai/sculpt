"""
Contract tests: prove the PNN engine matches the canonical specification at runtime.

Covers:
  1. Calibration — fp32 importance vector, correct shape, data-dependent
  2. Compile    — PHYSICAL SLICING: strictly smaller dense nn.Linear layers
  3. Repair     — FREEZE DISCIPLINE: only target MLP params unfrozen + in optimizer
  4. Eval       — every forward call passes use_cache=False

Run:
    cd pnn_compiler && pip install -e ".[dev]"
    pytest tests/test_engine_contract.py -v
"""

from __future__ import annotations

import pytest
import torch

# ── Hardware detection (GPU-first, CPU fallback) ─────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_STR = "bf16" if DEVICE == "cuda" else "fp32"
LAYER_IDX = 3


# ── Module-scoped fixture: load model once, calibrate, compile ───────────


@pytest.fixture(scope="module")
def engine_state():
    """Load Qwen2-0.5B, calibrate layer 3, compile via physical slicing.

    Shared across every test in this module so the expensive model load
    and HF dataset download happen exactly once.
    """
    from pnn_compiler.config import EngineConfig, resolve_dtype
    from pnn_compiler.model import load_model_and_tokenizer
    from pnn_compiler.data import load_text_sets
    from pnn_compiler.calibrate import collect_ffn_importance_swiglu
    from pnn_compiler.compile import select_blocks, compress_mlp_layer_swiglu_inplace

    cfg = EngineConfig(
        model_id="Qwen/Qwen2-0.5B",
        layers=[LAYER_IDX],
        keep_frac=0.5,
        block_size=128,
        dtype=DTYPE_STR,
        device=DEVICE,
        n_texts_cal=5,
        n_texts_train=5,
        n_texts_eval=5,
        repair_steps=2,
        max_len=64,
        max_eval_tokens=500,
        seed=0,
    )
    dtype = resolve_dtype(cfg.dtype)
    model, tok = load_model_and_tokenizer(cfg.model_id, cfg.device, dtype)
    texts = load_text_sets(cfg.n_texts_cal, cfg.n_texts_train, cfg.n_texts_eval)

    # ── Capture original dimensions BEFORE compile ──
    mlp_pre = model.model.layers[LAYER_IDX].mlp
    orig_ffn_dim = mlp_pre.gate_proj.out_features
    hidden_dim = mlp_pre.gate_proj.in_features

    # ── Calibrate ──
    imp = collect_ffn_importance_swiglu(
        model, tok, LAYER_IDX, texts["cal"], cfg.max_len, cfg.device,
    )

    # ── Block-select + compile (structural reparameterisation) ──
    kept_blocks, kept_idx = select_blocks(imp, cfg.block_size, cfg.keep_frac)
    rep = compress_mlp_layer_swiglu_inplace(model, LAYER_IDX, kept_idx, dtype, cfg.device)

    return dict(
        model=model,
        tok=tok,
        texts=texts,
        cfg=cfg,
        dtype=dtype,
        orig_ffn_dim=orig_ffn_dim,
        hidden_dim=hidden_dim,
        new_ffn_dim=rep["ffn_kept"],
        importance=imp,
        kept_idx=kept_idx,
    )


# ═════════════════════════════════════════════════════════════════════════
# 1. CALIBRATION CONTRACT
# ═════════════════════════════════════════════════════════════════════════


class TestCalibration:
    """imp[j] = mean_tokens |act_fn(gate(x))_j * up(x)_j|, fp32."""

    def test_importance_shape_matches_original_ffn(self, engine_state):
        """One importance score per original FFN channel."""
        imp = engine_state["importance"]
        assert imp.shape == (engine_state["orig_ffn_dim"],)

    def test_importance_accumulated_in_fp32(self, engine_state):
        """Spec: accumulate in fp32."""
        assert engine_state["importance"].dtype == torch.float32

    def test_importance_is_nonnegative(self, engine_state):
        """imp = mean |…| ≥ 0 by construction."""
        assert (engine_state["importance"] >= 0).all()

    def test_importance_is_nonzero_for_real_data(self, engine_state):
        """Data-dependent importance must be non-trivial on real text."""
        assert engine_state["importance"].sum().item() > 0


# ═════════════════════════════════════════════════════════════════════════
# 2. COMPILE: PHYSICAL SLICING CONTRACT
# ═════════════════════════════════════════════════════════════════════════


class TestCompilePhysicalSlicing:
    """After compile, the target layer's MLP must consist of strictly
    smaller, dense nn.Linear modules with consistent dimensions.
    """

    def test_ffn_dim_strictly_reduced(self, engine_state):
        """new_ffn_dim < orig_ffn_dim."""
        assert engine_state["new_ffn_dim"] < engine_state["orig_ffn_dim"]

    def test_gate_proj_weight_shape(self, engine_state):
        """gate_proj: [ffn_kept, hidden_dim]."""
        gate = engine_state["model"].model.layers[LAYER_IDX].mlp.gate_proj
        assert gate.weight.shape[0] == engine_state["new_ffn_dim"]
        assert gate.weight.shape[1] == engine_state["hidden_dim"]

    def test_up_proj_weight_shape(self, engine_state):
        """up_proj: [ffn_kept, hidden_dim]."""
        up = engine_state["model"].model.layers[LAYER_IDX].mlp.up_proj
        assert up.weight.shape[0] == engine_state["new_ffn_dim"]
        assert up.weight.shape[1] == engine_state["hidden_dim"]

    def test_down_proj_weight_shape(self, engine_state):
        """down_proj: [hidden_dim, ffn_kept]."""
        down = engine_state["model"].model.layers[LAYER_IDX].mlp.down_proj
        assert down.weight.shape[0] == engine_state["hidden_dim"]
        assert down.weight.shape[1] == engine_state["new_ffn_dim"]

    def test_all_projections_agree_on_ffn_dim(self, engine_state):
        """gate_proj.out == up_proj.out == down_proj.in == ffn_kept."""
        mlp = engine_state["model"].model.layers[LAYER_IDX].mlp
        n = engine_state["new_ffn_dim"]
        assert mlp.gate_proj.out_features == n
        assert mlp.up_proj.out_features == n
        assert mlp.down_proj.in_features == n

    def test_all_projections_agree_on_hidden_dim(self, engine_state):
        """gate_proj.in == up_proj.in == down_proj.out == hidden_dim."""
        mlp = engine_state["model"].model.layers[LAYER_IDX].mlp
        h = engine_state["hidden_dim"]
        assert mlp.gate_proj.in_features == h
        assert mlp.up_proj.in_features == h
        assert mlp.down_proj.out_features == h

    def test_projections_are_plain_nn_linear(self, engine_state):
        """No sparse wrappers, no masked modules — plain dense Linear."""
        mlp = engine_state["model"].model.layers[LAYER_IDX].mlp
        assert type(mlp.gate_proj) is torch.nn.Linear
        assert type(mlp.up_proj) is torch.nn.Linear
        assert type(mlp.down_proj) is torch.nn.Linear

    def test_non_target_layers_untouched(self, engine_state):
        """Every layer OTHER than the target must keep original FFN dim."""
        model = engine_state["model"]
        orig = engine_state["orig_ffn_dim"]
        for i, layer in enumerate(model.model.layers):
            if i == LAYER_IDX:
                continue
            assert layer.mlp.gate_proj.out_features == orig, (
                f"Layer {i} was modified (gate_proj.out_features="
                f"{layer.mlp.gate_proj.out_features}, expected {orig})"
            )


# ═════════════════════════════════════════════════════════════════════════
# 3. REPAIR: FREEZE DISCIPLINE CONTRACT
# ═════════════════════════════════════════════════════════════════════════


class TestRepairFreezeDiscipline:
    """During repair, ONLY model.model.layers[li].mlp parameters may have
    requires_grad=True.  The optimizer must receive exactly those params.
    """

    def test_only_target_mlp_params_require_grad(self, engine_state, monkeypatch):
        """Snapshot requires_grad on the first forward pass inside repair
        and verify the freeze/unfreeze pattern.
        """
        from pnn_compiler.repair import repair_layers

        model = engine_state["model"]
        tok = engine_state["tok"]
        texts = engine_state["texts"]
        cfg = engine_state["cfg"]

        # Capture requires_grad state during the first training forward pass
        grad_snapshot: dict[str, bool] = {}
        real_forward = model.forward

        def capturing_forward(*args, **kwargs):
            if not grad_snapshot:
                for name, p in model.named_parameters():
                    grad_snapshot[name] = p.requires_grad
            return real_forward(*args, **kwargs)

        monkeypatch.setattr(model, "forward", capturing_forward)

        repair_layers(
            model=model, tokenizer=tok, texts_train=texts["train"],
            layers=cfg.layers, steps=1, lr=cfg.lr,
            warmup=0, weight_decay=cfg.weight_decay,
            max_len=cfg.max_len, device=cfg.device,
        )

        assert grad_snapshot, "model.forward was never called during repair"

        target_prefix = f"model.layers.{LAYER_IDX}.mlp."
        unfrozen = {n for n, g in grad_snapshot.items() if g}
        frozen = {n for n, g in grad_snapshot.items() if not g}

        for name in unfrozen:
            assert target_prefix in name, (
                f"Non-target param '{name}' has requires_grad=True during repair"
            )
        for name in frozen:
            assert target_prefix not in name, (
                f"Target MLP param '{name}' has requires_grad=False during repair"
            )
        assert len(unfrozen) > 0, "No parameters were unfrozen during repair"

    def test_optimizer_receives_only_target_mlp_params(self, engine_state, monkeypatch):
        """AdamW must be constructed with exactly the target MLP parameters."""
        from pnn_compiler.repair import repair_layers

        model = engine_state["model"]
        tok = engine_state["tok"]
        texts = engine_state["texts"]
        cfg = engine_state["cfg"]

        captured_param_ids: set[int] = set()
        RealAdamW = torch.optim.AdamW

        class CapturingAdamW(RealAdamW):
            def __init__(self, params, **kwargs):
                param_list = list(params)
                captured_param_ids.update(id(p) for p in param_list)
                super().__init__(param_list, **kwargs)

        monkeypatch.setattr(torch.optim, "AdamW", CapturingAdamW)

        repair_layers(
            model=model, tokenizer=tok, texts_train=texts["train"],
            layers=cfg.layers, steps=1, lr=cfg.lr,
            warmup=0, weight_decay=cfg.weight_decay,
            max_len=cfg.max_len, device=cfg.device,
        )

        expected_ids: set[int] = set()
        for li in cfg.layers:
            for p in model.model.layers[li].mlp.parameters():
                expected_ids.add(id(p))

        assert captured_param_ids == expected_ids, (
            f"Optimizer params must exactly match target MLP params.\n"
            f"  Extra (not target MLP): {len(captured_param_ids - expected_ids)}\n"
            f"  Missing (target MLP not in opt): {len(expected_ids - captured_param_ids)}"
        )


# ═════════════════════════════════════════════════════════════════════════
# 4. EVAL: use_cache=False CONTRACT
# ═════════════════════════════════════════════════════════════════════════


class TestEvalContract:
    """Every model.forward call inside eval_perplexity must receive
    use_cache=False.
    """

    def test_every_forward_call_uses_no_cache(self, engine_state, monkeypatch):
        from pnn_compiler.eval import eval_perplexity

        model = engine_state["model"]
        tok = engine_state["tok"]
        texts = engine_state["texts"]
        cfg = engine_state["cfg"]

        recorded_use_cache: list[object] = []
        real_forward = model.forward

        def recording_forward(*args, **kwargs):
            recorded_use_cache.append(kwargs.get("use_cache", "NOT_PASSED"))
            return real_forward(*args, **kwargs)

        monkeypatch.setattr(model, "forward", recording_forward)

        eval_perplexity(
            model, tok, texts["eval_w2"][:2],
            cfg.max_len, cfg.device, cfg.max_eval_tokens,
        )

        assert len(recorded_use_cache) > 0, (
            "model.forward was never called during eval_perplexity"
        )
        for i, val in enumerate(recorded_use_cache):
            assert val is False, (
                f"Forward call {i}: use_cache={val!r}, expected False"
            )
