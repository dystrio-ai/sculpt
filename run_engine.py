"""Minimal entry point — the engine as you've been running it."""

from pnn_compiler.config import EngineConfig
from pnn_compiler.pipeline import run_pipeline

cfg = EngineConfig(
    model_id="Qwen/Qwen2-0.5B",
    layers=[3],
    keep_frac=0.5,
    block_size=128,
    repair_steps=2000,
    dtype="bf16",
    device="cuda",
)
print(run_pipeline(cfg))
