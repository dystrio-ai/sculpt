# PNN Compiler

Demand-aware FFN block compression + fast repair for HF Transformers.
Targets Qwen2-style SwiGLU MLPs (`gate_proj` / `up_proj` / `down_proj` + `act_fn`).

## How it works

1. **Calibrate** — forward-hook on each target MLP collects `mean( |act_fn(gate(x)) · up(x)| )` per neuron
2. **Block select** — partition neurons into contiguous blocks, rank by summed importance, keep top `keep_frac`
3. **Compile** — replace `gate_proj`, `up_proj`, `down_proj` with physically smaller `nn.Linear` layers
4. **Repair** — freeze everything except compressed MLPs, train with CE loss + cosine LR
5. **Eval** — token-level perplexity on wikitext-2 test and wikitext-103 validation
6. **Bench** — prefill tokens/sec on padded batch with CUDA sync

## Quickstart

```bash
cd pnn_compiler
pip install -e .

# Full pipeline (the engine):
python run_engine.py

# Or via CLI:
pnn run-all

# Individual stages:
pnn calibrate
pnn compile
pnn repair
pnn eval
pnn bench
```

## Programmatic usage

```python
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
results = run_pipeline(cfg)
```

## Configuration

Defaults in `configs/default.yaml`. Override via CLI `--set`:

```bash
pnn run-all --set keep_frac=0.6 --set lr=1e-4
pnn run-all --set layers=[3,5] --set repair_steps=3000
pnn run-all --config my_config.yaml
```

`EngineConfig` fields:

| Field              | Default          | Description                          |
|--------------------|------------------|--------------------------------------|
| `model_id`         | Qwen/Qwen2-0.5B | HF model name                        |
| `layers`           | [3]              | Transformer layer indices to compress |
| `block_size`       | 128              | Contiguous neuron block size          |
| `keep_frac`        | 0.50             | Fraction of FFN blocks to keep        |
| `max_len`          | 256              | Sequence length for all stages        |
| `n_texts_cal`      | 400              | Calibration texts                     |
| `n_texts_train`    | 2500             | Repair training texts                 |
| `n_texts_eval`     | 300              | Eval texts per split                  |
| `max_eval_tokens`  | 40000            | Token budget for perplexity eval      |
| `repair_steps`     | 2000             | Repair training steps                 |
| `lr`               | 3e-4             | Repair learning rate                  |
| `warmup`           | 100              | LR warmup steps                       |
| `weight_decay`     | 0.01             | AdamW weight decay                    |
| `bench_texts`      | 200              | Texts for throughput benchmark         |
| `bench_warmup_iters`| 20              | Warmup forward passes                 |
| `bench_iters`      | 80               | Timed forward passes                  |
| `device`           | cuda             | cuda or cpu                           |
| `dtype`            | bf16             | bf16, fp16, or fp32                   |
| `seed`             | 0                | Random seed                           |
| `allow_tf32`       | true             | Enable TF32 matmul on Ampere+         |

## Experiment harness

```bash
# Full experiment matrix (OOD eval + ablation baselines)
python experiments/multilayer_experiment.py

# Skip ablation baselines for faster iteration
python experiments/multilayer_experiment.py --skip-ablations

# With gradient accumulation
python experiments/multilayer_experiment.py --grad-accum-steps 4

# With optional vLLM serving benchmark (requires vLLM)
python experiments/multilayer_experiment.py --enable-vllm
```

## Optional: vLLM serving benchmark

```bash
pip install vllm

# Standalone
python experiments/vllm_benchmark.py --model-path Qwen/Qwen2-0.5B

# Integrated into harness
python experiments/multilayer_experiment.py --enable-vllm
```

## Hardware

- **GPU-first**: defaults to bf16 on CUDA (A100-optimised), TF32 enabled
- `torch.cuda.synchronize()` in bench for accurate timing
- Repair is single-text forward/backward (no batching), matches the original engine
