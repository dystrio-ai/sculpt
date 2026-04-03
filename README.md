# Dystrio Sculpt

Structural compression for transformer LLMs. Sculpt removes redundant neurons
from feed-forward blocks (dense models) and prunes redundant experts from
Mixture-of-Experts layers, then repairs quality with knowledge distillation.
The output is a smaller, faster model that loads with standard HuggingFace
`AutoModelForCausalLM.from_pretrained()` — no custom code, no special runtime.

## Highlights

- **One command** — `dystrio sculpt --model-id <hf_model>` handles everything
- **Dense + MoE** — prunes SwiGLU neurons (dense) or entire experts (MoE)
- **Drop-in output** — standard HuggingFace checkpoints, works with vLLM, TGI, llama.cpp, GGUF
- **Quality-aware** — Thompson Sampling search finds the fastest model within your quality budget
- **Workload presets** — calibrate for general, code, or custom workloads
- **Stackable** — output is structurally pruned, orthogonal to quantization (GPTQ, AWQ, GGUF)

## Published Models

Pre-sculpted models are available on [HuggingFace](https://huggingface.co/dystrio):

| Model | Type | Tier | Memory Reduction | Avg Benchmark Delta |
|-------|------|------|-----------------|-------------------|
| Mistral-7B-Instruct-v0.3 | Dense | Production | 11% | -0.2% |
| Mistral-7B-Instruct-v0.3 | Dense | Throughput | 18% | -1.4% |
| Llama-3.1-8B-Instruct | Dense | Production | 12% | -0.5% |
| Llama-3.2-3B-Instruct | Dense | Production | 10% | -0.3% |
| Qwen2.5-3B-Instruct | Dense | Production | 11% | -0.4% |
| gemma-2-2b-it | Dense | Production | 9% | -0.1% |
| OLMoE-1B-7B-0924 | MoE | Balanced | 9% | +0.04% |

All models evaluated with [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness)
on ARC-Challenge, HellaSwag, MMLU, and TruthfulQA. Full results in each model card.

## Workload-Aware Compression

Structural redundancy is not absolute — it depends on what you use the model for.
A neuron that's dead weight for code generation may be critical for math reasoning.
Sculpt uses workload-specific calibration data so the Physarum selector identifies
neurons and experts that are redundant **for your use case**.

```bash
# Code-optimized: finds more redundancy by deprioritizing general-knowledge neurons
dystrio sculpt --model-id meta-llama/Llama-3.1-8B-Instruct --workload code_v1

# General: preserves broad capability, more conservative compression
dystrio sculpt --model-id meta-llama/Llama-3.1-8B-Instruct --workload general_v2
```

Different workloads produce different models:
- **Code workload** finds more structural redundancy because general-knowledge
  neurons idle during code processing — quality loss concentrates in off-target
  benchmarks (MMLU) while code benchmarks (HumanEval, MBPP) are preserved.
- **General workload** compresses more conservatively to preserve all capabilities evenly.
- **MoE models** show the effect most dramatically: different workloads activate
  different expert subsets, so entire experts can be safely removed when they don't
  contribute to the target workload.

Run the full showcase study to see workload-aware divergence on your hardware:

```bash
bash scripts/workload_showcase.sh
```

## Quick Start

```bash
pip install -e .

# Dense model — search for fastest safe compression
dystrio sculpt --model-id Qwen/Qwen2-0.5B --outdir sculpt_out --frontier 4

# MoE model — drop 10% of experts
dystrio sculpt --model-id allenai/OLMoE-1B-7B-0924 --keep-fracs 0.90 --outdir sculpt_moe

# Code-specialized workload
dystrio sculpt --model-id mistralai/Mixtral-8x7B-v0.1 --workload code_v1 --outdir sculpt_code
```

Output:

```
sculpt_out/
  frontier_0_conservative/
    model/              # HuggingFace checkpoint (config.json, safetensors, tokenizer)
    metrics.json        # PPL, throughput, speedup, memory, risk score
    compile_report.json
    manifest.json       # Full reproducibility record
  frontier_1_balanced/
    ...
  summary.csv
```

Load a sculpted model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("sculpt_out/frontier_0_conservative/model")
tokenizer = AutoTokenizer.from_pretrained("sculpt_out/frontier_0_conservative/model")
```

## How It Works

1. **Prescan** — Measure operator sensitivity and inter-block coupling geometry
   across all FFN layers (or expert routing statistics for MoE)
2. **Risk scoring** — Compute structural risk per layer from sensitivity,
   coupling concentration, and spectrum rank
3. **Select** — Rank neurons/experts using a Physarum-inspired diversity penalty
   on top of operator-fidelity scores
4. **Compress** — Physically remove pruned neurons from weight matrices (dense)
   or drop + merge experts (MoE). Staged by compressibility: safest layers first
5. **Repair** — Knowledge distillation fine-tuning with cosine LR, regression
   tripwires, and best-checkpoint restore
6. **Validate** — Reload saved model, verify shapes and no NaN/Inf

### Dense Models

Sculpt removes neurons from `gate_proj`, `up_proj`, and `down_proj` weight
matrices. The output has a uniform reduced `intermediate_size` across all
layers — a standard HuggingFace config change with no per-layer width vectors.

### MoE Models

For Mixture-of-Experts architectures, Sculpt drops entire experts per layer.
Dropped experts are merged into their most-coupled surviving neighbor (weighted
by routing correlation) before removal. The router is patched to redistribute
load across remaining experts.

## Supported Architectures

| Family | Models | Mode |
|--------|--------|------|
| Llama | Llama 2, Llama 3, Llama 3.1, Llama 3.2 | Dense |
| Mistral | Mistral 7B, Mistral Nemo | Dense |
| Qwen | Qwen2, Qwen2.5 | Dense |
| Gemma | Gemma 2 | Dense |
| Phi | Phi-3, Phi-3.5 | Dense |
| Mixtral | Mixtral-8x7B, Mixtral-8x22B | MoE |
| OLMoE | OLMoE-1B-7B | MoE |
| Starcoder | Starcoder2-15B | Dense |
| MiniCPM | MiniCPM-o | Dense |

Any decoder-only transformer with SwiGLU FFN blocks (`gate_proj`/`up_proj`/`down_proj`)
should work out of the box. Use `dystrio factory fingerprint --model-id <model>`
to check compatibility.

## Workload Presets

Sculpt calibrates on task-specific data mixtures to preserve what matters for
your deployment:

| Preset | Focus | Sources |
|--------|-------|---------|
| `general_v2` | Balanced general capability | WikiText, MMLU, OpenHermes, HellaSwag, GSM8K, OpenOrca |
| `code_v1` | Code generation | CodeAlpaca, MBPP, HumanEval, WikiText, OpenHermes |

```bash
# General workload (default)
dystrio sculpt --model-id <model> --workload general_v2

# Code-focused workload
dystrio sculpt --model-id <model> --workload code_v1
```

## Search Behavior

By default Sculpt uses Thompson Sampling to search for the fastest model under
a quality ceiling (`--max-ppl-multiplier`, default 2.0x baseline PPL). The risk
score from prescan adapts the search bracket:

| Risk | Starting Bracket | Strategy |
|------|-----------------|----------|
| Low (< 0.35) | 0.85 - 0.55 | Aggressive — model tolerates pruning |
| Medium | 0.88 - 0.62 | Standard sweep |
| High (> 0.65) | 0.92 - 0.72 | Conservative — tight coupling |

To skip search and evaluate specific compression levels:

```bash
dystrio sculpt --model-id <model> --keep-fracs 0.90 0.85 0.75
```

## Benchmarking

Benchmark baseline vs. sculpted models across workloads:

```bash
dystrio bench \
  --models org/baseline org/sculpted-balanced \
  --workloads wikitext chat rag code \
  --prompts-dir prompts/ \
  --outdir bench_out
```

Generate plots and model card snippets:

```bash
dystrio bench-report --results-dir bench_out/results --outdir bench_out/report
```

Audit for publishability:

```bash
dystrio bench-audit --bench-out bench_out
```

## CLI Reference

```
Global flags (all commands):
  --quiet / -q      Minimal output
  --verbose / -v    Full debug output

dystrio sculpt [OPTIONS]
  --model-id TEXT                 HuggingFace model ID [required]
  --outdir TEXT                   Output directory [default: sculpt_out]
  --frontier INTEGER              Points to emit [default: 4]
  --max-ppl-multiplier FLOAT      Quality ceiling [default: 2.0]
  --keep-fracs FLOAT ...          Skip search, evaluate these fractions
  --workload TEXT                 Workload preset [default: general_v2]
  --target-prefill-speedup FLOAT  Min prefill speedup
  --max-compile-hours FLOAT       Time budget in hours
  --downstream-threshold FLOAT    Min downstream accuracy to accept
  --deterministic                 Bitwise-reproducible builds
  --push-dataset / --no-push-dataset  Push results to HF dataset
  --save-prescan / --no-save-prescan  Save prescan analysis JSON
  --policy TEXT                   Override auto-selected repair policy

  Calibration overrides:
  --calib-dataset TEXT            HF dataset [default: wikitext]
  --calib-config TEXT             Dataset config [default: wikitext-2-raw-v1]
  --calib-split TEXT              Dataset split [default: train]
  --calib-text-field TEXT         Text column [default: text]
  --calib-num-samples INTEGER     Max calibration samples
  --calib-seq-len INTEGER         Sequence length
  --calib-seed INTEGER            Sampling seed [default: 0]

dystrio bench [OPTIONS]
  --models TEXT ...               Models to benchmark [required]
  --workloads TEXT ...            Workloads [default: wikitext chat rag code]
  --prompts-dir TEXT              JSONL prompt packs directory
  --outdir TEXT                   Output [default: bench_out]
  --dtype TEXT                    bf16|fp16|fp32 [default: bf16]
  --device TEXT                   cuda|cpu [default: cuda]

dystrio bench-report [OPTIONS]
  --results-dir TEXT              Path to results/ [required]
  --outdir TEXT                   Report output [default: bench_out/report]

dystrio bench-audit [OPTIONS]
  --bench-out TEXT                Bench output dir [required]

dystrio factory fingerprint --model-id TEXT
  Check architecture support for a model.

dystrio factory run [OPTIONS]
  Orchestrated compile + bench + publish pipeline.
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- CUDA GPU (A100 80GB recommended for 7B+ models)

```bash
# Install with dev tools (pytest, ruff)
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Deterministic Builds

For bitwise-reproducible results:

```bash
dystrio sculpt --model-id <model> --deterministic
```

Seeds all RNGs (Python, NumPy, PyTorch, CUDA), disables TF32 and non-deterministic
cuDNN algorithms, and uses isolated random state in the structural selector.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache 2.0](LICENSE)
