# Dystrio Sculpt

Structural compression for transformer LLMs. Sculpt removes redundant neurons
from feed-forward blocks (dense models) and prunes redundant experts from
Mixture-of-Experts layers, then repairs quality with knowledge distillation.
The output is a smaller, faster model that loads with standard HuggingFace
`AutoModelForCausalLM.from_pretrained()` — no custom code, no special runtime.

## Highlights

- **One command** — `dystrio sculpt --model-id <hf_model>` handles everything
- **Standard output** — HuggingFace checkpoints that work with vLLM, TGI, llama.cpp, GGUF, Ollama
- **LoRA-ready** — sculpted models are standard architectures, fine-tune with PEFT/Unsloth/Axolotl
- **Stackable** — prune → LoRA fine-tune → quantize for compounding 5-6x size reduction
- **Workload-adaptive** — prune for your domain so downstream fine-tuning needs less correction
- **Dense + MoE** — prunes SwiGLU neurons (dense) or entire experts (MoE)
- **Quality-aware** — Thompson Sampling search finds the fastest model within your quality budget

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

Additional MoE families (for example Mixtral) are **supported by the CLI**; the table
above lists the pre-sculpted checkpoints we publish on HuggingFace today.

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

# One command — finds the best compression automatically
dystrio sculpt --model-id Qwen/Qwen2.5-3B-Instruct

# MoE model
dystrio sculpt --model-id allenai/OLMoE-1B-7B-0924

# Code-specialized workload
dystrio sculpt --model-id meta-llama/Llama-3.1-8B-Instruct --workload code_v1
```

Defaults: distillation ON, `general_v2` workload, 1 frontier point. The output model
is ready to load with HuggingFace or convert to GGUF — no special runtime needed.

### Customizing

```bash
# Emit 4 Pareto-optimal points instead of 1
dystrio sculpt --model-id <model> --frontier 4

# Disable distillation (faster, lower quality)
dystrio sculpt --model-id <model> --no-distill

# Skip search, evaluate specific compression levels
dystrio sculpt --model-id <model> --keep-fracs "0.90,0.80,0.70"

# Custom workload — provide your own calibration data
dystrio sculpt --model-id <model> \
  --workload none \
  --calib-dataset your-org/your-dataset \
  --calib-config default \
  --calib-split train \
  --calib-text-field text

# Raw wikitext-only calibration (no workload mixture)
dystrio sculpt --model-id <model> --workload none
```

Output:

Each run writes one directory per frontier point. Names come from the quality search
(see `src/dystrio_sculpt/search.py`): **`frontier_<i>_<tier>`** where `<tier>` is one of
`default`, `production`, `throughput`, `experimental`, `frontier` (or a generic
`pointN` label if a point sits above the quality ceiling).

With **`--frontier 1`** (the default), you typically get **`frontier_0_production`** if the
chosen point is within the quality ceiling, otherwise **`frontier_0_default`**. List
`sculpt_out/` after a run to see the exact name on disk.

```
sculpt_out/
  frontier_0_production/    # or frontier_0_default — see above
    model/                  # HuggingFace checkpoint (config.json, safetensors, tokenizer)
    metrics.json            # PPL, throughput, speedup, memory, risk score
    compile_report.json
    manifest.json           # Full reproducibility record
  summary.csv
```

Load a sculpted model (adjust the folder to match your run):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("sculpt_out/frontier_0_production/model")
tokenizer = AutoTokenizer.from_pretrained("sculpt_out/frontier_0_production/model")
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
| Mixtral | Mixtral-8x7B, Mixtral-8x22B | MoE (CLI; no pre-sculpted HF repo yet) |
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
dystrio sculpt --model-id <model> --keep-fracs "0.90,0.85,0.75"
```

## Benchmarking

Benchmark baseline vs. sculpted models across workloads:

```bash
dystrio bench \
  --models org/baseline org/sculpted-production \
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
  --frontier INTEGER              Points to emit [default: 1]
  --max-ppl-multiplier FLOAT      Quality ceiling [default: 2.0]
  --keep-fracs TEXT               Skip search, evaluate these (comma-separated)
  --workload TEXT                 Workload preset [default: general_v2]
  --distill / --no-distill        Knowledge distillation [default: on]
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

## Quality Gates

Sculpt classifies compressed models as **safe** or **over-ceiling** using an OR gate:

A model is safe if **either** condition holds:
1. **Downstream accuracy**: retains >= 95% of baseline accuracy on a fast
   multi-task probe (MMLU, HellaSwag, ARC, BoolQ)
2. **Perplexity ratio**: within the adaptive ceiling (scales with compression —
   tighter for light pruning, relaxed for aggressive compression, matching
   published SOTA operating points)

Only "safe" models get published tier labels (production, throughput, etc.).
Over-ceiling models are still emitted but labeled generically. To tighten quality:

```bash
# Stricter: model must be within 1.3x baseline perplexity
dystrio sculpt --model-id <model> --max-ppl-multiplier 1.3

# Stricter: must retain 98% of downstream accuracy
dystrio sculpt --model-id <model> --downstream-threshold 0.98
```

## The Full Stack: Sculpt → Fine-Tune → Quantize → Deploy

Sculpt is step one of a complete model optimization pipeline. Because the output is a
**standard HuggingFace checkpoint** with physically smaller weight matrices, every
downstream tool in the ecosystem works on it — LoRA fine-tuning, quantization, GGUF
conversion, serving frameworks. No adapters, no custom inference code, no sparse runtime.

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. SCULPT          2. FINE-TUNE       3. QUANTIZE     4. DEPLOY   │
│                                                                     │
│  dystrio sculpt     LoRA / QLoRA       GPTQ / AWQ      vLLM       │
│  --workload code    (PEFT, Unsloth,    GGUF Q4_K_M     TGI        │
│                      Axolotl)                           llama.cpp  │
│                                                         Ollama     │
│  Removes redundant  Adapts to your     4-bit weights   Runs on    │
│  neurons for your   specific task      on smaller       any stack  │
│  workload                              matrices                    │
│                                                                     │
│  7B → 5.6B          Cheaper: smaller   5.6B → ~1.5GB   Standard   │
│  (20% smaller)      base = less VRAM   (GGUF Q4)       model      │
└─────────────────────────────────────────────────────────────────────┘
```

### Why workload-aligned pruning makes fine-tuning cheaper

When you sculpt with your workload data, the pruning preserves the neurons your task
needs and the distillation repairs domain-relevant capabilities. The model arrives
**pre-aligned to your domain** before fine-tuning even starts.

This means LoRA has less to fix: fewer training steps, smaller adapter rank, less
compute. Compare this to fine-tuning a generically pruned model, where LoRA has to
recover capabilities that pruning blindly destroyed.

```bash
# Step 1: Sculpt with your workload
dystrio sculpt --model-id meta-llama/Llama-3.1-8B-Instruct \
  --workload none \
  --calib-dataset your-org/customer-support-logs \
  --calib-text-field message

# Step 2: LoRA fine-tune the sculpted model (any tool works)
# The model is a standard HF checkpoint — PEFT, Unsloth, Axolotl all work
python -m peft.train \
  --model_name sculpt_out/frontier_0_production/model \
  --dataset your-org/training-data \
  --lora_r 16

# Step 3: Quantize for deployment
python -m awq.entry --model_path sculpt_out/frontier_0_production/model --w_bit 4
# or: convert to GGUF for llama.cpp / Ollama
python convert_hf_to_gguf.py sculpt_out/frontier_0_production/model --outfile model.gguf
llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M
```

A sculpted + LoRA fine-tuned + quantized model can be **5-6x smaller** than the original,
run on a laptop, and perform **better on your task** than the full-size base model —
because every step in the pipeline was optimized for your workload.

### vs. other pruning methods

| | Sculpt | Unstructured (Wanda, SparseGPT) | LLM-Pruner |
|---|---|---|---|
| Output format | Standard HF checkpoint | Same-size model with zeros | Standard HF checkpoint |
| Speedup without special runtime | Yes | No (needs sparse kernels) | Yes |
| LoRA fine-tune after | Works perfectly | Sparse matrices break LoRA | Works |
| Stack with quantization | Yes (orthogonal) | Zeros conflict with quantization | Yes |
| Workload-adaptive | Yes | No | No |
| GGUF / llama.cpp / Ollama | Yes | No | Yes |
| Setup | One command | One command | Multi-step (prune + LoRA recovery) |

## Run Locally

Sculpt runs on consumer GPUs for smaller models:

| Model Size | Min GPU VRAM | Example GPU | Approx Time |
|------------|-------------|-------------|-------------|
| 1-3B | 8 GB | RTX 3070/4070 | 15-30 min |
| 3-8B | 16-24 GB | RTX 3090/4090 | 30-90 min |
| 7-14B | 40-48 GB | A6000 / A100 | 1-3 hours |
| 70B+ | 80 GB+ | H100 / multi-GPU | 4-8 hours |

```bash
# On a 3090 (24GB) — sculpt a 3B model locally
dystrio sculpt --model-id Qwen/Qwen2.5-3B-Instruct

# On a 4090 — sculpt a 7B model
dystrio sculpt --model-id mistralai/Mistral-7B-Instruct-v0.3

# Reduce memory: disable distillation (faster but lower quality)
dystrio sculpt --model-id <model> --no-distill
```

The output is a standard HuggingFace checkpoint. Convert to GGUF and run on CPU/laptop:

```bash
python convert_hf_to_gguf.py sculpt_out/frontier_0_production/model --outfile model.gguf
llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- CUDA GPU (8–24GB is enough for small models; 7B+ with distillation is more comfortable on 40GB+ — see **Run Locally**)

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
