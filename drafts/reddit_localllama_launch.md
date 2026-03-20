# We structurally compressed 6 LLM families and benchmarked every tier. Here's the efficiency frontier.

**TL;DR:** We physically shrink MLP dimensions inside transformers — not quantization, not sparsity — producing standard dense HuggingFace checkpoints that work with vLLM, llama.cpp, and anything that calls `from_pretrained`. We ran ARC-Challenge, HellaSwag, MMLU, and TruthfulQA across 6 model families at up to 5 compression tiers each. The result is an efficiency frontier showing exactly where quality degrades for each architecture.

---

## The efficiency frontier

**[Image 1: Efficiency Frontier Chart]**

Each curve is a model family compressed from 0% to ~47% MLP width reduction. Y-axis is average accuracy retention vs. the uncompressed baseline.

The story in one sentence: **all architectures follow smooth degradation — but they degrade at very different rates.**

---

## What jumps out from the data

**Gemma 2B is the compression champion.** It retains 92% of average accuracy at 14% MLP removal and still holds 77% at 29% removal. No other architecture in our test comes close at equivalent compression.

**Larger models aren't always more compressible.** Llama 3.1 8B drops to 85% retention at just 14% MLP removal — worse than the 2–3B class models at the same compression level. Bigger doesn't mean more redundant.

**TruthfulQA barely moves.** Across all models, factual consistency (TruthfulQA) is the last thing to degrade. Gemma holds 95% TruthfulQA retention even at 47% MLP removal. The capability that breaks first is factual recall (MMLU), followed by reasoning (ARC-C).

**[Image 2: Per-Benchmark Frontier Chart]**

The per-benchmark view makes this clear — MMLU drops steeply while TruthfulQA stays nearly flat. If your use case is conversational and doesn't require deep factual recall, you can compress harder than the benchmarks initially suggest.

---

## Full benchmark table

| Model | Tier | MLP Removed | ARC-C | HellaSwag | MMLU | TruthfulQA | Avg Retention |
|-------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gemma 2 2B** | baseline | 0% | 0.509 | 0.538 | 0.569 | 0.532 | 100% |
| | default | 14% | 0.439 | 0.503 | 0.502 | 0.524 | 92% |
| | production | 21% | 0.396 | 0.476 | 0.429 | 0.506 | 84% |
| | throughput | 29% | 0.345 | 0.435 | 0.368 | 0.499 | 77% |
| | experimental | 47% | 0.265 | 0.358 | 0.230 | 0.504 | 63% |
| **Llama 3.2 3B** | baseline | 0% | 0.436 | 0.533 | 0.622 | 0.514 | 100% |
| | default | 11% | 0.374 | 0.497 | 0.527 | 0.463 | 88% |
| | production | 17% | 0.345 | 0.477 | 0.488 | 0.439 | 83% |
| | throughput | 25% | 0.352 | 0.439 | 0.395 | 0.418 | 76% |
| | experimental | 39% | 0.279 | 0.375 | 0.305 | 0.442 | 67% |
| **Qwen 2.5 3B** | baseline | 0% | 0.457 | 0.564 | 0.655 | 0.587 | 100% |
| | default | 12% | 0.427 | 0.505 | 0.579 | 0.515 | 89% |
| | production | 17% | 0.381 | 0.473 | 0.550 | 0.523 | 85% |
| | throughput | 24% | 0.329 | 0.439 | 0.484 | 0.505 | 78% |
| | experimental | 35% | 0.294 | 0.386 | 0.408 | 0.453 | 68% |
| **Mistral 7B** | baseline | 0% | 0.579 | 0.657 | 0.598 | 0.594 | 100% |
| | default | 14% | 0.497 | 0.597 | 0.496 | 0.538 | 88% |
| | production | 21% | 0.457 | 0.566 | 0.481 | 0.502 | 83% |
| | throughput | 30% | 0.380 | 0.508 | 0.398 | 0.486 | 73% |
| | experimental | 38% | 0.335 | 0.480 | 0.382 | 0.505 | 70% |
| **Llama 3.1 8B** | baseline | 0% | 0.536 | 0.598 | 0.684 | 0.546 | 100% |
| | default | 14% | 0.428 | 0.542 | 0.559 | 0.482 | 85% |
| | production | 30% | 0.402 | 0.474 | 0.408 | 0.487 | 75% |
| | throughput | 38% | 0.344 | 0.435 | 0.328 | 0.482 | 67% |
| | experimental | 39% | 0.347 | 0.431 | 0.324 | 0.481 | 67% |
| **Qwen 2.5 7B** | baseline | 0% | 0.528 | 0.620 | 0.718 | 0.648 | 100% |
| | default | 12% | 0.468 | 0.565 | 0.651 | 0.546 | 89% |
| | production | 34% | 0.375 | 0.446 | 0.478 | 0.446 | 69% |
| | throughput | 40% | 0.357 | 0.432 | 0.410 | 0.452 | 66% |

All benchmarks: A100-80GB, bf16, lm-eval harness, zero-shot.

---

## How it works

This is structural MLP width reduction. For each layer in the model, we:

1. **Analyze** activation patterns and operator sensitivity to identify which blocks of the MLP intermediate dimension carry the least information
2. **Remove** those blocks — physically shrinking the gate, up, and down projection matrices
3. **Repair** with short calibration fine-tuning to recover quality

The output is a model with physically smaller weight matrices. For example, Gemma 2B's `intermediate_size` drops from 9216 to 7936 at the default tier. It's still a standard dense model — just narrower.

No distillation, no teacher model, no custom kernels. The entire process takes ~5 minutes per tier on a single A100.

---

## Why this matters

Most compression work shows you one operating point. We think the more useful thing is the full tradeoff curve — because the right compression level depends on your deployment constraints, and different architectures respond very differently to the same technique.

If you're choosing between Gemma and Llama for an edge deployment, this data tells you Gemma gives you significantly more room to compress before quality degrades. If you're deploying Mistral 7B and can tolerate a ~12pp accuracy drop, you can cut 30% of the MLP width and get real inference speedups.

**Compression compounds with quantization.** These are dense checkpoints, so you can apply AWQ, GPTQ, or GGUF quantization on top. A structurally compressed + quantized model is smaller than either technique alone.

---

## Compatibility

These are standard HuggingFace checkpoints. No special code required:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "dystrio/gemma-2-2b-it-sculpt-default",
    torch_dtype="bfloat16", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "dystrio/gemma-2-2b-it-sculpt-default"
)
```

Works with:
- **vLLM, TGI, SGLang** — standard serving
- **llama.cpp** — convert to GGUF normally
- **AWQ, GPTQ** — quantize on top for further compression
- Anything that calls `from_pretrained`

No runtime changes, no custom CUDA kernels.

---

## What's next

We're expanding architecture coverage, improving the repair phase, and building toward automatically finding the optimal compression point for a given quality/latency constraint. The efficiency frontier is different for every model — the goal is to map it automatically.

All models available on HuggingFace: [https://huggingface.co/dystrio](https://huggingface.co/dystrio)

Happy to answer questions about methodology, benchmarks, or specific models.
