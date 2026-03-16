# Every LLM has an efficiency frontier. We mapped it across 6 architectures.

We've been building a structural compression system for transformer models. Not quantization — physically rewriting MLP dimensions layer-by-layer and outputting standard dense HuggingFace checkpoints. `from_pretrained` and go.

We ran downstream benchmarks (ARC-Challenge, HellaSwag, MMLU, TruthfulQA) across 6 model families at 4 compression levels each. The result is an efficiency frontier — the tradeoff curve between compression and accuracy for each architecture.

**[Efficiency Frontier Chart]**

## What the chart shows

Each curve is a different model family compressed from 0% to ~40% MLP width reduction. The Y-axis is average accuracy retention relative to the uncompressed baseline.

Key findings:
- **All models follow smooth, predictable degradation curves.** No cliff edges — you can pick your tradeoff point.
- **Architectures compress differently.** Gemma 2B retains 91% of accuracy at 14% compression. Llama 8B retains 83% at the same level. Same technique, different resilience.
- **TruthfulQA is remarkably stable.** Gemma holds 95%+ TruthfulQA accuracy even at 47% MLP removal. Factual recall (MMLU) degrades fastest.
- **The default tier (least compressed) trades ~5-8pp avg accuracy for 9-14% smaller checkpoints.** Whether that's acceptable depends on your use case.

## Full benchmark table

| Model | Tier | MLP Removed | ARC-C | HellaSwag | MMLU | TruthfulQA |
|-------|------|:---:|:---:|:---:|:---:|:---:|
| **Gemma 2 2B** | baseline | 0% | 0.509 | 0.538 | 0.569 | 0.532 |
| | default | 14% | 0.439 | 0.503 | 0.502 | 0.524 |
| | production | 21% | 0.396 | 0.476 | 0.429 | 0.506 |
| | throughput | 29% | 0.345 | 0.435 | 0.368 | 0.499 |
| | experimental | 47% | 0.265 | 0.358 | 0.230 | 0.504 |
| **Llama 3.2 3B** | baseline | 0% | 0.436 | 0.533 | 0.622 | 0.514 |
| | default | 11% | 0.374 | 0.497 | 0.527 | 0.463 |
| | production | 17% | 0.345 | 0.477 | 0.488 | 0.439 |
| | throughput | 25% | 0.352 | 0.439 | 0.395 | 0.418 |
| | experimental | 39% | 0.279 | 0.375 | 0.305 | 0.442 |
| **Qwen 2.5 3B** | baseline | 0% | 0.457 | 0.564 | 0.655 | 0.587 |
| | default | 12% | 0.427 | 0.505 | 0.579 | 0.515 |
| | production | 17% | 0.381 | 0.473 | 0.550 | 0.523 |
| | throughput | 24% | 0.329 | 0.439 | 0.484 | 0.505 |
| | experimental | 35% | 0.294 | 0.386 | 0.408 | 0.453 |
| **Mistral 7B** | baseline | 0% | 0.579 | 0.657 | 0.598 | 0.594 |
| | default | 14% | 0.497 | 0.597 | 0.496 | 0.538 |
| | production | 21% | 0.457 | 0.566 | 0.481 | 0.502 |
| | throughput | 30% | 0.380 | 0.508 | 0.398 | 0.486 |
| | experimental | 38% | 0.335 | 0.480 | 0.382 | 0.505 |
| **Llama 3.1 8B** | baseline | 0% | 0.536 | 0.598 | 0.684 | 0.546 |
| | default | 14% | 0.428 | 0.542 | 0.559 | 0.482 |
| | production | 30% | 0.402 | 0.474 | 0.408 | 0.487 |
| | throughput | 38% | 0.344 | 0.435 | 0.328 | 0.482 |
| | experimental | 39% | 0.347 | 0.431 | 0.324 | 0.481 |
| **Qwen 2.5 7B** | baseline | 0% | 0.528 | 0.620 | 0.718 | 0.648 |
| | default | 12% | 0.468 | 0.565 | 0.651 | 0.546 |
| | production | 34% | 0.375 | 0.446 | 0.478 | 0.446 |
| | throughput | 40% | 0.357 | 0.432 | 0.410 | 0.452 |

All benchmarks: A100-80GB, bf16, lm-eval harness, zero-shot.

## What this is

Structural MLP width reduction. For each layer, we analyze activation patterns to identify which blocks of the intermediate dimension carry the least information, remove them, and repair with calibration. No distillation, no teacher model, no custom kernels.

The output has physically smaller weight matrices (e.g., `intermediate_size` drops from 9216 to 7936). Standard dense checkpoint that loads normally in any framework.

## Why this matters

Most compression work shows you one point: "here's the compressed model, here's the quality." We think the more useful thing is the full curve — showing exactly where the tradeoff gets steep for each architecture so you can pick the point that fits your deployment constraints.

Different architectures respond differently to the same compression. If you're choosing between Gemma and Llama for an edge deployment, this data tells you which one gives you more room to compress.

## Compatibility

These are standard HuggingFace checkpoints:

- vLLM, TGI, SGLang
- llama.cpp (convert to GGUF normally)
- Quantization stacks on top: AWQ, GPTQ, GGUF — compression compounds
- Any code that calls `from_pretrained`

No runtime changes, no custom CUDA kernels, no special inference code.

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

## What's next

We're expanding coverage to more architectures and building toward an automated system that finds the optimal compression point given your quality and latency constraints. The efficiency frontier is different for every model — the goal is to map it automatically.

Models: [https://huggingface.co/dystrio](https://huggingface.co/dystrio)

Happy to answer questions about methodology, benchmarks, or specific models.
