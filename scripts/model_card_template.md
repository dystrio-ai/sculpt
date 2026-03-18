---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language:
  - en
base_model: {BASE_MODEL_ID}
tags:
  - dystrio
  - sculpt
  - pruned
  - compressed
  - efficient
  - dense
  - runtime-agnostic
  - no-custom-kernels
  - hf-drop-in
  - drop-in-replacement
  - smaller
  - faster
  - {ARCH_TAG}
datasets:
  - wikitext
model-index:
  - name: Dystrio Sculpt ({MODEL_SHORT} {TIER_DISPLAY})
    results:
      - task:
          type: text-generation
        dataset:
          name: WikiText-103 (validation)
          type: wikitext
        metrics:
          - name: perplexity
            type: perplexity
            value: {PPL_WIKITEXT}
          - name: ppl_ratio
            type: ppl_ratio
            value: {PPL_RATIO}
      - task:
          type: text-generation
        dataset:
          name: Chat (200 prompts)
          type: custom
        metrics:
          - name: prefill_tokens_per_sec
            type: throughput
            value: {CHAT_PREFILL_TPS}
          - name: ttft_p95_ms
            type: latency
            value: {CHAT_TTFT_P95}
---

# {MODEL_NAME}

> **{SIZE_CUT}% smaller, {SPEED_CLAIM}, drop-in replacement. No custom kernels. No runtime changes.**

Dystrio Sculpt structurally compresses transformer models, producing dense models that load with standard `transformers` — no custom code, no new ops, no deployment friction.

This is the **{TIER_DISPLAY}** tier of [{BASE_MODEL_DISPLAY}](https://huggingface.co/{BASE_MODEL_ID}).

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{HF_MODEL_ID}", torch_dtype="bfloat16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{HF_MODEL_ID}")

inputs = tokenizer("The future of AI inference is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Benchmark Results

All tiers compiled from [{BASE_MODEL_DISPLAY}](https://huggingface.co/{BASE_MODEL_ID}) on A100 80GB, bf16:

{BENCHMARK_TABLE}

### Key Metrics (this model)

| Metric | Value |
|--------|-------|
| **Weights memory** | {WEIGHTS_GB} GB ({SIZE_CUT}% smaller) |
| **PPL ratio** | {PPL_RATIO} |
| **Chat prefill TPS** | {CHAT_PREFILL_TPS} ({PREFILL_CHANGE}) |
| **RAG TTFT p95** | {RAG_TTFT_P95} ({TTFT_CHANGE}) |
| **Decode TPS** | {DECODE_TPS} (flat) |
| **Parameters** | {NUM_PARAMS} |

## All Sculpt Tiers

| Tier | HuggingFace | Size | PPL Ratio | Use Case |
|------|-------------|------|-----------|----------|
{TIER_LINKS_TABLE}

## What is Dystrio Sculpt?

Dystrio Sculpt compiles transformer models into smaller, faster variants. Output models:

- Are **dense** (not sparse) — standard architecture, fewer parameters
- Load with **standard HuggingFace Transformers** — no custom code needed
- Require **no custom kernels** and **no runtime changes**
- Work as a one-step compile before deployment
- Stack with quantization (AWQ, GPTQ, GGUF) for compound savings

## Compatibility

- ✅ HuggingFace Transformers
- ✅ vLLM
- ✅ TGI (Text Generation Inference)
- ✅ llama.cpp / GGUF conversion
- ✅ AWQ / GPTQ quantization
- ✅ Any framework that loads standard safetensors

## Benchmark Environment

- **GPU**: NVIDIA A100-SXM4-80GB
- **dtype**: bf16
- **Torch**: 2.10.0+cu128
- **Transformers**: 5.3.0
- **Deterministic**: True
- Single-GPU, standard HuggingFace Transformers, no custom kernels.

## Metric Definitions

- **PPL ratio**: WikiText-103 perplexity relative to baseline. <1.0 = quality improved.
- **Prefill TPS**: Tokens per second during prompt encoding (higher = faster).
- **TTFT p95**: Time to first token at 95th percentile (lower = faster).
- **Decode TPS**: Tokens per second during generation (higher = faster).
- **Weights (GB)**: Model parameter memory (deterministic, runtime-independent).

## Citation

```bibtex
@misc{dystrio_sculpt_2026,
  title={Dystrio Sculpt: Structural Compilation for Transformer LLMs},
  author={Dystrio},
  year={2026},
  url={https://huggingface.co/dystrio}
}
```
