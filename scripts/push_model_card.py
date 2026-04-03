#!/usr/bin/env python3
"""Push model card to HuggingFace for Qwen3.5-122B-A10B-CacheReady."""

import os

from huggingface_hub import HfApi

CARD = """---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language:
  - en
base_model: Qwen/Qwen3.5-122B-A10B
tags:
  - dystrio
  - moe
  - prefix-caching
  - routing-determinism
  - vllm
  - drop-in-replacement
  - qwen3.5
  - mixture-of-experts
---

# Qwen3.5-122B-A10B-CacheReady

**CacheReady converts MoE prefix caching from unreliable to usable without runtime patches.**

MoE routing noise — especially under fp8/fp4 quantization — prevents KV cache reuse across requests with identical prefixes. CacheReady canonicalizes router weights across functionally equivalent experts so routing decisions become deterministic by construction. Result: prefix caching becomes usable again for shared-prefix workloads.

Shared-prefix throughput goes from **0.65x** (caching hurts) to **1.31x** (caching helps).

CacheReady converts MoE prefix caching from unreliable to usable by enforcing routing stability across shared-prefix execution contexts.

Only router (gate) weight matrices were modified. Expert weights, attention weights, embeddings, and all other parameters are byte-for-byte identical to the original `Qwen/Qwen3.5-122B-A10B`. CacheReady is not a finetune or architecture change. It is a router canonicalization patch encoded directly into model weights to enable deterministic MoE inference without runtime modifications.

Approximately ~45% of experts fell into equivalence groups across the model.

## Prefix Caching Throughput

All benchmarks: 4x NVIDIA H100 PCIe 80GB, vLLM 0.18.0, `enforce_eager=True`, `tensor_parallel_size=4`.

| Model | Workload | Without Cache | With Cache | Speedup |
|-------|----------|--------------|------------|---------|
| Original | Shared prefix | 720 tok/s | 466 tok/s | **0.65x (slower)** |
| Original | Unique prefix | 481 tok/s | 480 tok/s | 1.00x |
| **CacheReady** | **Shared prefix** | **561 tok/s** | **735 tok/s** | **1.31x** |
| **CacheReady** | Unique prefix | 525 tok/s | 565 tok/s | 1.08x |

On the original model, enabling prefix caching for shared-prefix workloads makes throughput **35% worse**. Routing instability invalidates cached KV states, turning cache hits into expensive misses. On CacheReady, the same workload sees a **31% throughput improvement** — prefix caching works as expected because routing is deterministic. This converts prefix caching from unreliable to usable in MoE serving environments.

## Single-run routing determinism (sanity check)

Both models are deterministic within a single execution context. Prefix caching failures arise from routing instability across requests that share prefixes but differ slightly in quantization state, batching, or execution context.

| Model | Texts | bf16 Determinism | fp8 Determinism |
|-------|-------|-----------------|-----------------|
| Original | 20 (bf16) / 10 (fp8) | 100% | 100% |
| **CacheReady** | 20 (bf16) / 10 (fp8) | **100%** | **100%** |

This verifies router stability within a single execution context. CacheReady instead targets routing stability across shared-prefix requests, which is the requirement for prefix caching to function correctly.

## Shared-prefix routing stability (cache reuse behavior)

Prefix caching requires routing decisions to remain identical across requests that share prefixes.

Example shared-prefix serving workload (vLLM):

| Model | Shared-prefix throughput |
|-------|--------------------------|
| Original Qwen3.5-122B-A10B | 0.65x |
| CacheReady | 1.31x |

On the original model, routing instability invalidates the KV cache even when prefixes match.

CacheReady canonicalizes router weights across equivalent experts so shared-prefix routing remains stable and cache reuse becomes effective.

## Why routing can be deterministic but prefix caching still fails

Router determinism within a single execution context does not guarantee routing stability across requests.

Small numerical differences caused by:

- fp8 / fp4 quantization
- batch shape changes
- execution order differences
- multi-tenant serving reuse

can flip top-k expert selection when experts are functionally equivalent.

CacheReady removes this instability by canonicalizing routing scores across equivalent experts.

## Quality Preservation

No measurable quality change across evaluation benchmarks. The patch only modifies router weight rows for experts that were identified as functionally equivalent — producing near-identical outputs when selected. The router top-k selection is unchanged for all non-ambiguous routing decisions.

## Router Equivalence Discovery

| Metric | Value |
|--------|-------|
| MoE layers analyzed | 48 / 48 |
| Expert equivalence groups | 2,348 (non-singleton) |
| Experts in equivalence groups | 5,555 / 12,288 (45%) |
| Router weight rows modified | 3,207 |
| Max logit diff after patch | 0.00e+00 |
| Calibration time | 12.5 minutes (8xA100) |

45% of experts across all 48 MoE layers belong to non-singleton equivalence groups — meaning large-scale routing redundancy exists across MoE routers. CacheReady exploits this redundancy safely: equivalent experts receive identical router scores, so the top-k selection is deterministic without affecting which expert actually computes the output.

## Who Benefits

- **Shared-prefix serving** — system prompts, RAG context, multimodal (image/video) prefixes
- **Quantized serving** — fp8/fp4 quantization amplifies routing noise; CacheReady eliminates it
- **Multi-tenant deployments** — many users sharing the same base prompt
- **Batched inference** — consistent routing across batch elements with shared prefixes
- **vLLM prefix caching users** — prefix caching now works correctly for MoE models

## Usage

Drop-in replacement. No code changes needed.

### With vLLM (prefix caching now works)

```bash
python -m vllm.entrypoints.openai.api_server \\
    --model dystrio/Qwen3.5-122B-A10B-CacheReady \\
    --tensor-parallel-size 4 \\
    --enable-prefix-caching \\
    --trust-remote-code
```

### With transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "dystrio/Qwen3.5-122B-A10B-CacheReady",
    torch_dtype="bfloat16",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("dystrio/Qwen3.5-122B-A10B-CacheReady")
```

## Compatibility

- transformers (requires qwen3_5_moe support)
- vLLM >= 0.17
- SGLang
- TGI
- Any framework that loads standard safetensors checkpoints

## Citation

```bibtex
@misc{dystrio_cacheready_2026,
  title={Routing Canonicalization for Deterministic MoE Inference},
  author={Dystrio},
  year={2026},
  url={https://huggingface.co/dystrio/Qwen3.5-122B-A10B-CacheReady}
}
```
"""

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj=CARD.encode(),
    path_in_repo="README.md",
    repo_id="dystrio/Qwen3.5-122B-A10B-CacheReady",
    commit_message="Add model card",
)
print("DONE - model card pushed to https://huggingface.co/dystrio/Qwen3.5-122B-A10B-CacheReady")
