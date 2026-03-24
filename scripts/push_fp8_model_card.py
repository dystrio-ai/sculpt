#!/usr/bin/env python3
"""Push model card to HuggingFace for Qwen3.5-122B-A10B-FP8-CacheReady.

Called at the end of provision_fp8_cacheready.sh after the benchmark completes.
Reads benchmark_report.md from the output directory and builds the card.

Usage:
    python scripts/push_fp8_model_card.py /path/to/output_dir
"""

import json
import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "dystrio/Qwen3.5-122B-A10B-FP8-CacheReady"
TOKEN = os.environ.get("HF_TOKEN", "")

output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None

# Try to extract benchmark numbers from results
prefix_results = {}
determinism = {}
quant_determinism = {}

if output_dir and output_dir.exists():
    report = output_dir / "benchmark_report.md"
    if report.exists():
        text = report.read_text()

        # Extract prefix caching throughput lines
        for line in text.split("\n"):
            m = re.search(
                r"\*\*(original|patched)\s*/\s*(shared|unique)_prefix\*\*.*?(\d+)\s*→\s*(\d+)\s*tok/s.*?(\d+\.\d+)x",
                line,
            )
            if not m:
                m = re.search(
                    r"(original|patched)\s*/\s*(shared|unique)_prefix.*?(\d+)\s*→\s*(\d+)\s*tok/s.*?(\d+\.\d+)x",
                    line,
                )
            if m:
                model, wtype, t_no, t_yes, speedup = m.groups()
                prefix_results[(model, wtype)] = {
                    "no_cache": int(t_no),
                    "with_cache": int(t_yes),
                    "speedup": float(speedup),
                }

        # Extract determinism results
        for label in ("original", "patched"):
            m = re.search(
                rf"{label}.*?(\d+)/(\d+).*?deterministic.*?(\d+\.?\d*)%",
                text,
            )
            if m:
                determinism[label] = f"{m.group(1)}/{m.group(2)} ({m.group(3)}%)"

# Build throughput table from results or use placeholders
orig_shared = prefix_results.get(("original", "shared"), {})
orig_unique = prefix_results.get(("original", "unique"), {})
patch_shared = prefix_results.get(("patched", "shared"), {})
patch_unique = prefix_results.get(("patched", "unique"), {})

def _tput_row(label, bold, data):
    if not data:
        return f"| {label} | — | — | — |"
    nc = data["no_cache"]
    wc = data["with_cache"]
    sp = data["speedup"]
    if bold:
        return f"| **{label}** | **{nc} tok/s** | **{wc} tok/s** | **{sp:.2f}x** |"
    return f"| {label} | {nc} tok/s | {wc} tok/s | {sp:.2f}x |"

tput_table = "\n".join([
    "| Model | Without Cache | With Cache | Speedup |",
    "|-------|--------------|------------|---------|",
    _tput_row("Original (FP8) — Shared prefix", False, orig_shared),
    _tput_row("Original (FP8) — Unique prefix", False, orig_unique),
    _tput_row("FP8-CacheReady — Shared prefix", True, patch_shared),
    _tput_row("FP8-CacheReady — Unique prefix", True, patch_unique),
])

det_orig = determinism.get("original", "—")
det_patch = determinism.get("patched", "—")

CARD = f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language:
  - en
base_model: Qwen/Qwen3.5-122B-A10B-FP8
tags:
  - dystrio
  - moe
  - fp8
  - prefix-caching
  - routing-determinism
  - vllm
  - drop-in-replacement
  - qwen3.5
  - mixture-of-experts
  - quantized
---

# Qwen3.5-122B-A10B-FP8-CacheReady

**FP8-quantized Qwen3.5-122B with routing canonicalization for reliable prefix caching.**

This is the FP8 variant of [dystrio/Qwen3.5-122B-A10B-CacheReady](https://huggingface.co/dystrio/Qwen3.5-122B-A10B-CacheReady), built directly from [Qwen/Qwen3.5-122B-A10B-FP8](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) (Qwen's official FP8 checkpoint).

CacheReady converts MoE prefix caching from unreliable to usable by enforcing routing stability across shared-prefix execution contexts.

## Why FP8 + CacheReady

FP8 quantization gives you 2x memory reduction and faster inference — but makes MoE routing *more* unstable. Small numerical differences from FP8 quantization flip top-k expert selection for functionally equivalent experts, invalidating the KV cache for identical prefixes. CacheReady eliminates this instability.

Only router (gate) weight matrices were modified. In Qwen's FP8 checkpoint, gate weights are already stored in bf16 precision (excluded from quantization by Qwen). All expert weights, attention weights, embeddings, and quantization scales are byte-for-byte identical to `Qwen/Qwen3.5-122B-A10B-FP8`.

Approximately ~45% of experts fell into equivalence groups across the model.

## Prefix Caching Throughput

{tput_table}

## Single-run routing determinism (sanity check)

Both models are deterministic within a single execution context. Prefix caching failures arise from routing instability across requests that share prefixes but differ slightly in batching or execution context.

| Model | Determinism |
|-------|------------|
| Original FP8 | {det_orig} |
| **FP8-CacheReady** | **{det_patch}** |

This verifies router stability within a single execution context. CacheReady targets routing stability across shared-prefix requests, which is the requirement for prefix caching to function correctly.

## Why routing can be deterministic but prefix caching still fails

Router determinism within a single execution context does not guarantee routing stability across requests.

Small numerical differences caused by:

- FP8 quantization rounding
- Batch shape changes
- Execution order differences
- Multi-tenant serving reuse

can flip top-k expert selection when experts are functionally equivalent.

CacheReady removes this instability by canonicalizing routing scores across equivalent experts.

## Quality Preservation

No measurable quality change. The patch only modifies router weight rows for experts that were identified as functionally equivalent. The same routing canonicalization patch from the [bf16 CacheReady variant](https://huggingface.co/dystrio/Qwen3.5-122B-A10B-CacheReady) applies directly because Qwen stores gate weights in bf16 precision even in FP8 checkpoints.

## Who Benefits

- **FP8 serving with prefix caching** — the combination that breaks most on vanilla MoE
- **Shared-prefix workloads** — system prompts, RAG context, multimodal prefixes
- **Multi-tenant deployments** — many users sharing the same base prompt
- **Batched inference** — consistent routing across batch elements with shared prefixes
- **Memory-constrained setups** — FP8 uses ~120GB vs ~240GB for bf16

## Usage

Drop-in replacement for `Qwen/Qwen3.5-122B-A10B-FP8`. No code changes needed.

### With vLLM (prefix caching now works)

```bash
python -m vllm.entrypoints.openai.api_server \\
    --model dystrio/Qwen3.5-122B-A10B-FP8-CacheReady \\
    --tensor-parallel-size 4 \\
    --enable-prefix-caching \\
    --trust-remote-code
```

### With transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "dystrio/Qwen3.5-122B-A10B-FP8-CacheReady",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("dystrio/Qwen3.5-122B-A10B-FP8-CacheReady")
```

## Compatibility

- vLLM >= 0.17 (native FP8 support)
- SGLang
- Any framework that loads FP8 safetensors checkpoints

## Related Models

| Variant | Precision | Size | Link |
|---------|-----------|------|------|
| CacheReady (bf16) | bf16 | ~240 GB | [dystrio/Qwen3.5-122B-A10B-CacheReady](https://huggingface.co/dystrio/Qwen3.5-122B-A10B-CacheReady) |
| **CacheReady (FP8)** | **FP8** | **~120 GB** | **this model** |

## Citation

```bibtex
@misc{{dystrio_cacheready_fp8_2026,
  title={{Routing Canonicalization for Deterministic MoE Inference (FP8)}},
  author={{Dystrio}},
  year={{2026}},
  url={{https://huggingface.co/dystrio/Qwen3.5-122B-A10B-FP8-CacheReady}}
}}
```
"""

api = HfApi(token=TOKEN)
api.upload_file(
    path_or_fileobj=CARD.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    commit_message="Add model card with benchmark results",
)
print(f"DONE - model card pushed to https://huggingface.co/{REPO_ID}")
