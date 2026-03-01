"""Model and tokenizer loading."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def resolve_dtype(s: str) -> torch.dtype:
    d = DTYPE_MAP.get(s.lower())
    if d is None:
        raise ValueError(f"Unknown dtype: {s!r}. Use one of {list(DTYPE_MAP)}")
    return d


def load_model_and_tokenizer(
    model_id: str, device: str, dtype: torch.dtype,
):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    return model, tok
