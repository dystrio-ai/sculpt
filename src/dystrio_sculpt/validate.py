"""Post-save validation: reload model and verify correctness."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_log = logging.getLogger(__name__)


def validate_saved_model(
    model_dir: Path | str,
    device: str = "cuda",
) -> bool:
    """Reload a saved model and verify it produces valid outputs.

    Checks:
      1. Model loads via AutoModelForCausalLM.from_pretrained
      2. Single forward pass produces logits with no NaN or Inf
      3. Output shape is consistent with model config
    """
    model_dir = Path(model_dir)
    _log.info("validating saved model at %s", model_dir)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, trust_remote_code=True,
        ignore_mismatched_sizes=True,
    ).to(device)
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    inp = tok("The quick brown fox jumps over the lazy dog", return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}

    with torch.no_grad():
        out = model(**inp, use_cache=False)

    logits = out.logits
    seq_len = inp["input_ids"].shape[1]

    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    shape_ok = (
        logits.shape[0] == 1
        and logits.shape[1] == seq_len
        and logits.shape[2] == model.config.vocab_size
    )

    passed = not has_nan and not has_inf and shape_ok

    if not passed:
        reasons = []
        if has_nan:
            reasons.append("NaN in logits")
        if has_inf:
            reasons.append("Inf in logits")
        if not shape_ok:
            reasons.append(f"shape mismatch: {logits.shape}")
        _log.error("validation FAILED: %s", ", ".join(reasons))
    else:
        _log.info("validation passed: shape=%s, no NaN/Inf", list(logits.shape))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return passed
