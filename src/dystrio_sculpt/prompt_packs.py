"""Prompt pack schema and JSONL loader for bench workloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def load_prompt_pack(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL prompt pack.  Each line: {id, prompt, max_new_tokens, ...}."""
    prompts: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            p.setdefault("max_new_tokens", 64)
            p.setdefault("temperature", 0.0)
            p.setdefault("top_p", 1.0)
            prompts.append(p)
    return prompts


def prompt_pack_hash(path: Path) -> str:
    """SHA-256 prefix (16 hex chars) for provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
