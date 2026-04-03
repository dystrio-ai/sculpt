#!/usr/bin/env python3
"""Fix the patched 122B MoE model on HuggingFace.

Downloads missing files from the original Qwen/Qwen3.5-122B-A10B model
and uploads them to dystrio/Qwen3.5-122B-A10B-CacheReady. Also fixes
known config issues (tokenizer_class, model_type).

Usage:
    export HF_TOKEN="hf_..."
    python3 scripts/fix_patched_hf_repo.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

TOKEN = os.environ.get("HF_TOKEN")
ORIGINAL = "Qwen/Qwen3.5-122B-A10B"
PATCHED = "dystrio/Qwen3.5-122B-A10B-CacheReady"

SKIP_EXTENSIONS = {".safetensors", ".bin", ".pt", ".gguf", ".onnx"}


def main():
    if not TOKEN:
        print("ERROR: set HF_TOKEN")
        sys.exit(1)

    api = HfApi(token=TOKEN)

    print(f"Listing files in original: {ORIGINAL}")
    orig_files = set(list_repo_files(ORIGINAL, token=TOKEN))

    print(f"Listing files in patched: {PATCHED}")
    patched_files = set(list_repo_files(PATCHED, token=TOKEN))

    missing = []
    for f in orig_files:
        if any(f.endswith(ext) for ext in SKIP_EXTENSIONS):
            continue
        if f not in patched_files:
            missing.append(f)

    print(f"\nMissing files in patched repo ({len(missing)}):")
    for f in sorted(missing):
        print(f"  {f}")

    if missing:
        print(f"\nCopying {len(missing)} files from original → patched...")
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in missing:
                try:
                    local = hf_hub_download(ORIGINAL, f, token=TOKEN, cache_dir=tmpdir)
                    api.upload_file(
                        path_or_fileobj=local,
                        path_in_repo=f,
                        repo_id=PATCHED,
                        commit_message=f"Copy {f} from original model",
                    )
                    print(f"  uploaded: {f}")
                except Exception as e:
                    print(f"  FAILED {f}: {e}")

    # Fix tokenizer_config.json
    print("\nFixing tokenizer_config.json...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tok_path = hf_hub_download(PATCHED, "tokenizer_config.json", token=TOKEN, cache_dir=tmpdir)
        with open(tok_path) as f:
            tok_cfg = json.load(f)

        if tok_cfg.get("tokenizer_class") == "TokenizersBackend":
            tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            fixed_path = Path(tmpdir) / "tokenizer_config.json"
            with open(fixed_path, "w") as f:
                json.dump(tok_cfg, f, indent=2)
            api.upload_file(
                path_or_fileobj=str(fixed_path),
                path_in_repo="tokenizer_config.json",
                repo_id=PATCHED,
                commit_message="Fix tokenizer_class: TokenizersBackend → PreTrainedTokenizerFast",
            )
            print("  fixed tokenizer_class")
        else:
            print(f"  tokenizer_class already: {tok_cfg.get('tokenizer_class')}")

    print("\nDone. Patched repo should now load cleanly in vLLM.")


if __name__ == "__main__":
    main()
