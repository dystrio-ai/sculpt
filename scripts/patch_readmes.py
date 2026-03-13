#!/usr/bin/env python3
"""Patch README.md across all HuggingFace repos to remove FFN references."""

from huggingface_hub import HfApi

api = HfApi()
ORG = "dystrio"

REPOS = [
    "Mistral-7B-Instruct-v0.3-sculpt-default",
    "Mistral-7B-Instruct-v0.3-sculpt-production",
    "Mistral-7B-Instruct-v0.3-sculpt-throughput",
    "Mistral-7B-Instruct-v0.3-sculpt-experimental",
    "Llama-3.1-8B-Instruct-sculpt-default",
    "Llama-3.1-8B-Instruct-sculpt-production",
    "Llama-3.1-8B-Instruct-sculpt-throughput",
    "Llama-3.1-8B-Instruct-sculpt-experimental",
    "Qwen2.5-7B-Instruct-sculpt-default",
    "Qwen2.5-7B-Instruct-sculpt-production",
    "Qwen2.5-7B-Instruct-sculpt-throughput",
]

REPLACEMENTS = [
    ("transformer FFN blocks", "transformer models"),
    ("Structural FFN Compilation for Transformer LLMs", "Structural Compilation for Transformer LLMs"),
]

for repo_name in REPOS:
    repo_id = f"{ORG}/{repo_name}"
    try:
        readme = api.hf_hub_download(repo_id=repo_id, filename="README.md")
        with open(readme) as f:
            content = f.read()

        updated = content
        for old, new in REPLACEMENTS:
            updated = updated.replace(old, new)

        if updated == content:
            print(f"  SKIP {repo_id}: no changes needed")
            continue

        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write(updated)
            tmp_path = tmp.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Remove FFN specificity from model card",
        )
        os.unlink(tmp_path)
        print(f"  OK {repo_id}")
    except Exception as e:
        print(f"  FAIL {repo_id}: {e}")
