#!/usr/bin/env python3
"""Generate benchmark prompt packs (JSONL) from HuggingFace datasets.

Usage:
    python scripts/make_prompt_packs.py --outdir prompts/

Produces:
    chat_200.jsonl   – short instruction/QA prompts
    rag_100.jsonl    – long-context retrieval-style prompts
    code_100.jsonl   – code completion / analysis prompts

Requires: pip install datasets
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} prompts -> {path}")


def _load_wikitext():
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    return [t.strip() for t in ds["text"] if len(t.strip()) > 100]


def make_chat(texts, n=200, seed=42):
    rng = random.Random(seed)
    sampled = rng.sample(texts, min(n, len(texts)))
    prompts = []
    templates = [
        "Summarize the following passage in two sentences:\n\n{text}",
        "What are the key facts mentioned in this passage?\n\n{text}",
        "Explain the following in simple terms:\n\n{text}",
        "Rewrite the following passage as a bulleted list:\n\n{text}",
    ]
    for i, t in enumerate(sampled):
        tpl = templates[i % len(templates)]
        # Truncate text to keep prompts short for chat
        short = " ".join(t.split()[:80])
        prompts.append({
            "id": f"chat_{i:03d}",
            "prompt": tpl.format(text=short),
            "max_new_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
        })
    return prompts


def make_rag(texts, n=100, seed=42):
    rng = random.Random(seed)
    prompts = []
    for i in range(n):
        # Concatenate 5-8 paragraphs for long context
        n_para = rng.randint(5, 8)
        paras = rng.sample(texts, min(n_para, len(texts)))
        context = "\n\n".join(paras)
        prompts.append({
            "id": f"rag_{i:03d}",
            "prompt": (
                "Based on the following context, provide a comprehensive summary "
                "of the main topics discussed.\n\n"
                f"Context:\n{context}\n\n"
                "Summary:"
            ),
            "max_new_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
        })
    return prompts


def make_code(texts, n=100, seed=42):
    rng = random.Random(seed)
    sampled = rng.sample(texts, min(n, len(texts)))
    templates = [
        "Write a Python function that extracts all proper nouns from the following text:\n\n{text}\n\ndef extract_proper_nouns(text: str) -> list[str]:",
        "Write a Python function that counts the word frequency in the following text:\n\n{text}\n\ndef word_frequency(text: str) -> dict[str, int]:",
        "Write a Python function that classifies the topic of the following text:\n\n{text}\n\ndef classify_topic(text: str) -> str:",
        "Write a Python function that generates a 3-sentence summary of the following text:\n\n{text}\n\ndef summarize(text: str) -> str:",
    ]
    prompts = []
    for i, t in enumerate(sampled):
        tpl = templates[i % len(templates)]
        short = " ".join(t.split()[:60])
        prompts.append({
            "id": f"code_{i:03d}",
            "prompt": tpl.format(text=short),
            "max_new_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
        })
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark prompt packs")
    parser.add_argument("--outdir", type=str, default="prompts", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    print("loading wikitext-103-v1 ...")
    texts = _load_wikitext()
    print(f"  {len(texts)} usable paragraphs")

    print("generating chat_200 ...")
    _write_jsonl(outdir / "chat_200.jsonl", make_chat(texts, 200, args.seed))

    print("generating rag_100 ...")
    _write_jsonl(outdir / "rag_100.jsonl", make_rag(texts, 100, args.seed))

    print("generating code_100 ...")
    _write_jsonl(outdir / "code_100.jsonl", make_code(texts, 100, args.seed))

    print("done.")


if __name__ == "__main__":
    main()
