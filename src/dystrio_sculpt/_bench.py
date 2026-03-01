"""Throughput benchmarks with CUDA synchronization."""

from __future__ import annotations

import time
from typing import Sequence

import torch


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _tokenize_and_pad(tokenizer, texts: Sequence[str], seq_len: int, device: str):
    ids_list = []
    for t in texts:
        inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=seq_len)
        ids_list.append(inp["input_ids"][0])
    if not ids_list:
        return None, None, 0
    maxl = max(int(x.shape[0]) for x in ids_list)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    ids = torch.full((len(ids_list), maxl), fill_value=pad_id, dtype=torch.long)
    attn = torch.zeros((len(ids_list), maxl), dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids[i, : x.shape[0]] = x
        attn[i, : x.shape[0]] = 1
    return ids.to(device), attn.to(device), int(attn.sum().item())


@torch.inference_mode()
def bench_prefill_tps(
    model,
    tokenizer,
    texts: Sequence[str],
    seq_len: int,
    device: str,
    warmup: int = 20,
    iters: int = 80,
) -> float:
    model.eval()
    ids, attn, tokens = _tokenize_and_pad(tokenizer, texts, seq_len, device)
    if ids is None:
        return 0.0
    for _ in range(warmup):
        model(input_ids=ids, attention_mask=attn, use_cache=False)
    _sync(device)
    t0 = time.time()
    for _ in range(iters):
        model(input_ids=ids, attention_mask=attn, use_cache=False)
    _sync(device)
    return (tokens * iters) / max(1e-9, time.time() - t0)


@torch.inference_mode()
def bench_decode_tps(
    model,
    tokenizer,
    text: str,
    prompt_len: int,
    device: str,
    decode_steps: int = 128,
    warmup: int = 3,
    iters: int = 10,
) -> float:
    model.eval()
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=prompt_len)
    prompt_ids = inp["input_ids"].to(device)

    def _prefill():
        out = model(input_ids=prompt_ids, use_cache=True)
        return out.past_key_values, out.logits[:, -1:, :].argmax(dim=-1)

    def _decode(past_kv, next_token):
        for _ in range(decode_steps):
            out = model(
                input_ids=next_token, past_key_values=past_kv, use_cache=True,
            )
            past_kv = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(dim=-1)

    for _ in range(warmup):
        past_kv, next_token = _prefill()
        _decode(past_kv, next_token)
    _sync(device)

    total_dt = 0.0
    for _ in range(iters):
        past_kv, next_token = _prefill()
        _sync(device)
        t0 = time.time()
        _decode(past_kv, next_token)
        _sync(device)
        total_dt += time.time() - t0

    return (decode_steps * iters) / max(1e-9, total_dt)
