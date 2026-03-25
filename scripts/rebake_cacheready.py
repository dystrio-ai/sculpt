#!/usr/bin/env python3
"""Re-bake CacheReady model with tiebreaker fix.

Downloads the original model + routing patch, applies the fixed
bake_routing_patch (with tiebreak_eps scaling), saves locally.

Usage:
    python scripts/rebake_cacheready.py \
        --base Qwen/Qwen3.5-122B-A10B \
        --patch-source dystrio/Qwen3.5-122B-A10B-CacheReady \
        --output /ephemeral/cacheready_v2 \
        --tp 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [rebake] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rebake")


def main():
    parser = argparse.ArgumentParser(description="Re-bake CacheReady with tiebreaker fix")
    parser.add_argument("--base", required=True, help="Base model HF ID (e.g. Qwen/Qwen3.5-122B-A10B)")
    parser.add_argument("--patch-source", required=True, help="Existing CacheReady repo with routing_patch.json")
    parser.add_argument("--output", required=True, help="Output directory for fixed model")
    parser.add_argument("--tiebreak-eps", type=float, default=1e-4, help="Tiebreaker epsilon (default 1e-4)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download, hf_hub_download

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / "models"))

    # Download base model
    base_local = hf_home / args.base.replace("/", "--")
    if base_local.exists() and (base_local / "config.json").exists():
        log.info("base model cached: %s", base_local)
    else:
        log.info("downloading base model %s...", args.base)
        snapshot_download(repo_id=args.base, local_dir=str(base_local))

    # Download routing patch from the existing CacheReady repo
    log.info("downloading routing_patch.json from %s...", args.patch_source)
    patch_path = hf_hub_download(args.patch_source, "routing_patch.json")

    from dystrio_sculpt.moe_routing_patch import RoutingPatch
    patch = RoutingPatch.load(patch_path)
    n_non_singleton = sum(1 for ecs in patch.layers.values() for ec in ecs if len(ec.members) > 1)
    log.info("loaded patch: %d layers, %d non-singleton classes", len(patch.layers), n_non_singleton)

    # Copy base model to output
    log.info("copying base model to %s...", output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(str(base_local), str(output_dir))

    # Copy routing patch
    shutil.copy2(patch_path, str(output_dir / "routing_patch.json"))

    # Apply tiebreaker bake directly to safetensors
    from safetensors import safe_open
    from safetensors.torch import save_file

    index_path = output_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    gate_keys = [k for k in weight_map if ".gate.weight" in k and not k.startswith("mtp.")]
    log.info("found %d gate weight keys to patch", len(gate_keys))

    shard_to_gates = defaultdict(list)
    for k in gate_keys:
        shard_to_gates[weight_map[k]].append(k)

    scale = 1.0 - args.tiebreak_eps
    total_rows = 0

    for shard_name, keys in sorted(shard_to_gates.items()):
        shard_path = output_dir / shard_name
        log.info("patching %s (%d gate weights)...", shard_name, len(keys))

        tensors = {}
        with safe_open(str(shard_path), framework="pt") as f:
            for tensor_name in f.keys():
                tensors[tensor_name] = f.get_tensor(tensor_name)

        for gate_key in keys:
            parts = gate_key.split(".")
            layer_idx = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        pass

            if layer_idx is None or layer_idx not in patch.layers:
                continue

            W = tensors[gate_key]
            n_experts = W.shape[0]

            for ec in patch.layers[layer_idx]:
                if len(ec.members) <= 1 or ec.canonical >= n_experts:
                    continue
                canonical_row = W[ec.canonical].clone()
                for member in ec.members:
                    if member != ec.canonical and member < n_experts:
                        W[member].copy_(canonical_row * scale)
                        total_rows += 1

            tensors[gate_key] = W

        save_file(tensors, str(shard_path))

    log.info("bake complete: %d gate weight rows modified (scale=%.6f)", total_rows, scale)
    log.info("fixed model saved to: %s", output_dir)


if __name__ == "__main__":
    main()
