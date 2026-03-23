#!/usr/bin/env python3
"""Extract missing vision encoder + MTP weights from original model
and add them to the patched HuggingFace repo.

The routing patch only modified gate weights in MoE layers. Vision encoder
and MTP (Multi-Token Prediction) weights are unchanged but were lost when
the original save used AutoModelForCausalLM (text-only) instead of the
full multimodal model class.

This script surgically copies only those missing tensors without touching
the patched routing weights.
"""

import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    print("ERROR: set HF_TOKEN environment variable")
    sys.exit(1)

ORIGINAL = "Qwen/Qwen3.5-122B-A10B"
PATCHED = "dystrio/Qwen3.5-122B-A10B-CacheReady"

api = HfApi(token=TOKEN)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1. Load both indexes
        print("Loading safetensors indexes...")
        orig_idx = json.load(
            open(hf_hub_download(ORIGINAL, "model.safetensors.index.json", token=TOKEN, cache_dir=str(tmp)))
        )
        patched_idx = json.load(
            open(hf_hub_download(PATCHED, "model.safetensors.index.json", token=TOKEN, cache_dir=str(tmp)))
        )

        orig_wm = orig_idx["weight_map"]
        patched_wm = patched_idx["weight_map"]

        # 2. Find missing vision + MTP keys (skip fused expert keys)
        missing = set(orig_wm.keys()) - set(patched_wm.keys())
        needed = {k for k in missing if k.startswith("mtp.") or "model.visual" in k}

        fused_skipped = missing - needed
        print(f"Missing keys: {len(missing)} total")
        print(f"  Vision + MTP to extract: {len(needed)}")
        print(f"  Fused experts skipped:   {len(fused_skipped)}")

        if not needed:
            print("Nothing to extract!")
            return

        # Group by source shard
        shard_to_keys = defaultdict(list)
        for k in needed:
            shard_to_keys[orig_wm[k]].append(k)

        print(f"\nExtracting from {len(shard_to_keys)} original shards...")

        # 3. Download each shard, extract needed tensors, delete shard
        all_tensors = {}
        for shard_name, keys in sorted(shard_to_keys.items()):
            print(f"\n  {shard_name}: {len(keys)} tensors")
            shard_path = hf_hub_download(
                ORIGINAL, shard_name, token=TOKEN, cache_dir=str(tmp)
            )
            print(f"    downloaded → extracting...")

            with safe_open(shard_path, framework="pt") as f:
                for k in keys:
                    all_tensors[k] = f.get_tensor(k)

            print(f"    extracted {len(keys)} tensors, freeing shard...")
            try:
                os.remove(shard_path)
            except OSError:
                pass

        print(f"\nTotal extracted: {len(all_tensors)} tensors")

        # 4. Save extracted tensors as new shard files
        #    Split into chunks of ~5 GB to stay within HF upload limits
        MAX_SHARD_BYTES = 5 * 1024 ** 3
        shard_groups = []
        current_group = {}
        current_size = 0

        for k in sorted(all_tensors.keys()):
            t = all_tensors[k]
            t_bytes = t.nelement() * t.element_size()
            if current_size + t_bytes > MAX_SHARD_BYTES and current_group:
                shard_groups.append(current_group)
                current_group = {}
                current_size = 0
            current_group[k] = t
            current_size += t_bytes

        if current_group:
            shard_groups.append(current_group)

        # Determine shard numbering (continue from patched model's max shard)
        existing_shards = set(patched_wm.values())
        max_existing = 0
        for s in existing_shards:
            try:
                num = int(s.split("-")[1])
                max_existing = max(max_existing, num)
            except (IndexError, ValueError):
                pass

        total_shards = max_existing + len(shard_groups)
        new_shard_files = []
        updated_wm = dict(patched_wm)

        for i, group in enumerate(shard_groups):
            shard_num = max_existing + 1 + i
            shard_name = f"model.safetensors-{shard_num:05d}-of-{total_shards:05d}.safetensors"
            shard_path = tmp / shard_name

            save_file(group, str(shard_path))
            new_shard_files.append((shard_name, shard_path))

            for k in group:
                updated_wm[k] = shard_name

            total_bytes = sum(t.nelement() * t.element_size() for t in group.values())
            print(f"  Saved {shard_name}: {len(group)} tensors, {total_bytes / 1e9:.2f} GB")

        # Also update total_size in index and rename existing shards in the weight_map
        # to reflect the new total count
        # Actually, the shard naming convention "XXXXX-of-YYYYY" needs consistent YYYYY.
        # But HF doesn't strictly enforce this. Let's keep it simple.

        # 5. Write updated index
        new_idx = {
            "metadata": patched_idx.get("metadata", {}),
            "weight_map": updated_wm,
        }
        idx_path = tmp / "model.safetensors.index.json"
        with open(idx_path, "w") as f:
            json.dump(new_idx, f, indent=2)

        print(f"\nUpdated index: {len(updated_wm)} total weight keys")
        print(f"  (was {len(patched_wm)}, added {len(updated_wm) - len(patched_wm)})")

        # 6. Upload to HuggingFace
        print(f"\nUploading to {PATCHED}...")

        # Upload new shard files
        for shard_name, shard_path in new_shard_files:
            print(f"  uploading {shard_name} ({shard_path.stat().st_size / 1e9:.2f} GB)...")
            api.upload_file(
                path_or_fileobj=str(shard_path),
                path_in_repo=shard_name,
                repo_id=PATCHED,
                commit_message=f"Add missing vision/MTP weights ({shard_name})",
            )

        # Upload updated index
        print("  uploading model.safetensors.index.json...")
        api.upload_file(
            path_or_fileobj=str(idx_path),
            path_in_repo="model.safetensors.index.json",
            repo_id=PATCHED,
            commit_message="Update weight index with vision encoder + MTP layers",
        )

        print(f"\nDone! {PATCHED} now has all {len(updated_wm)} weight keys.")
        print("  Vision encoder: restored")
        print("  MTP layers: restored")
        print("  Routing patch: preserved (unfused per-expert format)")


if __name__ == "__main__":
    main()
