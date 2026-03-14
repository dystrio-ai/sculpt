"""Sync the local JSONL dataset to a private HuggingFace dataset repo.

Called after each successful factory run to keep the remote copy current.
Requires HF_TOKEN in environment with write access to the target org.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

DEFAULT_REPO_ID = "dystrio/efficiency-dataset"
DEFAULT_FILENAME = "dystrio_efficiency_dataset.jsonl"


def sync_dataset_to_hub(
    local_path: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    token: Optional[str] = None,
) -> bool:
    """Upload the local JSONL dataset file to a private HuggingFace dataset repo.

    Creates the repo (private) if it doesn't exist. Returns True on success.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        _log.warning("dataset file not found: %s — skipping sync", local_path)
        return False

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        _log.warning("HF_TOKEN not set — skipping dataset sync")
        return False

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)

        api.create_repo(
            repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )

        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=local_path.name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"auto-sync after factory run ({local_path.name})",
        )

        _log.info("synced dataset to hf://datasets/%s/%s", repo_id, local_path.name)
        return True

    except ImportError:
        _log.warning("huggingface_hub not installed — skipping dataset sync")
        return False
    except Exception as exc:
        _log.error("dataset sync failed: %s", exc)
        return False
