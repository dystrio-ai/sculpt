"""Model watcher: discover new models on HuggingFace Hub for factory processing.

Polls HuggingFace model listings, filters by supported architecture and
popularity, and queues compatible models for the factory pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

_log = logging.getLogger(__name__)


@dataclass
class WatchCandidate:
    model_id: str
    family: str
    support_state: str
    downloads: int = 0
    likes: int = 0
    last_modified: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class WatchResult:
    candidates: List[WatchCandidate]
    scanned: int = 0
    filtered: int = 0
    errors: int = 0


def discover_models(
    *,
    architectures: Optional[List[str]] = None,
    min_downloads: int = 100,
    max_params_b: float = 15.0,
    limit: int = 50,
    already_processed: Optional[Set[str]] = None,
) -> WatchResult:
    """Discover candidate models from HuggingFace Hub.

    Filters by architecture support, download count, and parameter count.
    Returns WatchResult with ranked candidates.
    """
    from huggingface_hub import HfApi

    if architectures is None:
        architectures = ["llama", "mistral", "qwen", "phi", "gemma"]

    if already_processed is None:
        already_processed = set()

    api = HfApi()
    candidates: List[WatchCandidate] = []
    scanned = 0
    errors = 0

    for arch in architectures:
        _log.info("scanning architecture: %s", arch)
        try:
            models = api.list_models(
                filter=f"text-generation",
                search=arch,
                sort="downloads",
                direction=-1,
                limit=limit,
            )
            for model_info in models:
                scanned += 1
                model_id = model_info.id

                if model_id in already_processed:
                    continue

                downloads = getattr(model_info, "downloads", 0) or 0
                if downloads < min_downloads:
                    continue

                likes = getattr(model_info, "likes", 0) or 0
                tags = getattr(model_info, "tags", []) or []
                last_modified = getattr(model_info, "last_modified", None)

                # Fingerprint to check support
                try:
                    from ..architectures import fingerprint, get_adapter
                    from ..architectures.descriptor import SupportState

                    desc = fingerprint(model_id)

                    # Skip if too large
                    if desc.num_params and desc.num_params > max_params_b * 1e9:
                        continue

                    candidates.append(WatchCandidate(
                        model_id=model_id,
                        family=desc.family,
                        support_state=desc.support_state,
                        downloads=downloads,
                        likes=likes,
                        last_modified=str(last_modified) if last_modified else None,
                        tags=tags[:10],
                    ))

                except Exception as e:
                    _log.debug("fingerprint failed for %s: %s", model_id, e)
                    errors += 1

        except Exception as e:
            _log.error("failed to scan architecture %s: %s", arch, e)
            errors += 1

    # Sort: supported first, then by downloads
    from ..architectures.descriptor import SupportState
    priority = {
        SupportState.SUPPORTED: 0,
        SupportState.PARTIALLY_SUPPORTED: 1,
        SupportState.NEEDS_ADAPTER: 2,
        SupportState.UNSUPPORTED: 3,
    }
    candidates.sort(key=lambda c: (priority.get(c.support_state, 99), -c.downloads))

    filtered = len(candidates)
    _log.info("watcher: scanned=%d filtered=%d errors=%d", scanned, filtered, errors)

    return WatchResult(
        candidates=candidates,
        scanned=scanned,
        filtered=filtered,
        errors=errors,
    )


def watch_loop(
    *,
    interval_s: int = 3600,
    architectures: Optional[List[str]] = None,
    min_downloads: int = 100,
    max_params_b: float = 15.0,
    limit: int = 50,
    dry_run: bool = True,
    max_iterations: int = 0,
) -> None:
    """Continuously poll for new models and queue them for factory processing.

    When *max_iterations* is 0, runs forever. Otherwise runs that many polls.
    Currently only discovers and logs — factory dispatch is a future extension.
    """
    from ..architectures.descriptor import SupportState

    processed: Set[str] = set()
    iteration = 0

    while True:
        iteration += 1
        _log.info("watcher poll #%d at %s", iteration, datetime.now(timezone.utc).isoformat())

        result = discover_models(
            architectures=architectures,
            min_downloads=min_downloads,
            max_params_b=max_params_b,
            limit=limit,
            already_processed=processed,
        )

        supported = [c for c in result.candidates if c.support_state == SupportState.SUPPORTED]

        for c in supported:
            _log.info(
                "  candidate: %s (family=%s, downloads=%d)",
                c.model_id, c.family, c.downloads,
            )
            if not dry_run:
                _log.info("  -> would queue for factory: %s", c.model_id)
            processed.add(c.model_id)

        _log.info("poll complete: %d supported candidates", len(supported))

        if max_iterations > 0 and iteration >= max_iterations:
            break

        _log.info("sleeping %ds until next poll", interval_s)
        time.sleep(interval_s)
