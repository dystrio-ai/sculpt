"""Dataset logger: appends DatasetRecord entries to JSONL file.

Thread-safe, append-only. Each line is one self-contained JSON record.
Quality-gated: factory records whose default tier exceeds the PPL ceiling
are rejected before they touch the file.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import List, Optional

from .schema import DatasetRecord

_log = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = "dystrio_efficiency_dataset.jsonl"


def _quality_tags(record: DatasetRecord) -> List[str]:
    """Tag records with quality metadata for downstream filtering.

    These are NOT rejection criteria — every data point is useful for
    meta-learning. Tags help consumers decide what to surface publicly
    vs. what's internal-only training signal.
    """
    tags: List[str] = []
    if record.source != "factory":
        return tags

    if not record.tiers:
        tags.append("no_tiers")
        return tags

    default_tier = next((t for t in record.tiers if "default" in t.name), None)
    if default_tier is None:
        default_tier = record.tiers[0]

    if default_tier.ppl_ratio > 2.0:
        tags.append("high_ppl_default")

    if len(record.tiers) < 2:
        tags.append("single_tier")

    if default_tier.ppl_ratio > 5.0:
        tags.append("near_collapse")

    return tags


class DatasetLogger:
    """Append-only logger for the Dystrio Efficiency Dataset."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or DEFAULT_DATASET_PATH)
        self._lock = threading.Lock()

    def log(self, record: DatasetRecord, strict: bool = True) -> None:
        """Validate, tag, and append a record.

        Schema validation (strict) rejects structurally broken records.
        Quality tagging annotates records for downstream filtering but
        never rejects — every data point is useful for meta-learning,
        including collapse points and incomplete searches.
        """
        issues = record.validate()
        if issues:
            if strict and record.source == "factory":
                raise ValueError(
                    f"factory record failed rich-record contract "
                    f"({len(issues)} issues): {issues}"
                )
            _log.warning("dataset record has validation issues: %s", issues)

        tags = _quality_tags(record)
        if tags:
            _log.info("quality tags: %s", tags)
            if not record.error_category:
                record.error_category = ",".join(tags)

        line = record.to_json()
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(line + "\n")

        _log.info("logged dataset record: run_id=%s model=%s tiers=%d",
                   record.run_id, record.model_id, len(record.tiers))

    def read_all(self) -> List[DatasetRecord]:
        """Read all records from the dataset file."""
        if not self.path.exists():
            return []
        records = []
        with open(self.path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    records.append(DatasetRecord.from_dict(d))
                except (json.JSONDecodeError, TypeError) as e:
                    _log.warning("skipping invalid record at line %d: %s", line_num, e)
        return records

    def count(self) -> int:
        """Count records without loading all into memory."""
        if not self.path.exists():
            return 0
        n = 0
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    n += 1
        return n
