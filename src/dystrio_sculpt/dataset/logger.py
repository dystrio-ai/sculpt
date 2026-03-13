"""Dataset logger: appends DatasetRecord entries to JSONL file.

Thread-safe, append-only. Each line is one self-contained JSON record.
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


class DatasetLogger:
    """Append-only logger for the Dystrio Efficiency Dataset."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or DEFAULT_DATASET_PATH)
        self._lock = threading.Lock()

    def log(self, record: DatasetRecord, strict: bool = True) -> None:
        """Validate and append a record.

        For source="factory" records with strict=True, validation failures
        raise ValueError instead of silently appending junk.
        """
        issues = record.validate()
        if issues:
            if strict and record.source == "factory":
                raise ValueError(
                    f"factory record failed rich-record contract "
                    f"({len(issues)} issues): {issues}"
                )
            _log.warning("dataset record has validation issues: %s", issues)

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
