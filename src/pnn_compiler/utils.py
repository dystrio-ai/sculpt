"""Shared utilities: logging, IO helpers."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import yaml
from rich.logging import RichHandler


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure root logger with rich console + optional file handler."""
    fmt = "%(message)s"
    handlers: list[logging.Handler] = [RichHandler(rich_tracebacks=True, markup=True)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt, handlers=handlers)
    return logging.getLogger("pnn")


log = setup_logging()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"Saved {path}")


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_tensor(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    log.info(f"Saved tensor → {path}  (shape={list(tensor.shape)})")


@contextmanager
def timer(label: str = ""):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    log.info(f"[bold cyan]{label}[/] took {elapsed:.2f}s")
