"""Centralized logging configuration for the Dystrio CLI.

Suppresses noisy HF / httpx / datasets output by default while keeping
Dystrio's own logs visible.  Provides --quiet and --verbose modes.
"""

from __future__ import annotations

import logging
import os

LOG_FORMAT = "%(asctime)s [%(name)s] %(message)s"
LOG_DATEFMT = "%H:%M:%S"

# External loggers that are noisy at INFO/DEBUG
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "transformers",
    "datasets",
    "datasets.builder",
    "filelock",
    "urllib3",
    "fsspec",
)


def configure_logging(*, quiet: bool = False, verbose: bool = False) -> None:
    """Set up logging levels for the entire CLI process.

    - **default** (neither flag): Dystrio INFO, external libs WARNING/ERROR.
    - **quiet**: Dystrio WARNING, external libs ERROR, progress bars disabled.
    - **verbose**: Dystrio DEBUG, external libs DEBUG, full request tracing.
    """
    if quiet and verbose:
        raise ValueError("--quiet and --verbose are mutually exclusive")

    # ── Python logging hierarchy ──────────────────────────────────────────
    if quiet:
        root_level = logging.WARNING
        dystrio_level = logging.WARNING
        external_level = logging.ERROR
    elif verbose:
        root_level = logging.DEBUG
        dystrio_level = logging.DEBUG
        external_level = logging.DEBUG
    else:
        root_level = logging.INFO
        dystrio_level = logging.INFO
        external_level = logging.WARNING

    root = logging.getLogger()
    root.setLevel(root_level)

    # Remove any pre-existing handlers to avoid duplicates
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        root.addHandler(handler)

    # Dystrio's own loggers
    for name in ("dystrio_sculpt", "dystrio.sculpt", "dystrio.bench", "dystrio.audit"):
        logging.getLogger(name).setLevel(dystrio_level)

    # External noisy loggers
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(external_level)

    # ── Transformers verbosity + progress bars ────────────────────────────
    try:
        from transformers.utils import logging as tlog
        if quiet:
            tlog.set_verbosity_error()
            tlog.disable_progress_bar()
        elif verbose:
            tlog.set_verbosity_debug()
            tlog.enable_progress_bar()
        else:
            tlog.set_verbosity_error()
            tlog.enable_progress_bar()
    except Exception:
        pass

    # ── Datasets verbosity ────────────────────────────────────────────────
    try:
        from datasets.utils.logging import (
            set_verbosity_error as ds_error,
            set_verbosity_info as ds_info,
            set_verbosity_debug as ds_debug,
            disable_progress_bar as ds_disable_pb,
            enable_progress_bar as ds_enable_pb,
        )
        if quiet:
            ds_error()
            ds_disable_pb()
        elif verbose:
            ds_debug()
            ds_enable_pb()
        else:
            ds_error()
            ds_enable_pb()
    except Exception:
        pass

    # ── HF Hub in-process env override ────────────────────────────────────
    if verbose:
        os.environ["HF_HUB_VERBOSITY"] = "debug"
    else:
        os.environ["HF_HUB_VERBOSITY"] = "error"
