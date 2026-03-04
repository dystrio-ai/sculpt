"""Tests for centralized logging configuration."""

from __future__ import annotations

import logging
import os

import pytest


class TestConfigureLoggingDefault:
    """Default mode: Dystrio INFO, external libs WARNING/ERROR."""

    def test_dystrio_logger_info(self):
        from dystrio_sculpt.logging_utils import configure_logging

        configure_logging(quiet=False, verbose=False)
        assert logging.getLogger("dystrio_sculpt").level == logging.INFO

    def test_httpx_logger_warning(self):
        from dystrio_sculpt.logging_utils import configure_logging

        configure_logging(quiet=False, verbose=False)
        assert logging.getLogger("httpx").level >= logging.WARNING

    def test_httpcore_logger_warning(self):
        from dystrio_sculpt.logging_utils import configure_logging

        configure_logging(quiet=False, verbose=False)
        assert logging.getLogger("httpcore").level >= logging.WARNING

    def test_transformers_logger_error(self):
        from dystrio_sculpt.logging_utils import configure_logging

        configure_logging(quiet=False, verbose=False)
        assert logging.getLogger("transformers").level >= logging.ERROR

    def test_huggingface_hub_logger_suppressed(self):
        from dystrio_sculpt.logging_utils import configure_logging

        configure_logging(quiet=False, verbose=False)
        assert logging.getLogger("huggingface_hub").level >= logging.WARNING

    def test_datasets_logger_error(self):
        from dystrio_sculpt.logging_utils import configure_logging

        configure_logging(quiet=False, verbose=False)
        assert logging.getLogger("datasets").level >= logging.ERROR

    def test_hf_hub_verbosity_env(self):
        from dystrio_sculpt.logging_utils import configure_logging

        configure_logging(quiet=False, verbose=False)
        assert os.environ.get("HF_HUB_VERBOSITY") == "error"


class TestConfigureLoggingVerbose:
    """Verbose mode: everything at DEBUG."""

    def test_dystrio_logger_debug(self):
        from dystrio_sculpt.logging_utils import configure_logging

        configure_logging(quiet=False, verbose=True)
        assert logging.getLogger("dystrio_sculpt").level == logging.DEBUG

    def test_httpx_logger_debug(self):
        from dystrio_sculpt.logging_utils import configure_logging

        configure_logging(quiet=False, verbose=True)
        assert logging.getLogger("httpx").level == logging.DEBUG

    def test_transformers_logger_debug(self):
        from dystrio_sculpt.logging_utils import configure_logging

        configure_logging(quiet=False, verbose=True)
        assert logging.getLogger("transformers").level == logging.DEBUG

    def test_hf_hub_verbosity_env_debug(self):
        from dystrio_sculpt.logging_utils import configure_logging

        configure_logging(quiet=False, verbose=True)
        assert os.environ.get("HF_HUB_VERBOSITY") == "debug"

    def test_root_logger_debug(self):
        from dystrio_sculpt.logging_utils import configure_logging

        configure_logging(quiet=False, verbose=True)
        assert logging.getLogger().level == logging.DEBUG


class TestConfigureLoggingQuiet:
    """Quiet mode: only warnings/errors from Dystrio, errors from external."""

    def test_dystrio_logger_warning(self):
        from dystrio_sculpt.logging_utils import configure_logging

        configure_logging(quiet=True, verbose=False)
        assert logging.getLogger("dystrio_sculpt").level >= logging.WARNING

    def test_httpx_logger_error(self):
        from dystrio_sculpt.logging_utils import configure_logging

        configure_logging(quiet=True, verbose=False)
        assert logging.getLogger("httpx").level >= logging.ERROR

    def test_transformers_logger_error(self):
        from dystrio_sculpt.logging_utils import configure_logging

        configure_logging(quiet=True, verbose=False)
        assert logging.getLogger("transformers").level >= logging.ERROR

    def test_root_logger_warning(self):
        from dystrio_sculpt.logging_utils import configure_logging

        configure_logging(quiet=True, verbose=False)
        assert logging.getLogger().level >= logging.WARNING


class TestMutualExclusion:
    """--quiet and --verbose cannot both be True."""

    def test_raises_on_both(self):
        from dystrio_sculpt.logging_utils import configure_logging

        with pytest.raises(ValueError, match="mutually exclusive"):
            configure_logging(quiet=True, verbose=True)


class TestNoDuplicateHandlers:
    """Calling configure_logging multiple times should not add duplicate handlers."""

    def test_handler_count_stable(self):
        from dystrio_sculpt.logging_utils import configure_logging

        root = logging.getLogger()
        initial_count = len(root.handlers)
        configure_logging(quiet=False, verbose=False)
        count_after_first = len(root.handlers)
        configure_logging(quiet=False, verbose=False)
        count_after_second = len(root.handlers)
        assert count_after_second == count_after_first


class TestNoisyLoggersList:
    """Verify the noisy loggers list covers key offenders."""

    def test_known_noisy_loggers_suppressed(self):
        from dystrio_sculpt.logging_utils import _NOISY_LOGGERS

        expected = {"httpx", "httpcore", "huggingface_hub", "transformers", "datasets"}
        assert expected.issubset(set(_NOISY_LOGGERS))
