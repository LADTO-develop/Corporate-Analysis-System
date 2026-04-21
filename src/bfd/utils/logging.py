"""Structured logging setup for the Borderline Firm Detector.

Uses structlog for machine-parseable JSON logs in production and
human-friendly coloured output in development.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog
from structlog.typing import Processor


def configure_logging(
    level: str | int | None = None,
    *,
    json_output: bool | None = None,
) -> None:
    """Configure structlog + stdlib logging in a coherent way.

    Args:
        level: log level name or integer. Falls back to ``BFD_LOG_LEVEL`` env
            variable, then ``INFO``.
        json_output: force JSON output. Defaults to True when stdout is not a
            TTY (i.e. running in CI / piped).
    """
    if level is None:
        level = os.getenv("BFD_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if json_output is None:
        json_output = not sys.stdout.isatty()

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging through structlog formatting as well
    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str | None = None, **initial_values: Any) -> structlog.BoundLogger:
    """Return a structured logger bound with the given initial key/value pairs."""
    logger: structlog.BoundLogger = structlog.get_logger(name)
    if initial_values:
        logger = logger.bind(**initial_values)
    return logger
