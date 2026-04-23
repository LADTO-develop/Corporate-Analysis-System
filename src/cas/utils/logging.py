"""Structured logging with a standard-library fallback."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

try:
    import structlog
    from structlog.typing import Processor
except ImportError:  # pragma: no cover
    structlog = None  # type: ignore[assignment]
    Processor = Any  # type: ignore[misc]


def _env(primary: str, legacy: str, default: str) -> str:
    """Read a new env var name first and fall back to the legacy one."""
    return os.getenv(primary) or os.getenv(legacy) or default


def configure_logging(
    level: str | int | None = None,
    *,
    json_output: bool | None = None,
) -> None:
    """Configure logging with or without structlog."""
    if level is None:
        level = _env("CAS_LOG_LEVEL", "BFD_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    if structlog is None:
        return

    if json_output is None:
        json_output = not sys.stdout.isatty()

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Processor
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None, **initial_values: Any) -> Any:
    """Return a logger compatible with both structlog and stdlib logging."""
    if structlog is not None:
        logger = structlog.get_logger(name)
        return logger.bind(**initial_values) if initial_values else logger

    logger = logging.getLogger(name)
    return _StdlibBoundLogger(logger, initial_values)


class _StdlibBoundLogger:
    """Tiny adapter that mimics the bound-logger methods used in the project."""

    def __init__(self, logger: logging.Logger, context: dict[str, Any] | None = None) -> None:
        self._logger = logger
        self._context = context or {}

    def bind(self, **values: Any) -> _StdlibBoundLogger:
        return _StdlibBoundLogger(self._logger, {**self._context, **values})

    def debug(self, event: str, **values: Any) -> None:
        self._logger.debug(self._format(event, values))

    def info(self, event: str, **values: Any) -> None:
        self._logger.info(self._format(event, values))

    def warning(self, event: str, **values: Any) -> None:
        self._logger.warning(self._format(event, values))

    def error(self, event: str, **values: Any) -> None:
        self._logger.error(self._format(event, values))

    def _format(self, event: str, values: dict[str, Any]) -> str:
        merged = {**self._context, **values}
        if not merged:
            return event
        parts = ", ".join(f"{key}={value}" for key, value in merged.items())
        return f"{event} | {parts}"
