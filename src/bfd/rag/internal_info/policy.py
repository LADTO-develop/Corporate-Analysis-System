"""Security policy for the "internal information" LLM channel.

The project lets privileged users push confidential firm information
(private forecasts, undisclosed KPIs, M&A signals) into the LLM so the
scoring agent can factor it in. This creates real risk:

  1. Leakage into training data — solved by refusing any provider
     that can train on requests.
  2. Leakage across sessions / users — solved by per-session ephemeral
     stores and aggressive redaction at input and output.
  3. Compliance — every access is audit-logged (who, when, what class
     of fact was consulted).

This module implements the *policy*; ``redaction.py`` holds the concrete
pattern library.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from bfd.utils.io import ensure_dir
from bfd.utils.logging import get_logger

logger = get_logger(__name__)


class InternalInfoMode(str, Enum):
    """Modes for the internal-info channel, selected by ``BFD_INTERNAL_INFO_MODE``."""

    DISABLED = "disabled"
    READ_ONLY = "read_only"      # read previously ingested internal facts, no new input
    FULL = "full"                # accept new internal input during a session


class AccessDenied(RuntimeError):
    """Raised when the current mode forbids the attempted operation."""


@dataclass
class AuditEvent:
    """A single audit-log entry."""

    timestamp: str
    actor: str
    action: str                  # "read" | "write" | "refuse"
    fact_class: str              # "financial_forecast" | "m&a" | "personnel" | ...
    corp_code: str | None
    mode: str
    allowed: bool
    reason: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------
def current_mode() -> InternalInfoMode:
    raw = os.getenv("BFD_INTERNAL_INFO_MODE", "disabled").lower()
    try:
        return InternalInfoMode(raw)
    except ValueError:
        logger.warning("internal_info_invalid_mode", raw=raw, fallback="disabled")
        return InternalInfoMode.DISABLED


# ---------------------------------------------------------------------------
# Policy gates
# ---------------------------------------------------------------------------
def require_write_allowed() -> None:
    """Raises ``AccessDenied`` unless the current mode permits writes."""
    mode = current_mode()
    if mode is not InternalInfoMode.FULL:
        _emit_audit(
            action="refuse",
            fact_class="<write>",
            corp_code=None,
            allowed=False,
            reason=f"Write refused in mode={mode.value}",
        )
        raise AccessDenied(f"Internal-info write disallowed in mode={mode.value}")


def require_read_allowed() -> None:
    """Raises ``AccessDenied`` unless the current mode permits reads."""
    mode = current_mode()
    if mode is InternalInfoMode.DISABLED:
        _emit_audit(
            action="refuse",
            fact_class="<read>",
            corp_code=None,
            allowed=False,
            reason="Read refused; channel disabled",
        )
        raise AccessDenied("Internal-info channel is disabled")


def record_access(
    *,
    action: str,
    fact_class: str,
    corp_code: str | None,
    actor: str | None = None,
    extras: dict[str, Any] | None = None,
) -> None:
    """Emit a successful-access audit record."""
    _emit_audit(
        action=action,
        fact_class=fact_class,
        corp_code=corp_code,
        allowed=True,
        actor=actor,
        extras=extras or {},
    )


# ---------------------------------------------------------------------------
# Audit sink
# ---------------------------------------------------------------------------
def _emit_audit(
    *,
    action: str,
    fact_class: str,
    corp_code: str | None,
    allowed: bool,
    reason: str = "",
    actor: str | None = None,
    extras: dict[str, Any] | None = None,
) -> None:
    event = AuditEvent(
        timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        actor=actor or os.getenv("USER", "unknown"),
        action=action,
        fact_class=fact_class,
        corp_code=corp_code,
        mode=current_mode().value,
        allowed=allowed,
        reason=reason,
        extras=extras or {},
    )
    sink = os.getenv("BFD_AUDIT_SINK", "stdout")
    if sink == "stdout":
        logger.info("internal_info_audit", **event.__dict__)
    elif sink.startswith("file://"):
        path = Path(sink.replace("file://", ""))
        ensure_dir(path.parent)
        with open(path, "a", encoding="utf-8") as f:
            import json

            f.write(json.dumps(event.__dict__, ensure_ascii=False) + "\n")
    else:
        logger.warning("internal_info_audit_sink_unknown", sink=sink)
