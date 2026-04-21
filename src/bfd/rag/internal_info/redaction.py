"""Redaction utilities for the internal-info channel.

Removes or masks Korean PII (주민등록번호, 사업자번호, 계좌번호, 휴대전화번호)
and English-format emails / card numbers before passing text to an LLM.
Rules are intentionally conservative — false-positives are fine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------

# 주민등록번호: 6자리-7자리 (8번째 자리가 성별 코드 1-8)
_RRN = re.compile(r"\b\d{6}[-\s]?[1-8]\d{6}\b")

# 사업자등록번호: 3자리-2자리-5자리
_BRN = re.compile(r"\b\d{3}-\d{2}-\d{5}\b")

# 법인등록번호: 6자리-7자리 (두 번째 블록이 0~6으로 시작)
_CRN = re.compile(r"\b\d{6}-[0-6]\d{6}\b")

# 휴대전화번호 (Korean mobile): 010 / 011 / 016-19
_PHONE = re.compile(r"\b01[016789][-\s]?\d{3,4}[-\s]?\d{4}\b")

# 국내 계좌번호(은행별 형식이 다양) — 9~14자리 연속 숫자 or 하이픈
_BANK_ACCOUNT = re.compile(r"\b\d{2,6}[-\s]\d{2,6}[-\s]\d{2,7}\b")

# Email
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# Credit card: 13-19 digits with optional separators
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


@dataclass
class RedactionResult:
    """Output of ``redact`` — redacted text plus counts per class."""

    text: str
    counts: dict[str, int]


def redact(text: str) -> RedactionResult:
    """Mask PII in ``text`` with class labels, return both redacted text and counts."""
    counts: dict[str, int] = {}

    def _sub(pattern: re.Pattern[str], label: str, s: str) -> str:
        def _replace(_m: re.Match[str]) -> str:
            counts[label] = counts.get(label, 0) + 1
            return f"[{label}]"

        return pattern.sub(_replace, s)

    out = text
    out = _sub(_RRN, "RRN", out)
    out = _sub(_CRN, "CRN", out)
    out = _sub(_BRN, "BRN", out)
    out = _sub(_PHONE, "PHONE", out)
    out = _sub(_CARD, "CARD", out)
    out = _sub(_BANK_ACCOUNT, "ACCOUNT", out)
    out = _sub(_EMAIL, "EMAIL", out)

    return RedactionResult(text=out, counts=counts)
