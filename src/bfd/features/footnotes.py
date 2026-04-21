"""Footnotes-derived features — 주석 기반 파생변수.

Korean financial-statement footnotes tend to include boilerplate risk
disclosures but sometimes carry the first explicit mention of going-concern
doubt, litigation, or contingent liabilities that later materialise. We use
keyword-based extraction for deterministic pipeline behaviour; semantic
extraction is done by the RAG / LLM featurizer instead.

Mapping (신용평가과정 ↔ 주석):
    주석 → '공시 후 피드백' 단계의 우발부채·소송·규제 리스크 탐지.
"""

from __future__ import annotations

import re

import pandas as pd

from bfd.features.registry import feature

# ---------------------------------------------------------------------------
# Keyword lists — sourced from K-IFRS footnote conventions
# ---------------------------------------------------------------------------
GOING_CONCERN_KEYWORDS = [
    "계속기업",
    "계속기업 가정",
    "존속능력",
    "중대한 불확실성",
]

LITIGATION_KEYWORDS = [
    "소송",
    "피소",
    "제소",
    "가압류",
    "가처분",
]

CONTINGENT_KEYWORDS = [
    "우발부채",
    "지급보증",
    "손실충당",
    "연대보증",
]

REGULATORY_KEYWORDS = [
    "과징금",
    "행정처분",
    "영업정지",
    "감리",
    "지적사항",
]


def _contains_any(text: object, keywords: list[str]) -> int:
    if not isinstance(text, str) or not text:
        return 0
    pattern = "|".join(re.escape(k) for k in keywords)
    return 1 if re.search(pattern, text) else 0


@feature(
    source="footnotes",
    kind="boolean",
    description="주석에 '계속기업' 의심 문구 존재.",
)
def going_concern_mention(df: pd.DataFrame) -> pd.Series:
    col = df.get("going_concern_text", pd.Series("", index=df.index))
    return col.apply(lambda t: _contains_any(t, GOING_CONCERN_KEYWORDS))


@feature(
    source="footnotes",
    kind="boolean",
    description="주석에 중대한 소송/가압류 언급 존재.",
)
def litigation_mention(df: pd.DataFrame) -> pd.Series:
    col = df.get("litigation_text", pd.Series("", index=df.index))
    return col.apply(lambda t: _contains_any(t, LITIGATION_KEYWORDS))


@feature(
    source="footnotes",
    kind="boolean",
    description="주석에 우발부채/지급보증 언급 존재.",
)
def contingent_liability_mention(df: pd.DataFrame) -> pd.Series:
    col = df.get("contingent_liabilities_text", pd.Series("", index=df.index))
    return col.apply(lambda t: _contains_any(t, CONTINGENT_KEYWORDS))


@feature(
    source="footnotes",
    kind="numeric",
    description="주석 위험 키워드 전체 카운트 — 위 네 범주의 합.",
)
def footnote_risk_score(df: pd.DataFrame) -> pd.Series:
    parts = [
        going_concern_mention(df),
        litigation_mention(df),
        contingent_liability_mention(df),
    ]
    # Regulatory keywords — searched across any non-null text column
    reg = pd.Series(0, index=df.index, dtype=int)
    for col in ("going_concern_text", "litigation_text", "contingent_liabilities_text", "related_party_text"):
        if col in df.columns:
            reg = reg | df[col].apply(lambda t: _contains_any(t, REGULATORY_KEYWORDS))
    parts.append(reg)
    return sum(parts)  # type: ignore[return-value]


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every footnote feature and return a new DataFrame."""
    from bfd.features.registry import REGISTRY

    out = df[["corp_code", "fiscal_year"]].copy()
    for spec in REGISTRY.list_subset("kospi_v1"):
        if spec.source != "footnotes":
            continue
        out[spec.name] = spec.fn(df)
    return out
