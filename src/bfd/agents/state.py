"""LangGraph state schema.

We use a ``TypedDict`` with ``Annotated[..., reducer]`` fields so that:
    * scalar fields (current firm, market, fiscal year) overwrite by default
    * accumulator fields (audit trail, committee opinions) append

This mirrors the recommended pattern from the LangGraph docs — see
https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

Market = Literal["KOSPI", "KOSDAQ"]
Verdict = Literal["borderline", "healthy", "uncertain"]
NodeName = Literal[
    "data",
    "feature",
    "base_prediction",
    "macro_overlay",
    "news_overlay",
    "committee",
    "report",
]


# ---------------------------------------------------------------------------
# Pydantic sub-models — validated payloads inside the TypedDict state
# ---------------------------------------------------------------------------
class AuditEntry(BaseModel):
    """A structured audit-trail record emitted by every node."""

    node: NodeName
    timestamp: str
    summary: str
    payload_ref: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class BasePrediction(BaseModel):
    """Output of a single base model (TabPFN / LightGBM / EBM)."""

    model_name: str
    proba_borderline: float = Field(ge=0.0, le=1.0)
    feature_top_k: list[tuple[str, float]] = Field(default_factory=list)


class MacroOverlay(BaseModel):
    """Macro adjustment to the base prediction."""

    credit_spread: float | None = None
    real_rate: float | None = None
    fx_volatility_30d: float | None = None
    # Directional adjustment to P(borderline): positive = make more risky
    adjustment: float = 0.0
    rationale_ko: str = ""


class NewsOverlay(BaseModel):
    """News-derived adjustment to the base prediction."""

    firm_specific_sentiment: float = 0.0
    industry_sentiment: float = 0.0
    event_risk_flag: bool = False
    going_concern_mention: bool = False
    adjustment: float = 0.0
    rationale_ko: str = ""


class CommitteeOpinion(BaseModel):
    """One opinion from a bias-removal technique (6-hats, devil's advocate, …)."""

    technique: Literal[
        "devils_advocate",
        "six_hats",
        "nominal_group",
        "step_ladder",
    ]
    role: str = ""  # 모자 색 / 역할 등 기법 내부 식별자
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    rationale_ko: str = ""


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------
def append_audit(
    current: list[AuditEntry] | None, new: list[AuditEntry] | None
) -> list[AuditEntry]:
    """Append-only reducer for the audit log (never overwrites)."""
    current = current or []
    new = new or []
    return [*current, *new]


def append_opinions(
    current: list[CommitteeOpinion] | None, new: list[CommitteeOpinion] | None
) -> list[CommitteeOpinion]:
    """Append-only reducer for committee opinions."""
    current = current or []
    new = new or []
    return [*current, *new]


def merge_dict(current: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any]:
    """Dict-merge reducer — used for ``base_predictions`` keyed by model name."""
    out: dict[str, Any] = dict(current or {})
    out.update(new or {})
    return out


# ---------------------------------------------------------------------------
# The state object itself
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    """Full state flowing through the LangGraph pipeline."""

    # --- identity --------------------------------------------------------
    corp_code: str
    market: Market
    fiscal_year: int
    target_year: int

    # --- data / feature stages ------------------------------------------
    raw_financials: dict[str, Any]
    features: dict[str, float | int | bool]

    # --- prediction stages ----------------------------------------------
    # Each key is a base-model name; ``add``/``merge`` semantics via reducer
    base_predictions: Annotated[dict[str, BasePrediction], merge_dict]
    macro_overlay: MacroOverlay
    news_overlay: NewsOverlay
    ensemble_proba: float

    # --- committee ------------------------------------------------------
    committee_opinions: Annotated[list[CommitteeOpinion], append_opinions]
    final_verdict: Verdict
    final_confidence: float

    # --- audit / reporting ---------------------------------------------
    audit: Annotated[list[AuditEntry], append_audit]

    # --- plumbing -------------------------------------------------------
    # Free-form payload pointers (parquet paths, chroma ids, etc.) so large
    # blobs don't bloat the state itself.
    artifacts: Annotated[dict[str, str], merge_dict]

    # If data_node determines the data is insufficient, it sets this flag and
    # the conditional edge routes straight to the report node.
    insufficient_data: bool


__all__ = [
    "AgentState",
    "AuditEntry",
    "BasePrediction",
    "CommitteeOpinion",
    "MacroOverlay",
    "NewsOverlay",
    "Market",
    "NodeName",
    "Verdict",
    "append_audit",
    "append_opinions",
    "merge_dict",
]
