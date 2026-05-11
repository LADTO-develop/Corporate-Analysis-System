"""Typed schemas for ECOS macro data and MacroMarketAgent outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MacroCycle = Literal["D", "M", "Q", "A"]
MacroDirection = Literal["upgrade", "neutral", "downgrade"]
MacroRiskLevel = Literal["low", "moderate", "high", "very_high"]
MacroStance = Literal["supportive", "neutral", "cautious", "stress"]


class EcosIndicatorSpec(BaseModel):
    """One ECOS indicator definition used for live collection."""

    code: str
    name_kr: str
    name_en: str
    stat_code: str
    cycle: MacroCycle
    item_code: str
    unit: str
    lookback_days: int = Field(gt=0)
    stale_after_days: int = Field(gt=0)
    description_kr: str


class DerivedMetricSpec(BaseModel):
    """One derived macro metric computed from fetched ECOS observations."""

    code: str
    name_kr: str
    name_en: str
    formula: str
    unit: str
    input_codes: list[str]
    description_kr: str


class EcosRegistry(BaseModel):
    """Registry describing all ECOS indicators and derived macro metrics."""

    schema_version: str
    source: str
    source_name_kr: str
    indicators: list[EcosIndicatorSpec]
    derived_metrics: list[DerivedMetricSpec]


class EcosObservation(BaseModel):
    """Latest available observation for one ECOS indicator."""

    code: str
    name_kr: str
    name_en: str
    value: float
    unit: str
    time: str
    observed_at: str
    cycle: MacroCycle
    stat_code: str
    item_code: str
    source: Literal["ECOS"] = "ECOS"
    lag_days: int
    description_kr: str


class DerivedMacroMetric(BaseModel):
    """Computed macro metric such as a credit or quality spread."""

    code: str
    name_kr: str
    name_en: str
    value: float
    unit: str
    formula: str
    input_codes: list[str]
    observed_at: str
    source: Literal["derived_from_ECOS"] = "derived_from_ECOS"
    description_kr: str


class MacroMarketContext(BaseModel):
    """Macro context bundle passed from data collection into the macro agent."""

    schema_version: str = "1.0"
    produced_at: str
    as_of_date: str
    source_name: str
    observations: list[EcosObservation]
    derived_metrics: list[DerivedMacroMetric]
    missing_indicators: list[str] = Field(default_factory=list)
    stale_indicators: list[str] = Field(default_factory=list)
    notes_kr: list[str] = Field(default_factory=list)


class MacroFactorSignal(BaseModel):
    """One interpreted macro factor that can affect credit-risk judgment."""

    code: str
    name_kr: str
    value: float
    unit: str
    direction: MacroDirection
    severity: MacroRiskLevel
    affected_variable_groups: list[str]
    interpretation_kr: str
    source_ref: str


class MacroGroupAdjustment(BaseModel):
    """Weight-guidance for a feature group under the current macro context."""

    variable_group: str
    direction: MacroDirection
    weight_multiplier: float = Field(gt=0.0)
    rationale_kr: str
    evidence_refs: list[str]


class MacroMarketAgentOutput(BaseModel):
    """Deterministic MacroMarketAgent interpretation result."""

    schema_version: str = "1.0"
    agent_name: str = "MacroMarketAgent"
    produced_at: str
    as_of_date: str
    stance: MacroStance
    macro_risk_level: MacroRiskLevel
    current_context_risk_delta: float
    macro_summary_kr: str
    key_macro_factors: list[MacroFactorSignal]
    group_adjustments: list[MacroGroupAdjustment]
    limitations_kr: list[str]
    source_context_refs: list[str]
    model_handling_note_kr: str
