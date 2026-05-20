"""Input contract helpers for web-listed company selections."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from cas.agents.state import AgentState

Market = Literal["KOSPI", "KOSDAQ"]
SelectionSource = Literal["web_listing", "csv_upload", "manual_input"]

_INVISIBLE_CHARS = re.compile(r"[\u200b-\u200f\ufeff]")


class CompanySelectionError(ValueError):
    """Validation error with a stable code for API responses."""

    def __init__(self, code: str, message: str) -> None:
        """Store a stable error code and readable message."""
        self.code = code
        super().__init__(message)


class CompanySelectionCompany(BaseModel):
    """Company identifier block selected from a web listing."""

    model_config = ConfigDict(extra="forbid")

    market: Market
    stock_code: str = Field(min_length=6, max_length=6)
    corp_name: str = Field(min_length=1)
    corp_code: str | None = None

    @field_validator("market", mode="before")
    @classmethod
    def normalize_market(cls, value: object) -> str:
        """Normalize market labels accepted from browser UI controls."""
        market = _clean_text(value).upper()
        if market not in {"KOSPI", "KOSDAQ"}:
            raise ValueError("market must be KOSPI or KOSDAQ")
        return market

    @field_validator("stock_code", mode="before")
    @classmethod
    def normalize_stock_code(cls, value: object) -> str:
        """Normalize Korean stock codes to zero-padded six-digit strings."""
        text = _clean_text(_normalize_numeric_text(value))
        if not text.isdigit():
            raise ValueError("stock_code must contain digits only")
        if len(text) > 6:
            raise ValueError("stock_code must be at most six digits")
        return text.zfill(6)

    @field_validator("corp_name", mode="before")
    @classmethod
    def normalize_corp_name(cls, value: object) -> str:
        """Trim browser text input noise from company names."""
        text = _clean_text(value)
        if not text:
            raise ValueError("corp_name is required")
        return text

    @field_validator("corp_code", mode="before")
    @classmethod
    def normalize_corp_code(cls, value: object) -> str | None:
        """Normalize optional DART corporate codes."""
        if _is_blank(value):
            return None
        text = _clean_text(_normalize_numeric_text(value))
        if not text:
            return None
        if not text.isdigit():
            raise ValueError("corp_code must contain digits only")
        return text.zfill(8)


class CompanySelectionAnalysis(BaseModel):
    """Analysis-year block for the selected company snapshot."""

    model_config = ConfigDict(extra="forbid")

    fiscal_year: int | None = Field(default=None, ge=1900, le=2100)
    eval_year: int | None = Field(default=None, ge=1900, le=2100)

    @model_validator(mode="after")
    def validate_year_pair(self) -> CompanySelectionAnalysis:
        """Keep fiscal and evaluation year aligned when both are supplied."""
        if (
            self.fiscal_year is not None
            and self.eval_year is not None
            and self.eval_year != self.fiscal_year + 1
        ):
            raise ValueError("eval_year must equal fiscal_year + 1")
        return self


class CompanySelectionRequest(BaseModel):
    """Stable request shape passed from web listings into the pipeline."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: _default_request_id(), min_length=1)
    source: SelectionSource = "web_listing"
    selected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    as_of_date: date = Field(default_factory=date.today)
    company: CompanySelectionCompany
    analysis: CompanySelectionAnalysis = Field(default_factory=CompanySelectionAnalysis)

    @field_validator("request_id", mode="before")
    @classmethod
    def normalize_request_id(cls, value: object) -> str:
        """Trim request IDs while keeping caller-provided identifiers stable."""
        text = _clean_text(value)
        if not text:
            raise ValueError("request_id is required")
        return text

    @model_validator(mode="after")
    def validate_as_of_year(self) -> CompanySelectionRequest:
        """Reject feature snapshots that are not yet observable at the cut-off date."""
        fiscal_year = self.analysis.fiscal_year
        eval_year = self.analysis.eval_year
        if fiscal_year is not None and fiscal_year > self.as_of_date.year:
            raise ValueError("fiscal_year cannot be later than as_of_date year")
        if fiscal_year is None and eval_year is not None and eval_year > self.as_of_date.year + 1:
            raise ValueError("eval_year cannot be more than one year after as_of_date year")
        return self


def normalize_company_selection(
    payload: CompanySelectionRequest | Mapping[str, Any],
) -> CompanySelectionRequest:
    """Validate and normalize a raw company-selection payload."""
    if isinstance(payload, CompanySelectionRequest):
        return payload
    try:
        return CompanySelectionRequest.model_validate(payload)
    except ValidationError as error:
        raise CompanySelectionError(_error_code(error), str(error)) from error


def build_company_id(selection: CompanySelectionRequest) -> str:
    """Build the deterministic company snapshot ID used by downstream artifacts."""
    fiscal_year = selection.analysis.fiscal_year
    if fiscal_year is None and selection.analysis.eval_year is not None:
        fiscal_year = selection.analysis.eval_year - 1
    if fiscal_year is None:
        return selection.company.stock_code
    return f"{selection.company.market}-{selection.company.stock_code}-{fiscal_year}"


def build_agent_state_seed(
    payload: CompanySelectionRequest | Mapping[str, Any],
    *,
    base_state: Mapping[str, Any] | None = None,
) -> AgentState:
    """Build a graph initial state from a normalized company-selection request."""
    selection = normalize_company_selection(payload)
    analysis_year = selection.analysis.eval_year or (
        selection.analysis.fiscal_year + 1 if selection.analysis.fiscal_year else 0
    )
    seed: AgentState = {
        "company_id": build_company_id(selection),
        "company_name": selection.company.corp_name,
        "market": selection.company.market,
        "analysis_year": analysis_year,
        "company_selection": selection.model_dump(mode="json", exclude_none=True),
        "base_assessments": {},
        "committee_reviews": [],
        "agent_outputs": [],
        "agent_summary": {},
        "committee_view": {},
        "audit": [],
        "artifacts": {},
        "insufficient_data": False,
    }
    if base_state:
        seed.update(base_state)
        seed["company_selection"] = selection.model_dump(mode="json", exclude_none=True)
    return seed


def build_company_selection_from_row(
    row: Mapping[str, Any],
    *,
    request_id: str | None = None,
    selected_at: datetime | str | None = None,
    as_of_date: date | str | None = None,
    source: SelectionSource = "web_listing",
) -> dict[str, Any]:
    """Convert a listed-company row into the external input contract."""
    raw = dict(row)
    payload: dict[str, Any] = {
        "source": source,
        "company": {
            "market": raw.get("market"),
            "stock_code": raw.get("stock_code"),
            "corp_name": raw.get("corp_name"),
            "corp_code": raw.get("corp_code"),
        },
        "analysis": {
            "fiscal_year": _optional_int(raw.get("fiscal_year")),
            "eval_year": _optional_int(raw.get("eval_year")),
        },
    }
    if request_id is not None:
        payload["request_id"] = request_id
    if selected_at is not None:
        payload["selected_at"] = selected_at
    if as_of_date is not None:
        payload["as_of_date"] = as_of_date
    return normalize_company_selection(payload).model_dump(mode="json", exclude_none=True)


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return _INVISIBLE_CHARS.sub("", str(value)).strip()


def _normalize_numeric_text(value: object) -> object:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return value


def _optional_int(value: object) -> int | None:
    if _is_blank(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, int):
        return value
    text = _clean_text(value)
    if not text:
        return None
    return int(float(text)) if "." in text else int(text)


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return _clean_text(value) == ""


def _error_code(error: ValidationError) -> str:
    text = str(error).lower()
    if "field required" in text or "is required" in text:
        return "missing_required_field"
    if "stock_code" in text or "corp_code" in text:
        return "invalid_identifier"
    if "fiscal_year cannot be later" in text or "eval_year cannot be more than" in text:
        return "as_of_date_violation"
    if "eval_year must equal" in text:
        return "invalid_analysis_year"
    return "invalid_company_selection"


def _default_request_id() -> str:
    return datetime.now(UTC).strftime("req_%Y%m%d_%H%M%S_%f")


__all__ = [
    "CompanySelectionAnalysis",
    "CompanySelectionCompany",
    "CompanySelectionError",
    "CompanySelectionRequest",
    "SelectionSource",
    "build_agent_state_seed",
    "build_company_id",
    "build_company_selection_from_row",
    "normalize_company_selection",
]
