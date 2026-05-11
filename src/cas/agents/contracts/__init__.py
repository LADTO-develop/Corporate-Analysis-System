"""Stable input/output contracts shared by CAS agent entrypoints."""

from cas.agents.contracts.company_selection import (
    CompanySelectionAnalysis,
    CompanySelectionCompany,
    CompanySelectionError,
    CompanySelectionRequest,
    SelectionSource,
    build_agent_state_seed,
    build_company_id,
    build_company_selection_from_row,
    normalize_company_selection,
)

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
