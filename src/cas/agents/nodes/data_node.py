"""Select a company and validate the processed-company input shape."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from cas.agents.state import AgentState, AuditEntry
from cas.utils.io import read_yaml
from cas.utils.logging import get_logger

logger = get_logger(__name__)

_PROFILE_ROOT = Path("data/input/companies")
_FEATURE_MASTER_PATH = Path("data/input/credit_43_features/feature_43_master.csv")
_FEATURE_INFERENCE_2026_PATH = Path("data/input/credit_43_features/feature_43_inference_2026.csv")
_PEER_PERCENTILES_PATH = Path("data/outputs/dashboard/feature_43_mvp/peer_percentiles.csv")
_REQUIRED_FINANCIALS = {
    "revenue_growth_pct",
    "operating_margin_pct",
    "debt_to_equity",
    "current_ratio",
    "free_cash_flow_margin_pct",
    "interest_coverage",
}
_REQUIRED_QUALITATIVE = {"governance_score", "product_momentum_score"}


def run(state: AgentState) -> dict[str, Any]:
    """Load the selected company profile from the processed-company list."""
    company_id = str(state["company_id"])
    profile_path = _PROFILE_ROOT / f"{company_id}.yaml"
    logger.info("data_node_run", company_id=company_id, path=str(profile_path))

    if not profile_path.exists():
        dataset_row = _resolve_feature_row(
            company_id=company_id,
            analysis_year=int(state.get("analysis_year") or 0),
        )
        if dataset_row is None:
            audit = AuditEntry(
                node="data",
                timestamp=_now(),
                summary=f"Company profile not found: {profile_path}",
            )
            return {"insufficient_data": True, "audit": [audit]}
        return _dataset_backed_payload(dataset_row)

    profile = read_yaml(profile_path)
    company = profile.get("company", {})
    financials = profile.get("financials", {})
    qualitative = profile.get("qualitative", {})
    analysis_year = int(profile.get("analysis_year") or state.get("analysis_year") or 0)
    missing = sorted(
        [
            *(key for key in _REQUIRED_FINANCIALS if key not in financials),
            *(key for key in _REQUIRED_QUALITATIVE if key not in qualitative),
        ]
    )
    if missing:
        audit = AuditEntry(
            node="data",
            timestamp=_now(),
            summary=f"Missing required input fields: {missing}",
            metrics={"missing_fields": float(len(missing))},
        )
        return {"insufficient_data": True, "audit": [audit]}

    audit = AuditEntry(
        node="data",
        timestamp=_now(),
        summary=(
            "Selected company loaded from processed-company list: "
            f"{company.get('name', company_id)}"
        ),
        metrics={"n_financial_fields": float(len(financials))},
    )
    return {
        "company_name": company.get("name", company_id),
        "market": company.get("market", state.get("market", "UNKNOWN")),
        "analysis_year": analysis_year,
        "company_profile": profile,
        "processed_company": {
            "company_id": company.get("id", company_id),
            "company_name": company.get("name", company_id),
            "market": company.get("market", state.get("market", "UNKNOWN")),
            "analysis_year": analysis_year,
            "source": str(profile_path),
        },
        "processed_company_list_ref": str(_PROFILE_ROOT),
        "raw_financials": financials,
        "insufficient_data": False,
        "audit": [audit],
    }


def has_enough_data(state: AgentState) -> Literal["enough", "insufficient"]:
    """Conditional-edge predicate referenced by the graph config."""
    return "insufficient" if state.get("insufficient_data") else "enough"


@lru_cache(maxsize=1)
def _load_feature_master() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in (_FEATURE_MASTER_PATH, _FEATURE_INFERENCE_2026_PATH):
        if not path.exists():
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
        frame["__source_path"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No feature input found at {_FEATURE_MASTER_PATH} or {_FEATURE_INFERENCE_2026_PATH}"
        )
    return pd.concat(frames, ignore_index=True, sort=False)


@lru_cache(maxsize=1)
def _load_peer_percentiles() -> pd.DataFrame | None:
    if not _PEER_PERCENTILES_PATH.exists():
        return None
    return pd.read_csv(_PEER_PERCENTILES_PATH, encoding="utf-8-sig", dtype={"stock_code": str})


def _resolve_feature_row(company_id: str, analysis_year: int) -> dict[str, Any] | None:
    master = _load_feature_master().copy()
    normalized_company_id = company_id.strip()
    numeric_company_id = normalized_company_id.lstrip("0") or "0"

    stock_codes = master["stock_code"].astype(str)
    stock_codes_no_zero = stock_codes.str.lstrip("0").replace("", "0")
    company_names = master["corp_name"].astype(str)

    matches = master.loc[
        (stock_codes == normalized_company_id)
        | (stock_codes_no_zero == numeric_company_id)
        | (company_names == normalized_company_id)
    ].copy()
    if matches.empty:
        return None

    if analysis_year > 0:
        eval_matches = matches.loc[matches["eval_year"] == analysis_year]
        if not eval_matches.empty:
            matches = eval_matches
        else:
            fiscal_matches = matches.loc[matches["fiscal_year"] == analysis_year]
            if not fiscal_matches.empty:
                matches = fiscal_matches

    row = matches.sort_values(["fiscal_year", "eval_year"]).iloc[-1]
    return {key: (None if pd.isna(value) else value) for key, value in row.to_dict().items()}


def _dataset_backed_payload(dataset_row: dict[str, Any]) -> dict[str, Any]:
    company_name = str(dataset_row.get("corp_name") or dataset_row.get("stock_code") or "unknown")
    market = str(dataset_row.get("market") or "UNKNOWN")
    stock_code = str(dataset_row.get("stock_code") or "unknown")
    fiscal_year = int(dataset_row.get("fiscal_year") or 0)
    analysis_year = int(dataset_row.get("eval_year") or fiscal_year)
    size_group = str(dataset_row.get("firm_size_group") or "unknown")
    industry = str(dataset_row.get("industry_macro_category") or "unknown")
    source_path = str(dataset_row.get("__source_path") or _FEATURE_MASTER_PATH)
    peer_rows = _resolve_peer_rows(stock_code=stock_code, fiscal_year=fiscal_year)

    summary = (
        f"{industry} 업종의 {size_group} 상장기업이며, "
        f"{fiscal_year} 회계연도 기준 정량 예측 입력 데이터를 불러왔습니다."
    )
    audit = AuditEntry(
        node="data",
        timestamp=_now(),
        summary=f"Loaded feature-master row for {company_name} ({stock_code})",
        metrics={"fiscal_year": float(fiscal_year), "analysis_year": float(analysis_year)},
    )
    return {
        "company_name": company_name,
        "market": market,
        "analysis_year": analysis_year,
        "company_profile": {
            "company": {
                "id": stock_code,
                "name": company_name,
                "market": market,
                "summary": summary,
            },
            "financials": {},
            "qualitative": {},
            "market_context": {},
        },
        "processed_company": {
            "company_id": stock_code,
            "company_name": company_name,
            "market": market,
            "analysis_year": analysis_year,
            "fiscal_year": fiscal_year,
            "source": source_path,
        },
        "processed_company_list_ref": source_path,
        "raw_financials": {},
        "source_feature_row": dataset_row,
        "peer_comparison_rows": peer_rows,
        "insufficient_data": False,
        "audit": [audit],
    }


def _resolve_peer_rows(*, stock_code: str, fiscal_year: int) -> list[dict[str, Any]]:
    peer_percentiles = _load_peer_percentiles()
    if peer_percentiles is None:
        return []

    normalized_stock_code = stock_code.strip()
    numeric_company_id = normalized_stock_code.lstrip("0") or "0"
    stock_codes = peer_percentiles["stock_code"].astype(str)
    stock_codes_no_zero = stock_codes.str.lstrip("0").replace("", "0")

    matched = peer_percentiles.loc[
        ((stock_codes == normalized_stock_code) | (stock_codes_no_zero == numeric_company_id))
        & (peer_percentiles["fiscal_year"] == fiscal_year)
    ].copy()
    if matched.empty:
        return []

    return [
        {key: (None if pd.isna(value) else value) for key, value in row.items()}
        for row in matched.to_dict(orient="records")
    ]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
