"""Select a company and validate the processed-company input shape."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from cas.agents.contracts import (
    CompanySelectionError,
    CompanySelectionRequest,
    normalize_company_selection,
)
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


def _looks_like_stock_code(value: object) -> bool:
    """Return whether the value can be treated as a Korean stock code."""
    text = str(value).strip()
    return text.isdigit() and len(text) <= 6


def _normalize_stock_code(value: object) -> str:
    """Normalize listed-company stock codes to the six-digit display format."""
    text = str(value).strip()
    if _looks_like_stock_code(text):
        return text.zfill(6)
    return text


def run(state: AgentState) -> dict[str, Any]:
    """Load the selected company profile from the processed-company list."""
    selection_payload = state.get("company_selection")
    if selection_payload:
        return _run_company_selection(selection_payload)

    company_id = str(state["company_id"])
    profile_path = _PROFILE_ROOT / f"{company_id}.yaml"
    logger.info("data_node_run", company_id=company_id, path=str(profile_path))

    if not profile_path.exists():
        # Stage 1/2의 기본 입력은 feature master다. YAML 회사 프로필이 없으면
        # 학습·추론용 정형 입력셋에서 가장 가까운 company-year row를 찾아 계속 진행한다.
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
        # YAML 입력이 없는 운영 경로에서는 정형 입력셋 row 하나를 "회사 프로필"처럼 취급한다.
        # 여기서 만든 payload는 Stage 1 모델 실행과 Stage 2 에이전트 해석의 공통 출발점이 된다.
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


def _run_company_selection(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        selection = normalize_company_selection(payload)
    except CompanySelectionError as error:
        return _insufficient_selection(error.code, str(error))

    dataset_row, error_code = _resolve_feature_row_for_selection(selection)
    if dataset_row is None:
        return _insufficient_selection(
            error_code or "snapshot_not_found",
            (
                "Could not resolve company selection to a feature snapshot: "
                f"{selection.company.market} {selection.company.stock_code} "
                f"{selection.company.corp_name}"
            ),
        )
    return _dataset_backed_payload(dataset_row, company_selection=selection)


def _insufficient_selection(code: str, message: str) -> dict[str, Any]:
    audit = AuditEntry(
        node="data",
        timestamp=_now(),
        summary=f"Company selection rejected ({code}): {message}",
    )
    return {
        "insufficient_data": True,
        "selection_errors": [code],
        "audit": [audit],
    }


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
    # company_id는 stock_code 또는 corp_name으로 들어올 수 있어서 둘 다 허용한다.
    # 찾은 후보 중 가장 마지막 company-year row를 선택해 이후 노드의 기준 입력으로 쓴다.
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
        # analysis_year는 보통 eval_year 기준 요청이므로 먼저 eval_year를 맞춰 보고,
        # 없을 때만 fiscal_year fallback을 허용한다.
        eval_matches = matches.loc[matches["eval_year"] == analysis_year]
        if not eval_matches.empty:
            matches = eval_matches
        else:
            fiscal_matches = matches.loc[matches["fiscal_year"] == analysis_year]
            if not fiscal_matches.empty:
                matches = fiscal_matches

    row = matches.sort_values(["fiscal_year", "eval_year"]).iloc[-1]
    payload = {key: (None if pd.isna(value) else value) for key, value in row.to_dict().items()}
    payload["__requested_company_id"] = normalized_company_id
    return payload


def _resolve_feature_row_for_selection(
    selection: CompanySelectionRequest,
) -> tuple[dict[str, Any] | None, str | None]:
    master = _load_feature_master().copy()
    stock_code = selection.company.stock_code
    stock_code_no_zero = stock_code.lstrip("0") or "0"
    stock_codes = master["stock_code"].astype(str)
    stock_codes_no_zero = stock_codes.str.lstrip("0").replace("", "0")

    matches = master.loc[
        (master["market"].astype(str).str.upper() == selection.company.market)
        & ((stock_codes == stock_code) | (stock_codes_no_zero == stock_code_no_zero))
    ].copy()
    if matches.empty:
        return None, "snapshot_not_found"

    fiscal_year = selection.analysis.fiscal_year
    eval_year = selection.analysis.eval_year
    if fiscal_year is not None:
        matches = matches.loc[matches["fiscal_year"] == fiscal_year]
    elif eval_year is not None:
        matches = matches.loc[matches["eval_year"] == eval_year]
    else:
        matches = matches.loc[matches["eval_year"] <= selection.as_of_date.year]
        if not matches.empty:
            matches = matches.sort_values(["fiscal_year", "eval_year"]).tail(1)

    if matches.empty:
        return None, "snapshot_not_found"
    if len(matches) > 1:
        return None, "ambiguous_snapshot"

    row = matches.iloc[0]
    if int(row["eval_year"]) > selection.as_of_date.year:
        return None, "as_of_date_violation"
    return {key: (None if pd.isna(value) else value) for key, value in row.to_dict().items()}, None


def _dataset_backed_payload(
    dataset_row: dict[str, Any],
    *,
    company_selection: CompanySelectionRequest | None = None,
) -> dict[str, Any]:
    dataset_row = dict(dataset_row)
    company_name = str(dataset_row.get("corp_name") or dataset_row.get("stock_code") or "unknown")
    market = str(dataset_row.get("market") or "UNKNOWN")
    requested_company_id = str(dataset_row.get("__requested_company_id") or "").strip()
    stock_code = _normalize_stock_code(dataset_row.get("stock_code") or "unknown")
    normalized_stock_code = (
        company_selection.company.stock_code
        if company_selection is not None
        else _normalize_stock_code(
            requested_company_id if _looks_like_stock_code(requested_company_id) else stock_code
        )
    )
    fiscal_year = int(dataset_row.get("fiscal_year") or 0)
    analysis_year = int(dataset_row.get("eval_year") or fiscal_year)
    size_group = str(dataset_row.get("firm_size_group") or "unknown")
    industry = str(dataset_row.get("industry_macro_category") or "unknown")
    source_path = str(dataset_row.get("__source_path") or _FEATURE_MASTER_PATH)
    company_id = (
        _build_snapshot_company_id(
            market=market,
            stock_code=normalized_stock_code,
            fiscal_year=fiscal_year,
        )
        if company_selection is not None
        else normalized_stock_code
    )
    dataset_row["stock_code"] = normalized_stock_code
    dataset_row["company_id"] = company_id
    dataset_row["company_name"] = company_name
    dataset_row.pop("__requested_company_id", None)
    # 대시보드에서 미리 계산한 peer percentile 결과를 같이 실어 두면,
    # Stage 2 에이전트가 산업/시장 비교 문장을 별도 재계산 없이 바로 만들 수 있다.
    peer_rows = _resolve_peer_rows(stock_code=normalized_stock_code, fiscal_year=fiscal_year)

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

    processed_company: dict[str, Any] = {
        "company_id": company_id,
        "company_name": company_name,
        "market": market,
        "stock_code": normalized_stock_code,
        "analysis_year": analysis_year,
        "fiscal_year": fiscal_year,
        "eval_year": analysis_year,
        "source": company_selection.source if company_selection else source_path,
        "source_ref": source_path,
    }
    if company_selection is not None:
        processed_company["corp_code"] = company_selection.company.corp_code
        processed_company["request_id"] = company_selection.request_id
        processed_company["as_of_date"] = company_selection.as_of_date.isoformat()

    payload: dict[str, Any] = {
        "company_id": company_id,
        "company_name": company_name,
        "market": market,
        "analysis_year": analysis_year,
        "company_profile": {
            "company": {
                "id": company_id,
                "name": company_name,
                "market": market,
                "summary": summary,
            },
            "financials": {},
            "qualitative": {},
            "market_context": {},
        },
        "processed_company": processed_company,
        "processed_company_list_ref": source_path,
        "raw_financials": {},
        # source_feature_row는 Stage 1이 바로 모델 입력 벡터를 만들 때 쓰는 원본 row다.
        # peer_comparison_rows는 Stage 2 QuantCreditAgent가 산업/시장 비교 문장을 만들 때 쓴다.
        "source_feature_row": dataset_row,
        "peer_comparison_rows": peer_rows,
        "insufficient_data": False,
        "audit": [audit],
    }
    if company_selection is not None:
        payload["company_selection"] = company_selection.model_dump(mode="json", exclude_none=True)
    return payload


def _build_snapshot_company_id(*, market: str, stock_code: str, fiscal_year: int) -> str:
    return f"{market.upper()}-{stock_code}-{fiscal_year}"


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
