"""Dashboard settings, labels, and presentation constants."""

from __future__ import annotations

from typing import Final

from cas.dashboard.data_loader import DEFAULT_ARTIFACT_DIR

DASHBOARD_BASE_STAGE2_RUNNER: Final = "deterministic"
DASHBOARD_LIVE_STAGE2_RUNNER: Final = "agno"
DASHBOARD_COMMITTEE_CONTEXT_CACHE_VERSION: Final = "dashboard_committee_context_v3"

PREFERRED_DEFAULT_COMPANIES: Final[list[str]] = [
    "현대모비스(주)",
    "삼성물산(주)",
    "삼성SDI(주)",
    "(주)카카오",
    "에스케이이노베이션(주)",
]

COLOR_RISK: Final = "#c85050"
COLOR_MITIGATE: Final = "#2f9e5b"
COLOR_NEUTRAL: Final = "#4f6fad"
COLOR_MUTED: Final = "#9aa3b2"
COLOR_SOFT_BLUE: Final = "#7f93c9"
COLOR_DARK: Final = COLOR_MUTED
COLOR_COMPANY: Final = "#1d4ed8"
COLOR_INDUSTRY: Final = "#d97706"
COLOR_MARKET: Final = "#6b7280"

ARTIFACT_PRESETS: Final[dict[str, dict[str, object]]] = {
    "team_43": {
        "label": "2026 예측 결과",
        "path": DEFAULT_ARTIFACT_DIR,
        "description": "2025 회계연도 입력값으로 2026년 신용도를 예측한 기본 결과를 불러옵니다.",
    },
    "custom": {
        "label": "직접 경로 입력",
        "path": None,
        "description": "사용자가 직접 대시보드 아티팩트 폴더 경로를 입력합니다.",
    },
}

LLM_OUTPUT_FORMATS: Final[dict[str, str]] = {
    "brief": "빠르게 보기",
    "memo": "기본으로 보기",
    "detailed": "꼼꼼히 보기",
}

MONEY_DISPLAY_MODES: Final[dict[str, str]] = {
    "detailed": "상세 (억·만·원)",
    "eok_only": "단순 (억 원)",
}

FEATURE_DIRECTION_LABELS: Final[dict[str, str]] = {
    "accruals_ratio": "낮을수록 대체로 긍정",
    "depreciation": "맥락에 따라 다름",
    "intangible_assets_ratio": "맥락에 따라 다름",
    "ocf_to_total_liabilities": "높을수록 대체로 긍정",
    "total_debt_turnover": "높을수록 대체로 긍정",
    "firm_size_group": "맥락에 따라 다름",
    "industry_macro_category": "맥락에 따라 다름",
    "listed_year": "맥락에 따라 다름",
    "market": "맥락에 따라 다름",
    "spec_spread": "낮을수록 대체로 긍정",
    "dividend_payer": "O가 대체로 긍정",
    "market_to_book": "맥락에 따라 다름",
    "gross_profit": "높을수록 대체로 긍정",
    "interest_coverage_ratio": "높을수록 대체로 긍정",
    "net_margin": "높을수록 대체로 긍정",
    "operating_roa": "높을수록 대체로 긍정",
    "pretax_roa": "높을수록 대체로 긍정",
    "pretax_roe": "높을수록 대체로 긍정",
    "assets_total": "맥락에 따라 다름",
    "capital_impairment_ratio": "낮을수록 대체로 긍정",
    "cash_ratio": "높을수록 대체로 긍정",
    "current_ratio": "높을수록 대체로 긍정",
    "debt_ratio": "낮을수록 대체로 긍정",
    "equity_ratio": "높을수록 대체로 긍정",
    "total_borrowings_ratio": "낮을수록 대체로 긍정",
    "is_2y_consecutive_ocf_deficit": "아니오가 대체로 긍정",
    "net_margin_diff": "높을수록 대체로 긍정",
    "short_term_borrowings_share": "낮을수록 대체로 긍정",
    "total_assets_growth": "맥락에 따라 다름",
}

FINANCIAL_STATEMENT_DERIVED_FEATURES: Final[set[str]] = {
    "accruals_ratio",
    "assets_total",
    "capital_impairment_ratio",
    "cash_ratio",
    "cashflow_coverage_ratio",
    "current_ratio",
    "debt_ratio",
    "depreciation",
    "equity_ratio",
    "gross_profit",
    "icr_under_1",
    "interest_coverage_ratio",
    "intangible_assets_ratio",
    "is_2y_consecutive_ocf_deficit",
    "is_2y_consecutive_operating_loss",
    "net_margin",
    "net_margin_diff",
    "ocf_to_sales",
    "ocf_to_total_borrowings",
    "ocf_to_total_liabilities",
    "operating_roa",
    "pretax_roa",
    "pretax_roe",
    "short_term_borrowings_share",
    "total_assets_growth",
    "total_borrowings_ratio",
    "total_debt_turnover",
}
