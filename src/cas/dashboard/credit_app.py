"""Streamlit dashboard for 43-feature credit risk model exploration."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from datetime import date
from html import escape
from pathlib import Path
from typing import cast

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from cas.agents.contracts import build_company_selection_from_row
from cas.agents.nodes import committee_node, rule_engine_node
from cas.agents.state import AgentState
from cas.dashboard.data_loader import (
    DEFAULT_ARTIFACT_DIR,
    DashboardArtifacts,
    load_dashboard_artifacts,
)
from cas.dashboard.evidence_panel import (
    EvidencePanelColors,
    EvidencePanelRenderers,
    dashboard_veto_status_label,
    external_veto_candidate_count,
    render_external_evidence_items,
    render_external_evidence_judgment,
)
from cas.dashboard.formatting import format_ratio_value
from cas.dashboard.llm import generate_llm_explanation
from cas.dashboard.streamlit_compat import (
    stretch_altair_chart,
    stretch_dataframe,
    stretch_download_button,
)
from cas.evidence import collect_external_evidence

LOGGER = logging.getLogger(__name__)

MARKET_LABELS = {
    "KOSPI": "코스피",
    "KOSDAQ": "코스닥",
}

SIZE_LABELS = {
    "large": "대기업",
    "mid_sized": "중견기업",
    "small_and_medium": "중소기업",
    "other": "기타",
}

INDUSTRY_LABELS = {
    "construction": "건설업",
    "it_services": "IT·서비스업",
    "manufacturing": "제조업",
    "other": "기타",
    "transport_storage": "운수·창고업",
    "wholesale_retail": "도소매업",
}

PREDICTION_LABELS = {
    0: "투자적격",
    1: "투기등급",
}

STAGE2_RISK_BAND_LABELS = {
    "stable": "안정",
    "watch": "관찰",
    "high_risk": "고위험",
    "insufficient_data": "데이터 부족",
}

STAGE2_AGENT_ROLE_LABELS = {
    "quant_credit": "QuantCreditAgent",
    "evidence_audit": "EvidenceAuditAgent",
    "chair_report": "ChairReportAgent",
}

PREFERRED_DEFAULT_COMPANIES = [
    "현대모비스(주)",
    "삼성물산(주)",
    "삼성SDI(주)",
    "(주)카카오",
    "에스케이이노베이션(주)",
]

COLOR_RISK = "#c85050"
COLOR_MITIGATE = "#2f9e5b"
COLOR_NEUTRAL = "#4f6fad"
COLOR_MUTED = "#9aa3b2"
COLOR_SOFT_BLUE = "#7f93c9"
COLOR_DARK = COLOR_MUTED
COLOR_COMPANY = "#1d4ed8"
COLOR_INDUSTRY = "#d97706"
COLOR_MARKET = "#6b7280"
COLOR_CARD_BG = "var(--cas-panel)"
COLOR_CARD_BORDER = "var(--cas-border)"
COLOR_CARD_LABEL = "var(--cas-muted)"
COLOR_CARD_VALUE = "var(--cas-text)"
CARD_SHADOW = "var(--cas-shadow)"

ARTIFACT_PRESETS = {
    "team_43": {
        "label": "기본 결과",
        "path": DEFAULT_ARTIFACT_DIR,
        "description": "현재 연결된 기본 대시보드 결과를 불러옵니다.",
    },
    "custom": {
        "label": "직접 경로 입력",
        "path": None,
        "description": "사용자가 직접 대시보드 아티팩트 폴더 경로를 입력합니다.",
    },
}

LLM_PROVIDER_LABELS = {
    "openai": "OpenAI",
    "claude": "Claude",
}

RECOMMENDED_LLM_MODELS = {
    "openai": [
        ("gpt-5.5", "gpt-5.5 | 최고급 추론·요약"),
        ("gpt-5.4-mini", "gpt-5.4-mini | 속도·비용 균형"),
        ("gpt-4.1", "gpt-4.1 | 안정적인 고성능"),
        ("gpt-4.1-mini", "gpt-4.1-mini | 빠른 기본 옵션"),
    ],
    "claude": [
        ("claude-sonnet-4-20250514", "claude-sonnet-4-20250514 | 균형형"),
        ("claude-opus-4-20250514", "claude-opus-4-20250514 | 고급 추론"),
        ("claude-3-7-sonnet-20250219", "claude-3-7-sonnet-20250219 | 안정형"),
    ],
}

LLM_OUTPUT_FORMATS = {
    "brief": "간단 요약",
    "memo": "기본 심사 메모",
    "detailed": "상세 보고서형",
}

OUTPUT_FORMAT_DESCRIPTIONS = {
    "brief": "핵심 판단과 주요 근거만 빠르게 읽는 형식입니다.",
    "memo": "판단 차이, 위험/완화 요인, 근거 요약을 균형 있게 보여주는 기본 형식입니다.",
    "detailed": "에이전트별 검토, 규칙 기반 판단 근거, 1차 모델 세부 항목까지 함께 확인하는 상세 형식입니다.",
}

MONEY_DISPLAY_MODES = {
    "detailed": "상세 (억·만·원)",
    "eok_only": "단순 (억 원)",
}

FEATURE_DIRECTION_LABELS = {
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


def _load_dashboard_artifacts_cached(artifact_dir: str | None = None) -> DashboardArtifacts:
    """Cache dashboard artifact loading for Streamlit."""
    path = Path(artifact_dir) if artifact_dir else None
    return load_dashboard_artifacts(path)


cached_load_dashboard_artifacts: Callable[[str | None], DashboardArtifacts] = st.cache_data(
    show_spinner=False
)(_load_dashboard_artifacts_cached)


def inject_dashboard_theme() -> None:
    """Apply dashboard styling without forcing a fixed light theme.

    Streamlit's Settings menu changes the app theme on the client side.  CSS
    inserted through ``st.markdown`` cannot reliably read that setting in every
    Streamlit release, so this stylesheet deliberately avoids hard-coded page
    backgrounds.  Custom CAS cards are rendered as subtle currentColor-based
    translucent surfaces, which makes them follow both Streamlit light and dark
    themes automatically.
    """
    st.markdown(
        """
        <style>
        :root,
        .stApp {
          color-scheme: light dark;
          --cas-blue: var(--st-primary-color, var(--primary-color, #1d4ed8));
          --cas-risk: #c85050;
          --cas-risk-text: #d14a4a;
          --cas-success: #2f9e5b;
          --cas-warning: #b7791f;
          --cas-neutral: #4f6fad;
          --cas-text: inherit;
          --cas-muted: currentColor;
          --cas-panel: rgba(128, 128, 128, 0.08);
          --cas-panel-strong: rgba(128, 128, 128, 0.12);
          --cas-border: rgba(128, 128, 128, 0.28);
          --cas-border-soft: rgba(128, 128, 128, 0.18);
          --cas-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
          --cas-risk-soft: rgba(200, 80, 80, 0.14);
          --cas-risk-border: rgba(200, 80, 80, 0.38);
          --cas-success-soft: rgba(47, 158, 91, 0.14);
          --cas-success-border: rgba(47, 158, 91, 0.38);
          --cas-warning-soft: rgba(183, 121, 31, 0.14);
          --cas-warning-border: rgba(183, 121, 31, 0.38);
          --cas-neutral-soft: rgba(128, 128, 128, 0.10);
          --cas-neutral-border: rgba(128, 128, 128, 0.28);
        }

        @supports (color: color-mix(in srgb, white, black)) {
          :root,
          .stApp {
            --cas-muted: color-mix(in srgb, currentColor 64%, transparent);
            --cas-panel: color-mix(in srgb, currentColor 5%, transparent);
            --cas-panel-strong: color-mix(in srgb, currentColor 8%, transparent);
            --cas-border: color-mix(in srgb, currentColor 18%, transparent);
            --cas-border-soft: color-mix(in srgb, currentColor 10%, transparent);
            --cas-shadow: 0 1px 2px color-mix(in srgb, currentColor 13%, transparent);
            --cas-risk-soft: color-mix(in srgb, var(--cas-risk) 17%, transparent);
            --cas-risk-border: color-mix(in srgb, var(--cas-risk) 42%, transparent);
            --cas-success-soft: color-mix(in srgb, var(--cas-success) 17%, transparent);
            --cas-success-border: color-mix(in srgb, var(--cas-success) 42%, transparent);
            --cas-warning-soft: color-mix(in srgb, var(--cas-warning) 17%, transparent);
            --cas-warning-border: color-mix(in srgb, var(--cas-warning) 42%, transparent);
            --cas-neutral-soft: color-mix(in srgb, currentColor 7%, transparent);
            --cas-neutral-border: color-mix(in srgb, currentColor 18%, transparent);
          }
        }

        .stApp,
        div[data-testid="stAppViewContainer"],
        div[data-testid="stMain"],
        div[data-testid="stMainBlockContainer"],
        .main,
        .main .block-container {
          color: inherit !important;
        }

        .main .block-container {
          max-width: 1480px;
          padding-top: 1.25rem;
          padding-bottom: 2rem;
        }

        .main .block-container, .main .block-container * {
          letter-spacing: 0 !important;
        }

        h1 {
          color: inherit;
          font-size: 1.72rem !important;
          line-height: 1.25 !important;
          margin: 0 0 0.25rem 0 !important;
        }

        h2, h3 {
          color: inherit;
          line-height: 1.35 !important;
        }

        h2 {
          border-top: 1px solid var(--cas-border-soft);
          font-size: 1.12rem !important;
          margin-top: 1.1rem !important;
          padding-top: 0.8rem !important;
        }

        h3 {
          font-size: 1rem !important;
        }

        div[data-testid="stCaptionContainer"] {
          color: var(--cas-muted);
          font-size: 0.9rem;
          line-height: 1.58;
          max-width: 1120px;
        }

        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {
          line-height: 1.62;
        }

        section[data-testid="stSidebar"] {
          border-right: 1px solid var(--cas-border);
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
          color: var(--cas-muted);
          font-size: 0.88rem;
          line-height: 1.5;
        }

        div[data-testid="stTabs"] [role="tablist"] {
          align-items: center;
          background: transparent;
          border-bottom: 1px solid var(--cas-border);
          gap: 0.25rem;
          padding: 0.35rem 0 0.45rem 0;
          position: sticky;
          top: 0;
          z-index: 10;
        }

        button[role="tab"] {
          border-radius: 8px 8px 0 0 !important;
          color: var(--cas-muted) !important;
          font-size: 0.92rem !important;
          font-weight: 700 !important;
          min-height: 2.35rem;
          padding: 0.45rem 0.75rem !important;
        }

        button[role="tab"][aria-selected="true"] {
          background: var(--cas-panel-strong) !important;
          box-shadow: inset 0 -2px 0 var(--cas-blue);
          color: var(--cas-blue) !important;
        }

        div[data-testid="stHorizontalBlock"] {
          gap: 0.8rem;
        }

        div[data-testid="stExpander"] {
          background: var(--cas-panel);
          border: 1px solid var(--cas-border) !important;
          border-radius: 8px !important;
          box-shadow: var(--cas-shadow);
        }

        div[data-testid="stDataFrame"] {
          border: 1px solid var(--cas-border);
          border-radius: 8px;
          overflow: hidden;
        }

        div.stButton > button,
        div.stDownloadButton > button {
          border-radius: 8px !important;
          font-weight: 700;
          min-height: 2.6rem;
          width: 100%;
        }

        div[data-testid="stAlert"] {
          border-radius: 8px;
        }

        .market-search-panel {
          background: var(--cas-panel);
          border: 1px solid var(--cas-border);
          border-radius: 8px;
          box-shadow: var(--cas-shadow);
          margin: 0.9rem 0 1rem 0;
          padding: 1rem;
        }

        .market-search-panel h2 {
          border-top: 0;
          font-size: 1.08rem !important;
          margin: 0 0 0.25rem 0 !important;
          padding-top: 0 !important;
        }

        .market-search-panel p {
          color: var(--cas-muted);
          font-size: 0.9rem;
          line-height: 1.55;
          margin: 0;
        }

        .market-card {
          background: var(--cas-panel);
          border: 1px solid var(--cas-border);
          border-left: 5px solid var(--cas-risk);
          border-radius: 8px;
          box-shadow: var(--cas-shadow);
          min-height: 148px;
          padding: 0.9rem 1rem;
        }

        .market-card-rank {
          color: var(--cas-risk);
          font-size: 0.78rem;
          font-weight: 800;
          margin-bottom: 0.3rem;
          text-transform: uppercase;
        }

        .market-card-title {
          color: inherit;
          font-size: 1.02rem;
          font-weight: 800;
          line-height: 1.35;
          margin-bottom: 0.45rem;
          word-break: keep-all;
        }

        .market-card-meta {
          color: var(--cas-muted);
          display: flex;
          flex-wrap: wrap;
          font-size: 0.86rem;
          gap: 0.35rem;
          line-height: 1.45;
        }

        .market-card-risk {
          color: var(--cas-risk-text);
          font-size: 1.22rem;
          font-weight: 800;
          margin-top: 0.55rem;
        }

        .market-section-title {
          color: inherit;
          font-size: 1rem;
          font-weight: 800;
          margin: 0.2rem 0 0.45rem 0;
        }

        .selected-company-bar {
          background: var(--cas-panel);
          border: 1px solid var(--cas-border);
          border-radius: 8px;
          box-shadow: var(--cas-shadow);
          margin: 0 0 0.8rem 0;
          padding: 0.72rem 0.9rem;
        }

        .selected-company-title {
          color: inherit;
          font-size: 1rem;
          font-weight: 800;
          line-height: 1.35;
        }

        .selected-company-meta {
          color: var(--cas-muted);
          font-size: 0.86rem;
          line-height: 1.45;
          margin-top: 0.16rem;
        }

        .committee-decision-strip {
          background: var(--cas-panel);
          border: 1px solid var(--cas-border);
          border-left: 6px solid var(--cas-blue);
          border-radius: 8px;
          box-shadow: var(--cas-shadow);
          margin: 0.4rem 0 0.75rem 0;
          padding: 1rem;
        }

        .committee-decision-topline {
          align-items: center;
          display: flex;
          flex-wrap: wrap;
          gap: 0.65rem;
          margin-bottom: 0.55rem;
        }

        .committee-decision-label {
          color: var(--cas-muted);
          font-size: 0.88rem;
          font-weight: 800;
        }

        .committee-decision-summary {
          color: inherit;
          font-size: 1rem;
          font-weight: 700;
          line-height: 1.62;
          margin: 0;
          word-break: keep-all;
        }

        .committee-highlight-grid {
          display: grid;
          gap: 0.75rem;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          margin: 0.2rem 0 0.95rem 0;
        }

        .committee-highlight-card {
          background: var(--cas-panel);
          border: 1px solid var(--cas-border);
          border-top: 4px solid var(--cas-blue);
          border-radius: 8px;
          box-shadow: var(--cas-shadow);
          min-height: 132px;
          padding: 0.9rem 1rem;
        }

        .committee-highlight-card.risk {
          border-top-color: var(--cas-risk);
        }

        .committee-highlight-card.mitigate {
          border-top-color: var(--cas-success);
        }

        .committee-highlight-card.warning {
          border-top-color: var(--cas-warning);
        }

        .committee-highlight-title {
          color: var(--cas-muted);
          font-size: 0.9rem;
          font-weight: 800;
          margin-bottom: 0.45rem;
        }

        .committee-highlight-body {
          color: inherit;
          font-size: 0.96rem;
          font-weight: 700;
          line-height: 1.6;
          word-break: keep-all;
        }

        .committee-highlight-body ul {
          margin: 0;
          padding-left: 1.05rem;
        }

        .committee-highlight-body li {
          margin-bottom: 0.35rem;
        }

        .committee-detail-flow {
          background: var(--cas-panel);
          border: 1px solid var(--cas-border);
          border-radius: 8px;
          box-shadow: var(--cas-shadow);
          margin: 0.25rem 0 1rem 0;
          padding: 1rem 1.05rem;
        }

        .committee-detail-title {
          color: inherit;
          font-size: 1rem;
          font-weight: 800;
          margin-bottom: 0.65rem;
        }

        .committee-detail-section {
          border-top: 1px solid var(--cas-border);
          padding: 0.75rem 0 0.1rem 0;
        }

        .committee-detail-section:first-of-type {
          border-top: 0;
          padding-top: 0;
        }

        .committee-detail-heading {
          color: var(--cas-muted);
          font-size: 0.9rem;
          font-weight: 800;
          margin-bottom: 0.28rem;
        }

        .committee-detail-text,
        .committee-detail-section li {
          color: inherit;
          font-size: 0.97rem;
          line-height: 1.68;
          word-break: keep-all;
        }

        .committee-detail-section ul {
          margin: 0.1rem 0 0 1.1rem;
          padding: 0;
        }

        hr {
          margin: 1rem 0;
        }

        @media (max-width: 900px) {
          .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
          }

          div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
          }

          div[data-testid="stTabs"] [role="tablist"] {
            overflow-x: auto;
            white-space: nowrap;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def to_market_label(value: object) -> str:
    """Convert a market code into a Korean label."""
    return MARKET_LABELS.get(str(value), str(value))


def to_market_display_label(value: object) -> str:
    """Convert a market code into a readable label for the market selector."""
    labels = {"KOSPI": "코스피", "KOSDAQ": "코스닥"}
    return labels.get(str(value), to_market_label(value))


def to_size_label(value: object) -> str:
    """Convert a firm size code into a Korean label."""
    return SIZE_LABELS.get(str(value), str(value))


def to_industry_label(value: object) -> str:
    """Convert an industry code into a Korean label."""
    return INDUSTRY_LABELS.get(str(value), str(value))


def to_industry_display_label(value: object) -> str:
    """Convert an industry code into a readable label for the market selector."""
    labels = {
        "construction": "건설",
        "it_services": "IT/서비스",
        "manufacturing": "제조",
        "other": "기타",
        "transport_storage": "운수/창고",
        "wholesale_retail": "도소매",
    }
    return labels.get(str(value), to_industry_label(value))


def to_prediction_label(value: object) -> str:
    """Convert a numeric prediction label into a Korean label."""
    try:
        return PREDICTION_LABELS.get(int(float(str(value))), str(value))
    except (TypeError, ValueError):
        return str(value)


def to_stage2_model_label(value: object) -> str:
    """Convert dashboard prediction labels into the Stage 2 model_view label space."""
    label = to_prediction_label(value)
    if label in {"투기등급", "부적격"}:
        return "부적격"
    if label in {"투자적격", "적격"}:
        return "투자적격"
    return label


def to_committee_base_label(model_label: object) -> str:
    """Map a binary Stage 1 model label onto the committee label space."""
    label = str(model_label)
    if label == "투자적격":
        return "적격"
    if label in {"투기등급", "부적격"}:
        return "부적격"
    return "보류"


def pick_selected_company(artifacts: DashboardArtifacts) -> pd.Series:
    """Render sidebar selectors and return the chosen company snapshot."""
    return pick_selected_company_from_market_explorer(artifacts)

    latest = artifacts.company_latest.copy()
    markets = ["전체", *sorted(latest["market"].dropna().unique().tolist())]
    selected_market = st.sidebar.selectbox(
        "시장",
        markets,
        format_func=lambda value: "전체" if value == "전체" else to_market_label(value),
    )
    if selected_market != "전체":
        latest = latest.loc[latest["market"] == selected_market]

    industries = ["전체", *sorted(latest["industry_macro_category"].dropna().unique().tolist())]
    selected_industry = st.sidebar.selectbox(
        "산업",
        industries,
        format_func=lambda value: "전체" if value == "전체" else to_industry_label(value),
    )
    if selected_industry != "전체":
        latest = latest.loc[latest["industry_macro_category"] == selected_industry]

    search_query = st.sidebar.text_input(
        "기업 검색",
        value="",
        placeholder="기업명 또는 종목코드 입력",
        help="기업명이나 종목코드 일부를 입력하면 선택 목록을 좁힐 수 있습니다.",
    ).strip()
    if search_query:
        stock_code_query = search_query.zfill(6) if search_query.isdigit() else search_query
        mask = latest["corp_name"].astype(str).str.contains(
            search_query, case=False, na=False
        ) | latest["stock_code"].map(_stock_code_text).str.contains(
            stock_code_query, case=False, na=False
        )
        latest = latest.loc[mask]

    if latest.empty:
        st.sidebar.warning("검색 조건에 맞는 기업이 없습니다. 검색어 또는 필터를 조정해 주세요.")
        st.stop()

    options = latest.assign(
        label=lambda frame: (
            frame["corp_name"]
            + " | "
            + frame["stock_code"].map(_stock_code_text)
            + " | FY"
            + frame["fiscal_year"].astype(int).astype(str)
        )
    )
    labels = options["label"].tolist()
    default_index = 0
    for preferred_name in PREFERRED_DEFAULT_COMPANIES:
        matched = options.index[options["corp_name"].astype(str) == preferred_name].tolist()
        if matched:
            default_index = int(options.index.get_loc(matched[0]))
            break
    selected_label: str = st.sidebar.selectbox("기업 선택", labels, index=default_index)
    return options.loc[options["label"] == selected_label].iloc[0]


def pick_selected_company_from_market_explorer(artifacts: DashboardArtifacts) -> pd.Series:
    """Render the market-style company selector and return the chosen company snapshot."""
    explorer_frame = build_company_explorer_frame(artifacts)
    if explorer_frame.empty:
        st.warning("분석 가능한 종목 데이터가 없습니다. 대시보드 산출물을 다시 확인해주세요.")
        st.stop()

    current_key = str(st.session_state.get("selected_company_key", ""))
    if current_key:
        matched = explorer_frame.loc[explorer_frame["_company_key"] == current_key]
        if not matched.empty:
            selected_row = matched.iloc[0]
            render_selected_company_detail_header(selected_row)
            return selected_row
        st.session_state.pop("selected_company_key", None)

    selected_key = render_company_market_explorer(explorer_frame)
    if not selected_key:
        st.info("상단 검색창이나 시장별 목록에서 종목을 선택하면 상세 분석 화면이 열립니다.")
        st.stop()

    matched = explorer_frame.loc[explorer_frame["_company_key"] == selected_key]
    if matched.empty:
        st.session_state.pop("selected_company_key", None)
        st.warning("선택한 종목을 현재 산출물에서 찾을 수 없습니다. 다시 선택해주세요.")
        st.stop()
    return matched.iloc[0]


def render_selected_company_detail_header(selected_row: pd.Series) -> None:
    """Render detail-page navigation once a company has been selected."""
    nav_col, title_col = st.columns([0.16, 0.84])
    with nav_col:
        if st.button("← 종목 목록", use_container_width=True):
            st.session_state.pop("selected_company_key", None)
            st.rerun()

    market = selected_row.get("_display_market") or to_market_display_label(
        selected_row.get("market")
    )
    industry = selected_row.get("_display_industry") or to_industry_display_label(
        selected_row.get("industry_macro_category")
    )
    probability = selected_row.get("_display_probability") or format_percent(
        selected_row.get("prob_speculative")
    )
    stock_code = _stock_code_text(selected_row.get("stock_code"))
    fiscal_year = format_scalar(selected_row.get("fiscal_year"))
    risk_band = selected_row.get("risk_band") or "-"

    with title_col:
        st.markdown(
            (
                "<div class='selected-company-bar'>"
                f"<div class='selected-company-title'>{escape(str(selected_row.get('corp_name') or '-'))}</div>"
                "<div class='selected-company-meta'>"
                f"{escape(str(market))} · {escape(stock_code)} · FY{escape(fiscal_year)} · "
                f"{escape(str(industry))} · 부적합 가능성 {escape(str(probability))} · 위험 구간 {escape(str(risk_band))}"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def build_company_explorer_frame(artifacts: DashboardArtifacts) -> pd.DataFrame:
    """Build the selectable market overview frame for the dashboard landing area."""
    latest = artifacts.company_latest.copy()
    latest["_stock_code_text"] = latest["stock_code"].map(_stock_code_text)
    latest["_company_key"] = latest.apply(_company_selection_key, axis=1)

    if artifacts.prediction_scores is not None:
        prediction_frame = artifacts.prediction_scores.copy()
        prediction_frame["_stock_code_text"] = prediction_frame["stock_code"].map(_stock_code_text)
        prediction_columns = [
            "_stock_code_text",
            "fiscal_year",
            "prob_speculative",
            "predicted_label",
            "risk_band",
            "stage2_review_priority",
            "trigger_reason",
        ]
        available_columns = [
            column for column in prediction_columns if column in prediction_frame.columns
        ]
        latest = latest.merge(
            prediction_frame.loc[:, available_columns],
            on=["_stock_code_text", "fiscal_year"],
            how="left",
        )

    latest["_prob_speculative_number"] = pd.to_numeric(
        latest.get("prob_speculative"),
        errors="coerce",
    )
    latest["_display_market"] = latest["market"].map(to_market_display_label)
    latest["_display_industry"] = latest["industry_macro_category"].map(to_industry_display_label)
    latest["_display_size"] = latest["firm_size_group"].map(to_size_label)
    latest["_display_probability"] = latest["prob_speculative"].map(format_percent)
    latest["_display_label"] = latest["predicted_label"].map(to_prediction_label)
    latest["_search_label"] = latest.apply(_company_search_label, axis=1)
    return latest.sort_values(["market", "corp_name", "fiscal_year"]).reset_index(drop=True)


def _company_selection_key(row: pd.Series) -> str:
    """Build a stable selection key for one company-year row."""
    fiscal_year = row.get("fiscal_year")
    try:
        fiscal_year_text = str(int(float(str(fiscal_year))))
    except (TypeError, ValueError):
        fiscal_year_text = str(fiscal_year)
    return f"{row.get('market')}-{_stock_code_text(row.get('stock_code'))}-{fiscal_year_text}"


def _company_search_label(row: pd.Series) -> str:
    """Return the searchable option label shown in the top selectbox."""
    probability = row.get("_display_probability") or "-"
    risk_band = row.get("risk_band") or "-"
    return (
        f"{row.get('corp_name')} · {_stock_code_text(row.get('stock_code'))} · "
        f"{row.get('_display_market')} · FY{format_scalar(row.get('fiscal_year'))} · "
        f"부적합 가능성 {probability} · {risk_band}"
    )


def render_company_market_explorer(explorer_frame: pd.DataFrame) -> str | None:
    """Render search, top unsuitable highlights, and KOSPI/KOSDAQ company lists."""
    current_key = str(st.session_state.get("selected_company_key", ""))
    valid_keys = explorer_frame["_company_key"].astype(str).tolist()
    if current_key and current_key not in valid_keys:
        current_key = ""
        st.session_state.pop("selected_company_key", None)

    st.markdown(
        """
        <div class="market-search-panel">
          <h2>종목 검색</h2>
          <p>이 프로젝트 산출물에 포함된 종목만 검색됩니다. 회사명, 종목코드, 시장을 입력해 분석할 기업을 고르세요.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    option_labels = dict(
        zip(
            explorer_frame["_company_key"].astype(str),
            explorer_frame["_search_label"].astype(str),
            strict=False,
        )
    )
    selected_index = valid_keys.index(current_key) if current_key in valid_keys else None
    selected_from_search = st.selectbox(
        "회사명 또는 종목코드",
        options=valid_keys,
        index=selected_index,
        format_func=lambda value: option_labels.get(str(value), str(value)),
        placeholder="예: 삼성전자, 005930, KOSDAQ",
        label_visibility="collapsed",
    )
    if selected_from_search and selected_from_search != current_key:
        st.session_state["selected_company_key"] = str(selected_from_search)
        st.rerun()

    render_top_unsuitable_companies(explorer_frame)
    render_market_company_lists(explorer_frame)
    return str(st.session_state.get("selected_company_key", ""))


def render_top_unsuitable_companies(explorer_frame: pd.DataFrame) -> None:
    """Render the three highest-risk companies as prominent cards."""
    ranked = (
        explorer_frame.dropna(subset=["_prob_speculative_number"])
        .sort_values("_prob_speculative_number", ascending=False)
        .head(3)
    )
    if ranked.empty:
        return

    st.markdown("### 부적합 가능성 상위 3개")
    columns = st.columns(3)
    for index, row in enumerate(ranked.to_dict(orient="records"), start=1):
        container = columns[index - 1]
        render_unsuitable_company_card(container, row, rank=index)
        if container.button(
            "분석 보기",
            key=f"top_unsuitable_{row['_company_key']}",
            use_container_width=True,
        ):
            st.session_state["selected_company_key"] = str(row["_company_key"])
            st.rerun()


def render_unsuitable_company_card(
    container: st.delta_generator.DeltaGenerator,
    row: dict[str, object],
    *,
    rank: int,
) -> None:
    """Render one high-risk company card in the market selector."""
    container.markdown(
        (
            "<div class='market-card'>"
            f"<div class='market-card-rank'>Rank {rank}</div>"
            f"<div class='market-card-title'>{escape(str(row.get('corp_name') or '-'))}</div>"
            "<div class='market-card-meta'>"
            f"<span>{escape(str(row.get('_display_market') or '-'))}</span>"
            f"<span>·</span><span>{escape(_stock_code_text(row.get('stock_code')))}</span>"
            f"<span>·</span><span>FY{escape(format_scalar(row.get('fiscal_year')))}</span>"
            f"<span>·</span><span>{escape(str(row.get('_display_industry') or '-'))}</span>"
            "</div>"
            f"<div class='market-card-risk'>{escape(str(row.get('_display_probability') or '-'))}</div>"
            f"<div class='market-card-meta'>위험 구간 {escape(str(row.get('risk_band') or '-'))}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_market_company_lists(explorer_frame: pd.DataFrame) -> None:
    """Render separate selectable lists for KOSPI and KOSDAQ."""
    st.markdown("### 시장별 종목")
    kospi_col, kosdaq_col = st.columns(2)
    render_market_company_table(kospi_col, explorer_frame, market="KOSPI")
    render_market_company_table(kosdaq_col, explorer_frame, market="KOSDAQ")


def render_market_company_table(
    container: st.delta_generator.DeltaGenerator,
    explorer_frame: pd.DataFrame,
    *,
    market: str,
) -> None:
    """Render one market table and handle row selection."""
    market_frame = explorer_frame.loc[explorer_frame["market"].astype(str) == market].copy()
    market_frame = market_frame.sort_values(
        ["_prob_speculative_number", "corp_name"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    container.markdown(
        f"<div class='market-section-title'>{escape(to_market_display_label(market))}</div>",
        unsafe_allow_html=True,
    )
    if market_frame.empty:
        container.info(f"{to_market_display_label(market)} 종목이 없습니다.")
        return

    display_frame = market_frame.loc[
        :,
        [
            "corp_name",
            "_stock_code_text",
            "_display_probability",
            "risk_band",
            "_display_industry",
            "fiscal_year",
        ],
    ].rename(
        columns={
            "corp_name": "기업",
            "_stock_code_text": "종목코드",
            "_display_probability": "부적합 가능성",
            "risk_band": "위험 구간",
            "_display_industry": "산업",
            "fiscal_year": "회계연도",
        }
    )
    event = container.dataframe(
        display_frame,
        hide_index=True,
        height=360,
        on_select="rerun",
        selection_mode="single-row",
        use_container_width=True,
    )
    selected_rows = _selected_dataframe_rows(event)
    if selected_rows:
        selected_row_index = int(selected_rows[0])
        if 0 <= selected_row_index < len(market_frame):
            selected_key = str(market_frame.iloc[selected_row_index]["_company_key"])
            if selected_key != st.session_state.get("selected_company_key"):
                st.session_state["selected_company_key"] = selected_key
                st.rerun()


def _selected_dataframe_rows(event: object) -> list[int]:
    """Extract selected row indexes from a Streamlit dataframe selection event."""
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if selection is None:
        return []
    if isinstance(selection, dict):
        rows = selection.get("rows", [])
    else:
        rows = getattr(selection, "rows", [])
    if not isinstance(rows, list):
        return []
    return [int(row) for row in rows]


def build_company_feature_map(
    selected_row: pd.Series, feature_dictionary: pd.DataFrame
) -> pd.DataFrame:
    """Build a long-form feature value table for the selected company."""
    rows: list[dict[str, object]] = []
    for record in feature_dictionary.to_dict(orient="records"):
        feature = str(record["feature"])
        rows.append(
            {
                "feature": feature,
                "feature_group": record["feature_group"],
                "korean_name": record["korean_name"],
                "value": selected_row.get(feature),
                "unit": record["unit"],
                "description": record["description"],
            }
        )
    feature_map = pd.DataFrame(rows)
    return feature_map


def resolve_company_local_shap(
    selected_row: pd.Series,
    local_shap: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return local SHAP rows for the selected company-year if available."""
    if local_shap is None:
        return pd.DataFrame()
    stock_code = _stock_code_text(selected_row["stock_code"])
    matched = local_shap.loc[
        (local_shap["stock_code"].map(_stock_code_text) == stock_code)
        & (local_shap["fiscal_year"] == selected_row["fiscal_year"])
    ].copy()
    return matched.sort_values("abs_shap", ascending=False)


def resolve_company_peer_slice(
    selected_row: pd.Series,
    peer_percentiles: pd.DataFrame,
) -> pd.DataFrame:
    """Return peer comparison rows for the selected company-year."""
    stock_code = _stock_code_text(selected_row["stock_code"])
    return peer_percentiles.loc[
        (peer_percentiles["stock_code"].map(_stock_code_text) == stock_code)
        & (peer_percentiles["fiscal_year"] == selected_row["fiscal_year"])
    ].copy()


def resolve_industry_latest_row(
    selected_row: pd.Series,
    industry_latest_summary: pd.DataFrame | None,
) -> pd.Series | None:
    """Return the latest industry summary row for the selected company."""
    if industry_latest_summary is None:
        return None
    matched = industry_latest_summary.loc[
        (industry_latest_summary["market"] == str(selected_row["market"]))
        & (
            industry_latest_summary["industry_macro_category"]
            == str(selected_row["industry_macro_category"])
        )
    ]
    if matched.empty:
        return None
    return matched.iloc[0]


def display_name(feature: str, feature_map: pd.DataFrame) -> str:
    """Return a Korean-first display name for a feature."""
    matched = feature_map.loc[feature_map["feature"] == feature]
    if matched.empty:
        return feature
    korean_name = matched.iloc[0].get("korean_name")
    if pd.isna(korean_name) or not str(korean_name).strip():
        return feature
    return str(korean_name)


def resolve_company_prediction(
    selected_row: pd.Series,
    prediction_scores: pd.DataFrame | None,
) -> pd.Series | None:
    """Return the optional per-company prediction row if available."""
    if prediction_scores is None:
        return None
    stock_code = _stock_code_text(selected_row["stock_code"])
    matched = prediction_scores.loc[
        (prediction_scores["stock_code"].map(_stock_code_text) == stock_code)
        & (prediction_scores["fiscal_year"] == selected_row["fiscal_year"])
    ]
    if matched.empty:
        return None
    return matched.iloc[0]


def _clean_dashboard_value(value: object) -> object:
    """Convert pandas/numpy scalars into plain values for Stage 2 payloads."""
    if isinstance(value, dict):
        return {str(key): _clean_dashboard_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_clean_dashboard_value(item) for item in value]
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, str):
        try:
            return _clean_dashboard_value(value.item())
        except (AttributeError, TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    return value


def _optional_float(value: object, *, default: float = 0.0) -> float:
    """Return a safe float from dashboard artifacts."""
    cleaned = _clean_dashboard_value(value)
    if cleaned is None:
        return default
    try:
        return float(str(cleaned))
    except (TypeError, ValueError):
        return default


def _optional_int(value: object) -> int | None:
    """Return a safe int from dashboard artifacts."""
    cleaned = _clean_dashboard_value(value)
    if cleaned is None:
        return None
    try:
        return int(float(str(cleaned)))
    except (TypeError, ValueError):
        return None


def _optional_bool(value: object, *, default: bool = False) -> bool:
    """Return a safe bool from dashboard artifacts."""
    cleaned = _clean_dashboard_value(value)
    if cleaned is None:
        return default
    if isinstance(cleaned, bool):
        return cleaned
    if isinstance(cleaned, int | float):
        return bool(cleaned)
    text = str(cleaned).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _stock_code_text(value: object) -> str:
    """Normalize stock code text while preserving leading zeroes."""
    cleaned = _clean_dashboard_value(value)
    text = str(cleaned or "").strip()
    if text.endswith(".0"):
        text = text.removesuffix(".0")
    return text.zfill(6) if text.isdigit() else text


def _optional_text(value: object) -> str | None:
    """Return a clean optional text value."""
    cleaned = _clean_dashboard_value(value)
    if cleaned is None:
        return None
    text = str(cleaned).strip()
    return text or None


def _series_record(row: pd.Series) -> dict[str, object]:
    """Convert a pandas row into a plain dict for the Stage 2 state."""
    return {str(key): _clean_dashboard_value(value) for key, value in row.to_dict().items()}


def _frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a dataframe into plain records for the Stage 2 state."""
    if frame.empty:
        return []
    records: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        records.append({str(key): _clean_dashboard_value(value) for key, value in record.items()})
    return records


def _as_plain_dict(value: object) -> dict[str, object]:
    """Return a plain dict when Streamlit context values are nested mappings."""
    if isinstance(value, dict):
        return dict(value)
    return {}


def _as_text_list(value: object) -> list[str]:
    """Return a list of non-empty display strings."""
    if not isinstance(value, list | tuple):
        return []
    return [str(item) for item in value if str(item).strip()]


def _friendly_committee_text(
    value: object,
    feature_map: pd.DataFrame | None = None,
) -> str:
    """Make deterministic committee text easier to read in the dashboard."""
    text = str(value or "").strip()
    if not text:
        return ""

    replacements = {
        "모델 원판단": "1차 모델 판단",
        "위원회 라벨": "2차 위원회 판단",
        "Stage 2는": "2차 검토는",
        "정량 해석, 부채/유동성 교차 검증, 외부 근거 상태": "모델 해석, 부채·유동성 점검, 외부 근거",
        "부채 및 유동성 지표는 현재 투자적격 판단에 일부 완충 근거를 제공합니다.": (
            "부채와 유동성 지표는 현재 판단을 뒷받침하는 완충 근거로 보입니다."
        ),
        "적격로": "적격으로",
        "부적격로": "부적격으로",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    text = re.sub(
        r"(.+?)이\(가\) 현재 모델에서 위험을 높이는 방향으로 작용했습니다\.",
        r"\1 항목은 모델에서 위험을 높이는 신호로 잡혔습니다.",
        text,
    )
    text = re.sub(
        r"(.+?)이\(가\) 현재 모델에서 위험을 낮추는 방향으로 작용했습니다\.",
        r"\1 항목은 모델에서 위험을 낮추는 신호로 잡혔습니다.",
        text,
    )
    text = text.replace("산업 중앙값은", "같은 산업의 중앙값은")
    text = text.replace("시장 중앙값은", "같은 시장의 중앙값은")
    text = text.replace("입니다 시장", "이고, 시장")
    text = text.replace("입니다 같은 시장", "이고, 같은 시장")
    text = text.replace("산업 내 위치는", "산업 안에서의 위치는")
    text = text.replace("입니다 산업 안에서의 위치는", "이며, 산업 안에서의 위치는")
    text = _format_committee_feature_values(text, feature_map)
    text = _format_committee_percentile_phrases(text)
    text = text.replace("입니다..", "입니다.")
    return text


def _friendly_committee_items(
    items: list[str],
    feature_map: pd.DataFrame | None = None,
) -> list[str]:
    """Split and polish committee bullets for user-facing display."""
    friendly_items: list[str] = []
    for item in items:
        parts = [part.strip() for part in str(item).split(" / ") if part.strip()]
        for part in parts or [item]:
            friendly_text = _friendly_committee_text(part, feature_map)
            if friendly_text:
                friendly_items.append(friendly_text)
    return friendly_items


def _format_committee_feature_values(
    text: str,
    feature_map: pd.DataFrame | None,
) -> str:
    """Format numbers embedded in committee sentences using dashboard unit settings."""
    if feature_map is None or feature_map.empty:
        return text
    required_columns = {"feature", "korean_name", "unit"}
    if not required_columns.issubset(set(feature_map.columns)):
        return text

    feature_records = sorted(
        feature_map.loc[:, ["feature", "korean_name", "unit"]].to_dict(orient="records"),
        key=lambda record: len(str(record.get("korean_name") or "")),
        reverse=True,
    )
    feature_matches: list[tuple[int, dict[str, object]]] = []
    for record in feature_records:
        korean_name = str(record.get("korean_name") or "").strip()
        if not korean_name:
            continue
        for match in re.finditer(rf"{re.escape(korean_name)}\(", text):
            feature_matches.append((match.start(), record))

    if not feature_matches:
        return text

    sorted_matches = sorted(feature_matches, key=lambda item: item[0])
    formatted_parts: list[str] = []
    cursor = 0
    for index, (start, record) in enumerate(sorted_matches):
        end = sorted_matches[index + 1][0] if index + 1 < len(sorted_matches) else len(text)
        if start < cursor:
            continue
        formatted_parts.append(text[cursor:start])
        feature = str(record.get("feature") or "")
        korean_name = str(record.get("korean_name") or "").strip()
        unit = str(record.get("unit") or "")
        segment = text[start:end]
        formatted_parts.append(
            _format_committee_feature_segment(
                segment,
                feature=feature,
                korean_name=korean_name,
                unit=unit,
            )
        )
        cursor = end
    formatted_parts.append(text[cursor:])
    return "".join(formatted_parts)


def _format_committee_feature_segment(
    text: str,
    *,
    feature: str,
    korean_name: str,
    unit: str,
) -> str:
    """Format one feature-specific clause in a committee sentence."""

    def format_parenthesized_value(match: re.Match[str]) -> str:
        raw_value = match.group("value").strip()
        formatted_value = format_value_with_unit(
            _committee_numeric_literal(raw_value),
            unit,
            feature,
        )
        return f"{korean_name}({formatted_value})"

    formatted_text = re.sub(
        rf"{re.escape(korean_name)}\((?P<value>[^)]*)\)",
        format_parenthesized_value,
        text,
    )

    def format_median_value(match: re.Match[str]) -> str:
        label = match.group("label")
        raw_value = match.group("value").strip()
        formatted_value = format_value_with_unit(
            _committee_numeric_literal(raw_value),
            unit,
            feature,
        )
        return f"{label} {formatted_value}"

    return re.sub(
        r"(?P<label>같은 산업의 중앙값은|같은 시장의 중앙값은)\s*"
        r"(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)",
        format_median_value,
        formatted_text,
    )


def _committee_numeric_literal(value: str) -> str:
    """Normalize comma-separated numeric literals before unit formatting."""
    cleaned = value.replace(",", "").strip()
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
        return cleaned
    return value


def _format_committee_percentile_phrases(text: str) -> str:
    """Turn terse percentile text into a friendlier relative-position phrase."""

    def replace_percentile(match: re.Match[str]) -> str:
        raw_value = match.group("value")
        try:
            percentile = float(raw_value)
        except (TypeError, ValueError):
            return match.group(0)
        if 45.0 <= percentile <= 55.0:
            position_text = "중간 수준입니다"
        elif percentile > 55.0:
            position_text = f"상위 {_format_percentile_gap(100.0 - percentile)}% 수준입니다"
        else:
            position_text = f"하위 {_format_percentile_gap(percentile)}% 수준입니다"
        return f"산업 안에서는 {position_text}"

    return re.sub(
        r"산업 안에서의 위치는\s*(?P<value>[-+]?\d+(?:\.\d+)?)백분위입니다",
        replace_percentile,
        text,
    )


def _format_percentile_gap(value: float) -> str:
    """Format a percentile gap without noisy trailing zeros."""
    rounded = round(max(0.0, min(100.0, value)), 1)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.1f}"


def _empty_dashboard_evidence_snapshot() -> dict[str, object]:
    """Return a friendly empty evidence snapshot for dashboard-first review."""
    return {
        "status": "not_requested",
        "source": "external_evidence",
        "enabled": False,
        "items": [],
        "providers": {},
        "has_critical_risk": False,
        "critical_terms": [],
        "message": "대시보드에서 아직 실시간 뉴스/웹/OpenDART 외부 근거를 수집하지 않았습니다.",
    }


def _dashboard_evidence_key(selected_row: pd.Series) -> str:
    """Build a Streamlit session key for cached dashboard evidence."""
    return (
        f"external_evidence:v2:{_stock_code_text(selected_row.get('stock_code'))}:"
        f"{_optional_int(selected_row.get('fiscal_year')) or 'latest'}"
    )


def _dashboard_evidence_as_of_date(selected_row: pd.Series) -> str:
    """Return the date cut-off for dashboard evidence collection."""
    fiscal_year = _optional_int(selected_row.get("fiscal_year"))
    if fiscal_year is not None:
        return min(date(fiscal_year, 12, 31), date.today()).isoformat()
    eval_year = _optional_int(selected_row.get("eval_year"))
    if eval_year is None:
        return date.today().isoformat()
    return min(date(eval_year, 12, 31), date.today()).isoformat()


def collect_dashboard_external_evidence(selected_row: pd.Series) -> dict[str, object]:
    """Collect live external evidence for the selected dashboard company on demand."""
    env = dict(os.environ)
    env["CAS_ENABLE_EXTERNAL_EVIDENCE"] = "1"
    env.setdefault("CAS_OPENDART_CORP_CODE_CACHE_PATH", "/private/tmp/cas_opendart_corp_codes.csv")
    return cast(
        dict[str, object],
        collect_external_evidence(
            company_name=str(selected_row.get("corp_name") or ""),
            stock_code=_stock_code_text(selected_row.get("stock_code")),
            corp_code=_optional_text(selected_row.get("corp_code")),
            as_of_date=_dashboard_evidence_as_of_date(selected_row),
            env=env,
        ),
    )


def resolve_dashboard_external_evidence(selected_row: pd.Series) -> dict[str, object]:
    """Return cached live evidence, collecting it automatically on first tab render."""
    evidence_key = _dashboard_evidence_key(selected_row)
    cached = st.session_state.get(evidence_key)
    if isinstance(cached, dict):
        return cast(dict[str, object], cached)
    try:
        with st.spinner("외부 근거를 자동 수집하고 2차 위원회 판단에 반영하는 중입니다..."):
            snapshot = collect_dashboard_external_evidence(selected_row)
    except Exception as error:  # pragma: no cover - runtime/network dependent
        snapshot = {
            "status": "error",
            "source": "external_evidence",
            "enabled": True,
            "items": [],
            "providers": {},
            "has_critical_risk": False,
            "critical_terms": [],
            "message": str(error),
        }
    st.session_state[evidence_key] = snapshot
    return snapshot


def to_stage2_risk_band(value: object) -> str:
    """Map dashboard risk-band labels to the Stage 2 internal labels."""
    label = str(_clean_dashboard_value(value) or "")
    return {
        "안정": "stable",
        "관찰": "watch",
        "고위험": "high_risk",
        "데이터 부족": "insufficient_data",
    }.get(label, label or "insufficient_data")


def format_stage2_risk_band(value: object) -> str:
    """Map Stage 2 internal risk-band labels back to dashboard Korean labels."""
    return STAGE2_RISK_BAND_LABELS.get(str(value), str(value))


def _dashboard_company_id(selected_row: pd.Series) -> str:
    """Build the same human-readable company-year key used by Stage 2."""
    market = str(selected_row.get("market") or "UNKNOWN")
    stock_code = _stock_code_text(selected_row.get("stock_code"))
    fiscal_year = _optional_int(selected_row.get("fiscal_year"))
    if fiscal_year is None:
        return f"{market}-{stock_code}"
    return f"{market}-{stock_code}-{fiscal_year}"


def _dashboard_top_drivers(local_shap: pd.DataFrame, *, limit: int = 5) -> list[dict[str, object]]:
    """Return Stage 2-compatible top driver records from local SHAP output."""
    if local_shap.empty:
        return []
    top_rows = local_shap.sort_values("abs_shap", ascending=False).head(limit)
    drivers: list[dict[str, object]] = []
    for record in top_rows.to_dict(orient="records"):
        name = str(record.get("feature") or "")
        if not name:
            continue
        drivers.append(
            {
                "name": name,
                "feature": name,
                "value": _optional_float(record.get("shap_value")),
                "abs_value": _optional_float(record.get("abs_shap")),
                "feature_value": _clean_dashboard_value(record.get("feature_value")),
            }
        )
    return drivers


def build_dashboard_model_view(
    prediction_row: pd.Series,
    local_shap: pd.DataFrame,
) -> dict[str, object]:
    """Build the read-only Stage 1 model_view shown and passed to Stage 2."""
    probability = _optional_float(prediction_row.get("prob_speculative"))
    risk_band = to_stage2_risk_band(prediction_row.get("risk_band"))
    model_label = to_stage2_model_label(prediction_row.get("predicted_label"))
    return {
        "source": "dashboard_prediction_scores",
        "model_name": "feature_43_xgboost",
        "model_version": "dashboard_artifacts",
        "prediction_label": model_label,
        "probability_speculative": probability,
        "y_proba": probability,
        "threshold": _optional_float(prediction_row.get("threshold"), default=0.5),
        "risk_band": risk_band,
        "risk_band_display": format_stage2_risk_band(risk_band),
        "top_drivers": _dashboard_top_drivers(local_shap),
        "probability_speculative_45": _optional_float(prediction_row.get("prob_speculative_45")),
        "threshold_45": _optional_float(prediction_row.get("threshold_45")),
        "threshold_45_it_services_review": _optional_float(
            prediction_row.get("threshold_45_it_services_review")
        ),
        "stage2_review_trigger": _optional_bool(prediction_row.get("stage2_review_trigger")),
        "stage2_secondary_trigger": _optional_bool(prediction_row.get("stage2_secondary_trigger")),
        "stage2_review_priority": str(
            _clean_dashboard_value(prediction_row.get("stage2_review_priority")) or "none"
        ),
        "trigger_reason_code": str(
            _clean_dashboard_value(prediction_row.get("trigger_reason_code")) or "none"
        ),
        "trigger_reason": str(
            _clean_dashboard_value(prediction_row.get("trigger_reason"))
            or "추가 위원회 검토 트리거 없음"
        ),
        "probability_speculative_overwarning_filter": _optional_float(
            prediction_row.get("prob_speculative_overwarning_filter")
        ),
        "threshold_overwarning_filter": _optional_float(
            prediction_row.get("threshold_overwarning_filter")
        ),
        "stage2_overwarning_filter_candidate": _optional_bool(
            prediction_row.get("stage2_overwarning_filter_candidate")
        ),
        "overwarning_filter_reason_code": str(
            _clean_dashboard_value(prediction_row.get("overwarning_filter_reason_code")) or "none"
        ),
        "overwarning_filter_reason": str(
            _clean_dashboard_value(prediction_row.get("overwarning_filter_reason"))
            or "과민 경고 보조필터 특이 신호 없음"
        ),
    }


def build_dashboard_committee_context(
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    local_shap: pd.DataFrame,
    peer_slice: pd.DataFrame,
    external_evidence_snapshot: dict[str, object] | None = None,
) -> dict[str, object] | None:
    """Run the deterministic Stage 2 review for the selected dashboard company."""
    if prediction_row is None:
        return None

    model_view = build_dashboard_model_view(prediction_row, local_shap)
    source_feature_row = _series_record(selected_row)
    source_feature_row["company_name"] = str(selected_row.get("corp_name") or "")
    source_feature_row["stock_code"] = _stock_code_text(selected_row.get("stock_code"))
    source_feature_row["company_id"] = _dashboard_company_id(selected_row)
    xgboost_result = dict(model_view)

    state_payload: dict[str, object] = {
        "company_id": source_feature_row["company_id"],
        "company_name": source_feature_row["company_name"],
        "market": str(selected_row.get("market") or "UNKNOWN"),
        "analysis_year": _optional_int(selected_row.get("eval_year")) or 0,
        "company_selection": build_company_selection_from_row(selected_row.to_dict()),
        "company_profile": {
            "company_id": source_feature_row["company_id"],
            "company_name": source_feature_row["company_name"],
            "stock_code": source_feature_row["stock_code"],
            "market": str(selected_row.get("market") or ""),
            "industry_macro_category": str(selected_row.get("industry_macro_category") or ""),
            "firm_size_group": str(selected_row.get("firm_size_group") or ""),
            "fiscal_year": _optional_int(selected_row.get("fiscal_year")),
            "eval_year": _optional_int(selected_row.get("eval_year")),
        },
        "source_feature_row": source_feature_row,
        "peer_comparison_rows": _frame_records(peer_slice),
        "model_view": model_view,
        "xgboost_result": xgboost_result,
        "news_cache_snapshot": external_evidence_snapshot or _empty_dashboard_evidence_snapshot(),
        "base_assessments": {},
        "committee_reviews": [],
        "agent_outputs": [],
        "agent_summary": {},
        "committee_view": {},
        "audit": [],
        "artifacts": {},
        "insufficient_data": False,
    }
    state = cast(AgentState, state_payload)
    state.update(rule_engine_node.run(state))
    stage2_result = _run_dashboard_stage2(state)
    state.update(stage2_result)

    return {
        "model_view": model_view,
        "rule_result": dict(state.get("rule_result") or {}),
        "agent_summary": dict(state.get("agent_summary") or {}),
        "committee_view": dict(state.get("committee_view") or {}),
        "final_confidence": state.get("final_confidence"),
    }


def _run_dashboard_stage2(state: AgentState) -> dict[str, object]:
    """Run Stage 2 using the dashboard-selected runner."""
    previous_runner = os.environ.get("CAS_STAGE2_RUNNER")
    dashboard_runner = (
        os.environ.get("CAS_DASHBOARD_STAGE2_RUNNER")
        or os.environ.get("CAS_STAGE2_RUNNER")
        or "deterministic"
    ).strip()
    if not dashboard_runner:
        dashboard_runner = "deterministic"
    os.environ["CAS_STAGE2_RUNNER"] = dashboard_runner
    try:
        return cast(dict[str, object], committee_node.run(state))
    finally:
        if previous_runner is None:
            os.environ.pop("CAS_STAGE2_RUNNER", None)
        else:
            os.environ["CAS_STAGE2_RUNNER"] = previous_runner


def _committee_evidence_frame(evidence_summary: object) -> pd.DataFrame:
    """Build a user-facing evidence summary table from committee_view."""
    if not isinstance(evidence_summary, list | tuple):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for item in evidence_summary:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "근거 출처": item.get("source", "-"),
                "요약": _humanize_evidence_summary(
                    source=item.get("source"),
                    summary=item.get("summary", "-"),
                ),
                "신뢰도": item.get("reliability", "-"),
            }
        )
    return pd.DataFrame(rows)


def _humanize_evidence_summary(*, source: object, summary: object) -> str:
    """Hide internal evidence status codes from user-facing tables."""
    text = str(summary)
    if str(source) != "news_cache":
        return text
    replacements = {
        "`not_requested`": "아직 수집하지 않음",
        "`dashboard_not_collected`": "아직 수집하지 않음",
        "`ready`": "수집 완료",
        "`no_results`": "검색 결과 없음",
        "`missing_credentials`": "API 키 미설정",
        "`partial_error`": "일부 수집 오류",
        "`disabled`": "수집 비활성화",
    }
    for raw, label in replacements.items():
        text = text.replace(raw, label)
    if "뉴스/공시 근거 번들 상태는" in text:
        return text.replace("뉴스/공시 근거 번들 상태는", "외부 근거 수집 상태는")
    return text


def format_scalar(value: object) -> str:
    """Format scalars for display."""
    if pd.isna(value):
        return "-"
    if isinstance(value, bool):
        return "예" if value else "아니오"
    if isinstance(value, int | float):
        number = float(value)
        if number.is_integer():
            return f"{number:,.0f}"
        if abs(number) >= 1000:
            return f"{number:,.2f}"
        return f"{number:.2f}"
    return str(value)


def format_percent(value: object) -> str:
    """Format probability-like values as percentages."""
    if pd.isna(value):
        return "-"
    try:
        return f"{float(str(value)) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def get_money_display_mode() -> str:
    """Return the selected KRW display mode from Streamlit session state."""
    return str(st.session_state.get("money_display_mode", "detailed"))


def format_krw_human(amount_won: float) -> str:
    """Format KRW amounts into Korean large-number units such as 억/만/원."""
    negative = amount_won < 0
    remaining = round(abs(amount_won))
    if remaining == 0:
        return "0원"

    parts: list[str] = []
    for unit_value, unit_label in ((10**12, "조"), (10**8, "억"), (10**4, "만")):
        chunk, remaining = divmod(remaining, unit_value)
        if chunk:
            parts.append(f"{chunk:,}{unit_label}")

    if remaining:
        parts.append(f"{remaining:,}원")

    if not parts:
        body = "0원"
    elif parts[-1].endswith("원"):
        body = " ".join(parts)
    else:
        body = " ".join(parts) + "원"

    return f"-{body}" if negative else body


def format_krw_eok(amount_won: float) -> str:
    """Format KRW amounts in 억 원 only."""
    eok_value = float(amount_won) / 100_000_000
    return f"{eok_value:,.2f}억 원"


def format_value_with_unit(value: object, unit: object, feature: str | None = None) -> str:
    """Format a value using the feature unit for user-facing display."""
    if pd.isna(value):
        return "-"

    unit_text = str(unit) if pd.notna(unit) else ""
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return str(value)

    if unit_text == "ratio":
        return cast(str, format_ratio_value(number, feature))
    if unit_text == "%p":
        return f"{number:.2f}%p"
    if unit_text == "KRW thousand":
        amount_won = number * 1000
        if get_money_display_mode() == "eok_only":
            return format_krw_eok(amount_won)
        return format_krw_human(amount_won)
    if unit_text == "year":
        return f"{round(number)}년"
    if unit_text == "0/1":
        if feature == "dividend_payer":
            return "O" if round(number) == 1 else "X"
        return "예" if round(number) == 1 else "아니오"
    if unit_text == "category":
        return str(value)
    return format_scalar(value)


def format_delta_with_unit(value: object, unit: object, feature: str | None = None) -> str:
    """Format a signed delta using the feature unit for comparison views."""
    if pd.isna(value):
        return "-"

    unit_text = str(unit) if pd.notna(unit) else ""
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return str(value)

    sign = "+" if number > 0 else ""
    if unit_text == "ratio":
        return cast(str, format_ratio_value(number, feature, signed=True))
    if unit_text == "%p":
        return f"{sign}{number:.2f}%p"
    if unit_text == "KRW thousand":
        amount_won = abs(number * 1000)
        base = (
            format_krw_eok(amount_won)
            if get_money_display_mode() == "eok_only"
            else format_krw_human(amount_won)
        )
        return f"{sign}{base}" if number != 0 else base
    if unit_text == "year":
        return f"{sign}{round(number)}년"
    return f"{sign}{format_scalar(number)}"


def format_percentile_label(value: object) -> str:
    """Format percentile-like values for tables."""
    if pd.isna(value):
        return "-"
    try:
        return f"{float(str(value)):.2f}백분위"
    except (TypeError, ValueError):
        return str(value)


def describe_unit(unit: str) -> str:
    """Return a short Korean label for a unit group."""
    mapping = {
        "ratio": "비율 변수",
        "%p": "퍼센트포인트 변수",
        "KRW thousand": "금액 변수",
        "0/1": "이진 변수",
        "year": "연도 변수",
        "category": "범주 변수",
        "": "기타 변수",
    }
    return mapping.get(unit, "기타 변수")


def get_feature_unit(feature: str, feature_map: pd.DataFrame) -> str:
    """Return the unit text for a feature from the feature map."""
    matched = feature_map.loc[feature_map["feature"] == feature, "unit"]
    if matched.empty or pd.isna(matched.iloc[0]):
        return ""
    return str(matched.iloc[0])


def get_feature_direction_label(feature: str) -> str:
    """Return a user-friendly interpretation direction for a feature."""
    return FEATURE_DIRECTION_LABELS.get(feature, "맥락에 따라 다름")


def style_direction_badge(value: object) -> str:
    """Return CSS styles for interpretation direction badges inside tables."""
    text = str(value)
    base_style = "font-weight:700;text-align:center;border-radius:999px;padding:0.15rem 0.45rem;"
    if "높을수록" in text or "O가" in text:
        return (
            f"{base_style}"
            "background-color:var(--cas-success-soft);"
            "color:var(--cas-success);"
            "border:1px solid var(--cas-success-border);"
        )
    if "낮을수록" in text or "아니오가" in text:
        return (
            f"{base_style}"
            "background-color:var(--cas-warning-soft);"
            "color:var(--cas-warning);"
            "border:1px solid var(--cas-warning-border);"
        )
    return (
        f"{base_style}"
        "background-color:var(--cas-neutral-soft);"
        "color:var(--cas-text);"
        "border:1px solid var(--cas-neutral-border);"
    )


def render_direction_badge_html(value: object) -> str:
    """Render an interpretation direction badge as HTML."""
    text = str(value)
    style = style_direction_badge(text)
    return f"<span style='{style}'>{escape(text)}</span>"


def render_risk_band_badge(risk_band: object) -> str:
    """Render a colored HTML badge for the risk band."""
    label = str(risk_band) if pd.notna(risk_band) else "-"
    style_map = {
        "안정": {
            "bg": "var(--cas-success-soft)",
            "fg": "var(--cas-success)",
            "border": "var(--cas-success-border)",
        },
        "관찰": {
            "bg": "var(--cas-warning-soft)",
            "fg": "var(--cas-warning)",
            "border": "var(--cas-warning-border)",
        },
        "고위험": {
            "bg": "var(--cas-risk-soft)",
            "fg": "var(--cas-risk)",
            "border": "var(--cas-risk-border)",
        },
    }
    style = style_map.get(
        label,
        {
            "bg": "var(--cas-neutral-soft)",
            "fg": "var(--cas-text)",
            "border": "var(--cas-neutral-border)",
        },
    )
    return (
        f"<div style='display:inline-block;padding:0.45rem 0.8rem;border-radius:999px;"
        f"background:{style['bg']};color:{style['fg']};border:1px solid {style['border']};"
        "font-weight:700;font-size:0.95rem;'>"
        f"{label}</div>"
    )


def render_decision_badge(label: object, *, muted: bool = False) -> str:
    """Render a colored badge for model and committee decisions."""
    text = str(label) if pd.notna(label) else "-"
    style_map = {
        "적격": {
            "bg": "var(--cas-success-soft)",
            "fg": "var(--cas-success)",
            "border": "var(--cas-success-border)",
        },
        "투자적격": {
            "bg": "var(--cas-success-soft)",
            "fg": "var(--cas-success)",
            "border": "var(--cas-success-border)",
        },
        "보류": {
            "bg": "var(--cas-warning-soft)",
            "fg": "var(--cas-warning)",
            "border": "var(--cas-warning-border)",
        },
        "부적격": {
            "bg": "var(--cas-risk-soft)",
            "fg": "var(--cas-risk)",
            "border": "var(--cas-risk-border)",
        },
        "투기등급": {
            "bg": "var(--cas-risk-soft)",
            "fg": "var(--cas-risk)",
            "border": "var(--cas-risk-border)",
        },
        "차이 있음": {
            "bg": "var(--cas-warning-soft)",
            "fg": "var(--cas-warning)",
            "border": "var(--cas-warning-border)",
        },
        "일치": {
            "bg": "var(--cas-success-soft)",
            "fg": "var(--cas-success)",
            "border": "var(--cas-success-border)",
        },
        "발동": {
            "bg": "var(--cas-risk-soft)",
            "fg": "var(--cas-risk)",
            "border": "var(--cas-risk-border)",
        },
        "후보 검토": {
            "bg": "var(--cas-warning-soft)",
            "fg": "var(--cas-warning)",
            "border": "var(--cas-warning-border)",
        },
        "미발동": {
            "bg": "var(--cas-neutral-soft)",
            "fg": "var(--cas-text)",
            "border": "var(--cas-neutral-border)",
        },
    }
    if muted:
        style = {
            "bg": "var(--cas-neutral-soft)",
            "fg": "var(--cas-text)",
            "border": "var(--cas-neutral-border)",
        }
    else:
        style = style_map.get(
            text,
            {
                "bg": "var(--cas-neutral-soft)",
                "fg": "var(--cas-text)",
                "border": "var(--cas-neutral-border)",
            },
        )
    return (
        f"<div style='display:inline-block;padding:0.45rem 0.8rem;border-radius:999px;"
        f"background:{style['bg']};color:{style['fg']};border:1px solid {style['border']};"
        "font-weight:700;font-size:0.95rem;'>"
        f"{escape(text)}</div>"
    )


def render_bold_value_block(
    container: st.delta_generator.DeltaGenerator, label: str, value: object
) -> None:
    """Render a bold label and value inside a consistent overview card."""
    container.markdown(
        (
            f"<div style='min-height:104px;padding:0.9rem 1rem;border-radius:8px;"
            f"background:{COLOR_CARD_BG};border:1px solid {COLOR_CARD_BORDER};"
            "display:flex;flex-direction:column;justify-content:space-between;"
            f"margin-bottom:0.5rem;box-shadow:{CARD_SHADOW};'>"
            f"<div style='font-size:0.95rem;font-weight:700;color:{COLOR_CARD_LABEL};'>"
            f"{escape(label)}"
            "</div>"
            f"<div style='font-size:1.2rem;line-height:1.45;font-weight:700;color:{COLOR_CARD_VALUE};"
            "word-break:keep-all;'>"
            f"{escape(str(value))}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_badge_value_block(
    container: st.delta_generator.DeltaGenerator,
    label: str,
    badge_html: str,
) -> None:
    """Render a bold label and badge inside the same overview card layout."""
    container.markdown(
        (
            f"<div style='min-height:104px;padding:0.9rem 1rem;border-radius:8px;"
            f"background:{COLOR_CARD_BG};border:1px solid {COLOR_CARD_BORDER};"
            "display:flex;flex-direction:column;justify-content:space-between;"
            f"margin-bottom:0.5rem;box-shadow:{CARD_SHADOW};'>"
            f"<div style='font-size:0.95rem;font-weight:700;color:{COLOR_CARD_LABEL};'>"
            f"{escape(label)}"
            "</div>"
            f"<div style='line-height:1.45;'>{badge_html}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_value_detail_block(
    container: st.delta_generator.DeltaGenerator,
    label: str,
    value: object,
    description: str | None = None,
    badge_html: str | None = None,
) -> None:
    """Render a value card with an optional short description."""
    badge_section = ""
    if badge_html:
        badge_section = f"<div style='margin-top:0.45rem;'>{badge_html}</div>"
    description_html = ""
    if description:
        description_html = (
            f"<div style='font-size:0.88rem;line-height:1.45;color:{COLOR_CARD_LABEL};"
            "margin-top:0.45rem;word-break:keep-all;'>"
            f"{escape(description)}"
            "</div>"
        )

    container.markdown(
        (
            f"<div style='min-height:136px;padding:0.9rem 1rem;border-radius:8px;"
            f"background:{COLOR_CARD_BG};border:1px solid {COLOR_CARD_BORDER};"
            "display:flex;flex-direction:column;justify-content:space-between;"
            f"margin-bottom:0.5rem;box-shadow:{CARD_SHADOW};'>"
            f"<div style='font-size:0.95rem;font-weight:700;color:{COLOR_CARD_LABEL};'>"
            f"{escape(label)}"
            "</div>"
            f"<div style='font-size:1.18rem;line-height:1.45;font-weight:700;color:{COLOR_CARD_VALUE};"
            "word-break:keep-all;'>"
            f"{escape(str(value))}"
            "</div>"
            f"{badge_section}"
            f"{description_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_text_card(
    container: st.delta_generator.DeltaGenerator,
    label: str,
    body: str,
) -> None:
    """Render an explanatory gray card with bold label."""
    container.markdown(
        (
            f"<div style='min-height:120px;padding:0.95rem 1rem;border-radius:8px;"
            f"background:{COLOR_CARD_BG};border:1px solid {COLOR_CARD_BORDER};"
            "display:flex;flex-direction:column;justify-content:space-between;"
            f"margin-bottom:0.5rem;box-shadow:{CARD_SHADOW};'>"
            f"<div style='font-size:0.95rem;font-weight:700;color:{COLOR_CARD_LABEL};'>"
            f"{escape(label)}"
            "</div>"
            f"<div style='font-size:0.97rem;line-height:1.6;color:{COLOR_CARD_VALUE};"
            "word-break:keep-all;'>"
            f"{escape(body)}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_bullet_card(
    container: st.delta_generator.DeltaGenerator,
    label: str,
    items: list[str],
    accent_color: str,
    empty_message: str,
) -> None:
    """Render a summary card with short bullet items."""
    if items:
        bullet_html = "".join(
            (f"<li style='margin:0 0 0.35rem 0;'>{escape(item)}</li>") for item in items
        )
        body_html = (
            "<ul style='margin:0.15rem 0 0 1rem;padding:0;"
            f"color:{COLOR_CARD_VALUE};font-size:0.94rem;line-height:1.6;'>"
            f"{bullet_html}"
            "</ul>"
        )
    else:
        body_html = (
            f"<div style='font-size:0.95rem;line-height:1.6;color:{COLOR_CARD_LABEL};'>"
            f"{escape(empty_message)}"
            "</div>"
        )

    container.markdown(
        (
            f"<div style='min-height:168px;padding:0.95rem 1rem;border-radius:8px;"
            f"background:{COLOR_CARD_BG};border:1px solid {COLOR_CARD_BORDER};"
            f"border-left:6px solid {accent_color};"
            "display:flex;flex-direction:column;justify-content:flex-start;"
            f"margin-bottom:0.5rem;box-shadow:{CARD_SHADOW};'>"
            f"<div style='font-size:0.95rem;font-weight:700;color:{COLOR_CARD_LABEL};margin-bottom:0.35rem;'>"
            f"{escape(label)}"
            "</div>"
            f"{body_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_summary_banner(
    label: str,
    body: str,
    accent_color: str,
) -> None:
    """Render a wide summary banner for quick interpretation."""
    st.markdown(
        (
            f"<div style='padding:0.95rem 1.05rem;border-radius:8px;"
            f"background:{COLOR_CARD_BG};border:1px solid {COLOR_CARD_BORDER};"
            f"border-left:6px solid {accent_color};box-shadow:{CARD_SHADOW};"
            "margin:0.25rem 0 0.9rem 0;'>"
            f"<div style='font-size:0.93rem;font-weight:700;color:{COLOR_CARD_LABEL};margin-bottom:0.3rem;'>"
            f"{escape(label)}"
            "</div>"
            f"<div style='font-size:1rem;line-height:1.65;color:{COLOR_CARD_VALUE};word-break:keep-all;'>"
            f"{escape(body)}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _first_committee_text(items: list[str], fallback: str) -> str:
    """Return the first non-empty committee sentence for highlight cards."""
    for item in items:
        text = str(item).strip()
        if text:
            return text
    return fallback


def _normalize_committee_text(text: object) -> str:
    """Normalize committee text while preserving its full meaning."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _split_committee_highlight_fragments(text: object) -> list[str]:
    """Split dense numbered committee text into readable highlight fragments."""
    normalized = _normalize_committee_text(text)
    if not normalized:
        return []

    numbered_matches = re.findall(r"\(\d+\)\s*(.*?)(?=\s*\(\d+\)\s*|$)", normalized)
    if numbered_matches:
        return [match.strip(" ,;") for match in numbered_matches if match.strip()]

    sentence_parts = re.split(r"(?<=[.!?。])\s+", normalized)
    return [part.strip(" ,;") for part in sentence_parts if part.strip()]


def _committee_highlight_items(
    items: list[str],
    fallback: str,
    *,
    max_items: int = 3,
) -> list[str]:
    """Build short highlight items without truncating text."""
    highlights: list[str] = []
    for item in items:
        fragments = _split_committee_highlight_fragments(item)
        if len(fragments) > 1:
            highlights.extend(fragments)
        elif fragments:
            highlights.append(fragments[0])
        if len(highlights) >= max_items:
            break
    return highlights[:max_items] or [fallback]


def _committee_highlight_body_html(items: list[str]) -> str:
    """Render highlight items without ellipses."""
    clean_items = [item for item in (_normalize_committee_text(item) for item in items) if item]
    if len(clean_items) == 1:
        return escape(clean_items[0])
    list_html = "".join(f"<li>{escape(item)}</li>" for item in clean_items)
    return f"<ul>{list_html}</ul>"


def render_committee_key_highlights(
    *,
    committee_label: str,
    model_display_label: str,
    decision_gap_label: str,
    veto_label: str,
    final_confidence: object,
    summary_text: str,
    conflict_text: str,
    final_memo: str,
    risk_items: list[str],
    mitigation_items: list[str],
) -> None:
    """Render the committee result as a quick executive summary."""
    top_risk_items = _committee_highlight_items(
        risk_items,
        "위원회가 별도로 강조한 위험 요인은 없습니다.",
    )
    top_mitigation_items = _committee_highlight_items(
        mitigation_items,
        "위원회가 별도로 강조한 완화 요인은 없습니다.",
    )
    checkpoint_text = final_memo or conflict_text or summary_text
    decision_meta = (
        f"1차 모델: {model_display_label} / 판단 차이: {decision_gap_label} / "
        f"강제 경고: {veto_label} / 신뢰도: {format_percent(final_confidence)}"
    )
    cards = [
        (
            "",
            "판단 상태",
            decision_meta,
        ),
        (
            "risk",
            "가장 먼저 볼 위험",
            _committee_highlight_body_html(top_risk_items),
        ),
        (
            "mitigate",
            "완화해서 본 근거",
            _committee_highlight_body_html(top_mitigation_items),
        ),
        (
            "warning",
            "사용자 체크 포인트",
            _committee_highlight_body_html(
                _committee_highlight_items([checkpoint_text], checkpoint_text, max_items=2)
            ),
        ),
    ]
    card_html = "".join(
        (
            f"<div class='committee-highlight-card {escape(css_class)}'>"
            f"<div class='committee-highlight-title'>{escape(title)}</div>"
            f"<div class='committee-highlight-body'>{body}</div>"
            "</div>"
        )
        for css_class, title, body in cards
    )
    st.markdown(
        (
            "<div class='committee-decision-strip'>"
            "<div class='committee-decision-topline'>"
            "<span class='committee-decision-label'>2차 위원회 최종 판단</span>"
            f"{render_decision_badge(committee_label)}"
            "</div>"
            f"<p class='committee-decision-summary'>{escape(_normalize_committee_text(summary_text))}</p>"
            "</div>"
            f"<div class='committee-highlight-grid'>{card_html}</div>"
        ),
        unsafe_allow_html=True,
    )


def _committee_detail_section_html(title: str, body: str) -> str:
    if not body.strip():
        return ""
    return (
        "<div class='committee-detail-section'>"
        f"<div class='committee-detail-heading'>{escape(title)}</div>"
        f"<div class='committee-detail-text'>{escape(body)}</div>"
        "</div>"
    )


def _committee_detail_list_html(title: str, items: list[str], empty_message: str) -> str:
    clean_items = [str(item).strip() for item in items if str(item).strip()]
    if clean_items:
        list_html = "".join(f"<li>{escape(item)}</li>" for item in clean_items)
    else:
        list_html = f"<li>{escape(empty_message)}</li>"
    return (
        "<div class='committee-detail-section'>"
        f"<div class='committee-detail-heading'>{escape(title)}</div>"
        f"<ul>{list_html}</ul>"
        "</div>"
    )


def render_committee_full_review(
    *,
    summary_text: str,
    conflict_text: str,
    final_memo: str,
    risk_items: list[str],
    mitigation_items: list[str],
) -> None:
    """Render the complete committee rationale below the highlighted summary."""
    sections_html = "".join(
        [
            _committee_detail_section_html("판단 차이 해석", summary_text),
            _committee_detail_section_html("왜 이렇게 판단했나", conflict_text),
            _committee_detail_list_html(
                "주의해서 볼 점",
                risk_items,
                "위원회가 별도로 강조한 위험 요인은 없습니다.",
            ),
            _committee_detail_list_html(
                "긍정적으로 본 점",
                mitigation_items,
                "위원회가 별도로 강조한 완화 요인은 없습니다.",
            ),
            _committee_detail_section_html("최종 검토 의견", final_memo),
        ]
    )
    with st.expander("상세 판단 전문", expanded=False):
        st.markdown(
            (
                "<div class='committee-detail-flow'>"
                "<div class='committee-detail-title'>전체 검토 내용</div>"
                f"{sections_html}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_list_card(
    container: st.delta_generator.DeltaGenerator,
    label: str,
    items: list[str],
    accent_color: str,
) -> None:
    """Render a structured list card for short bullet summaries."""
    list_html = "".join(
        (f"<li style='margin-bottom:0.38rem;'>{escape(item)}</li>")
        for item in items
        if str(item).strip()
    )
    if not list_html:
        list_html = "<li>요약할 항목이 없습니다.</li>"

    container.markdown(
        (
            f"<div style='min-height:188px;padding:0.95rem 1rem;border-radius:8px;"
            f"background:{COLOR_CARD_BG};border:1px solid {COLOR_CARD_BORDER};"
            f"border-top:4px solid {accent_color};box-shadow:{CARD_SHADOW};"
            "margin-bottom:0.5rem;'>"
            f"<div style='font-size:0.95rem;font-weight:700;color:{COLOR_CARD_LABEL};margin-bottom:0.55rem;'>"
            f"{escape(label)}"
            "</div>"
            f"<ul style='margin:0;padding-left:1.15rem;font-size:0.97rem;line-height:1.65;color:{COLOR_CARD_VALUE};'>"
            f"{list_html}"
            "</ul>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_badge_hint_card(
    container: st.delta_generator.DeltaGenerator,
    label: str,
    badge_items: list[tuple[str, str]],
    accent_color: str,
    empty_message: str,
) -> None:
    """Render a compact card containing inline interpretation badges."""
    if badge_items:
        badge_html = "".join(
            (
                "<div style='display:flex;align-items:center;gap:0.45rem;flex-wrap:wrap;"
                "margin:0 0 0.45rem 0;'>"
                f"<span style='font-size:0.92rem;font-weight:700;color:{COLOR_CARD_VALUE};'>{escape(name)}</span>"
                f"{render_direction_badge_html(direction)}"
                "</div>"
            )
            for name, direction in badge_items
        )
    else:
        badge_html = (
            f"<div style='font-size:0.93rem;line-height:1.55;color:{COLOR_CARD_LABEL};'>"
            f"{escape(empty_message)}"
            "</div>"
        )

    container.markdown(
        (
            f"<div style='min-height:112px;padding:0.85rem 0.95rem;border-radius:8px;"
            f"background:{COLOR_CARD_BG};border:1px solid {COLOR_CARD_BORDER};"
            f"border-top:4px solid {accent_color};box-shadow:{CARD_SHADOW};"
            "margin:-0.15rem 0 0.6rem 0;'>"
            f"<div style='font-size:0.92rem;font-weight:700;color:{COLOR_CARD_LABEL};margin-bottom:0.45rem;'>"
            f"{escape(label)}"
            "</div>"
            f"{badge_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def parse_llm_report_sections(text: str) -> dict[str, list[str]]:
    """Parse bracketed report sections from the LLM output."""
    sections: dict[str, list[str]] = {
        "한줄 판단": [],
        "핵심 위험 요인": [],
        "완화 요인": [],
        "종합 의견": [],
    }
    current: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header_match = re.fullmatch(r"\[(.+?)\]", line)
        if header_match:
            title = header_match.group(1).strip()
            current = title if title in sections else None
            continue
        if current is None:
            continue
        cleaned = re.sub(r"^[-*•]\s*", "", line).strip()
        if cleaned:
            sections[current].append(cleaned)
    return sections


def build_exportable_llm_report(
    *,
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    model: str,
    output_format_label: str,
    report_text: str,
    local_shap: pd.DataFrame,
    peer_slice: pd.DataFrame,
    feature_map: pd.DataFrame,
) -> str:
    """Build a copy/export-friendly markdown report."""
    header_lines = [
        "# AI 심사 메모",
        "",
        f"- 기업명: {selected_row.get('corp_name')}",
        f"- 종목코드: {_stock_code_text(selected_row.get('stock_code'))}",
        f"- 시장: {to_market_label(selected_row.get('market'))}",
        f"- 산업: {to_industry_label(selected_row.get('industry_macro_category'))}",
        f"- 규모: {to_size_label(selected_row.get('firm_size_group'))}",
        f"- 회계연도: {format_scalar(selected_row.get('fiscal_year'))}",
        f"- 사용 모델: {model}",
        f"- 출력 형식: {output_format_label}",
    ]
    if prediction_row is not None:
        header_lines.extend(
            [
                f"- 투기등급 확률: {format_percent(prediction_row.get('prob_speculative'))}",
                f"- 예측 라벨: {to_prediction_label(prediction_row.get('predicted_label'))}",
                f"- 위험 밴드: {prediction_row.get('risk_band')}",
                f"- 판정 기준선: {format_scalar(prediction_row.get('threshold'))}",
            ]
        )
    local_frame = _prepare_local_driver_report_frame(local_shap, feature_map, top_n=5)
    peer_frame = _prepare_peer_report_frame(peer_slice, feature_map, top_n=5)
    header_lines.extend(["", "## 심사 메모", "", report_text.strip()])
    if not local_frame.empty:
        header_lines.extend(
            [
                "",
                "## 주요 설명 변수 표",
                "",
                _markdown_table_from_frame(
                    local_frame,
                    ["표시명", "실제값", "영향방향", "일반 해석 방향", "SHAP 표시"],
                ),
            ]
        )
    if not peer_frame.empty:
        header_lines.extend(
            [
                "",
                "## 동종업계 비교 표",
                "",
                _markdown_table_from_frame(
                    peer_frame,
                    [
                        "표시명",
                        "선택 기업",
                        "산업 중앙값",
                        "시장 중앙값",
                        "산업 내 위치",
                        "일반 해석 방향",
                    ],
                ),
            ]
        )
    return "\n".join(header_lines).strip() + "\n"


def build_onepage_llm_report(
    *,
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    model: str,
    output_format_label: str,
    sections: dict[str, list[str]],
    local_shap: pd.DataFrame,
    peer_slice: pd.DataFrame,
    feature_map: pd.DataFrame,
) -> str:
    """Build a one-page compact markdown memo."""
    top_local = local_shap.head(3).copy() if not local_shap.empty else pd.DataFrame()
    if not top_local.empty:
        top_local["표시명"] = top_local["feature"].map(
            lambda value: display_name(value, feature_map)
        )
        top_local["실제값"] = top_local.apply(
            lambda row: format_value_with_unit(
                row["feature_value"],
                get_feature_unit(str(row["feature"]), feature_map),
                str(row["feature"]),
            ),
            axis=1,
        )
    peer_summary = peer_slice.copy()
    if not peer_summary.empty:
        peer_summary["distance_from_industry_mid"] = (
            peer_summary["industry_percentile"] - 50.0
        ).abs()
        peer_summary = peer_summary.sort_values("distance_from_industry_mid", ascending=False).head(
            3
        )
        peer_summary["표시명"] = peer_summary["feature"].map(
            lambda value: display_name(value, feature_map)
        )
        peer_summary["산업 대비 차이"] = peer_summary.apply(
            lambda row: format_delta_with_unit(
                row["value"] - row["industry_median"],
                get_feature_unit(str(row["feature"]), feature_map),
                str(row["feature"]),
            ),
            axis=1,
        )

    lines = [
        "# 원페이지 심사 메모",
        "",
        "## 기업 개요",
        f"- 기업명: {selected_row.get('corp_name')}",
        f"- 종목코드: {_stock_code_text(selected_row.get('stock_code'))}",
        f"- 시장/산업: {to_market_label(selected_row.get('market'))} / {to_industry_label(selected_row.get('industry_macro_category'))}",
        f"- 규모/회계연도: {to_size_label(selected_row.get('firm_size_group'))} / {format_scalar(selected_row.get('fiscal_year'))}",
        f"- 사용 모델: {model}",
        f"- 출력 형식: {output_format_label}",
    ]
    if prediction_row is not None:
        lines.extend(
            [
                f"- 투기등급 확률: {format_percent(prediction_row.get('prob_speculative'))}",
                f"- 예측 라벨: {to_prediction_label(prediction_row.get('predicted_label'))}",
                f"- 위험 밴드: {prediction_row.get('risk_band')}",
                f"- 판정 기준선: {format_scalar(prediction_row.get('threshold'))}",
            ]
        )

    headline = " ".join(sections.get("한줄 판단", [])).strip()
    if headline:
        lines.extend(["", "## 한줄 판단", headline])

    risk_items = sections.get("핵심 위험 요인", [])[:3]
    if risk_items:
        lines.extend(["", "## 핵심 위험 요인"])
        lines.extend([f"- {item}" for item in risk_items])

    mitigate_items = sections.get("완화 요인", [])[:2]
    if mitigate_items:
        lines.extend(["", "## 완화 요인"])
        lines.extend([f"- {item}" for item in mitigate_items])

    if not top_local.empty:
        lines.extend(["", "## 주요 설명 변수"])
        for row in top_local.to_dict(orient="records"):
            direction = "위험 증가" if float(row["shap_value"]) > 0 else "위험 완화"
            lines.append(f"- {row['표시명']}: {row['실제값']} ({direction})")
        lines.extend(
            [
                "",
                _markdown_table_from_frame(
                    top_local.rename(columns={"표시명": "지표", "실제값": "실제값"}),
                    ["지표", "실제값"],
                ),
            ]
        )

    if not peer_summary.empty:
        lines.extend(["", "## 동종업계 비교 핵심 차이"])
        for row in peer_summary.to_dict(orient="records"):
            lines.append(
                f"- {row['표시명']}: 산업 중앙값 대비 {row['산업 대비 차이']}, 산업 내 위치 {format_percentile_label(row['industry_percentile'])}"
            )
        lines.extend(
            [
                "",
                _markdown_table_from_frame(
                    peer_summary.rename(
                        columns={"표시명": "지표", "산업 대비 차이": "산업 대비 차이"}
                    ),
                    ["지표", "산업 대비 차이"],
                ),
            ]
        )

    opinion = " ".join(sections.get("종합 의견", [])).strip()
    if opinion:
        lines.extend(["", "## 종합 의견", opinion])

    return "\n".join(lines).strip() + "\n"


def _html_list(items: list[str]) -> str:
    """Render list items for HTML report sections."""
    if not items:
        return "<li>해당 사항이 없습니다.</li>"
    return "".join(f"<li>{escape(item)}</li>" for item in items if str(item).strip())


def _prepare_local_driver_report_frame(
    local_shap: pd.DataFrame,
    feature_map: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Prepare a compact local SHAP frame for report tables and charts."""
    if local_shap.empty:
        return pd.DataFrame()
    frame = local_shap.sort_values("abs_shap", ascending=False).head(top_n).copy()
    frame["표시명"] = frame["feature"].map(lambda value: display_name(str(value), feature_map))
    frame["실제값"] = frame.apply(
        lambda row: format_value_with_unit(
            row["feature_value"],
            get_feature_unit(str(row["feature"]), feature_map),
            str(row["feature"]),
        ),
        axis=1,
    )
    frame["영향방향"] = frame["shap_value"].map(
        lambda value: "위험 증가" if float(value) > 0 else "위험 완화"
    )
    frame["SHAP 표시"] = frame["shap_value"].map(lambda value: f"{float(value):.2f}")
    frame["|SHAP| 표시"] = frame["abs_shap"].map(lambda value: f"{float(value):.2f}")
    frame["일반 해석 방향"] = frame["feature"].map(
        lambda value: get_feature_direction_label(str(value))
    )
    return frame


def _prepare_peer_report_frame(
    peer_slice: pd.DataFrame,
    feature_map: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Prepare a compact peer-comparison frame for report tables and charts."""
    if peer_slice.empty:
        return pd.DataFrame()
    frame = peer_slice.copy()
    frame["distance_from_industry_mid"] = (frame["industry_percentile"] - 50.0).abs()
    frame = frame.sort_values("distance_from_industry_mid", ascending=False).head(top_n).copy()
    frame["표시명"] = frame["feature"].map(lambda value: display_name(str(value), feature_map))
    frame["선택 기업"] = frame.apply(
        lambda row: format_value_with_unit(
            row["value"],
            get_feature_unit(str(row["feature"]), feature_map),
            str(row["feature"]),
        ),
        axis=1,
    )
    frame["산업 중앙값"] = frame.apply(
        lambda row: format_value_with_unit(
            row["industry_median"],
            get_feature_unit(str(row["feature"]), feature_map),
            str(row["feature"]),
        ),
        axis=1,
    )
    frame["시장 중앙값"] = frame.apply(
        lambda row: format_value_with_unit(
            row["market_median"],
            get_feature_unit(str(row["feature"]), feature_map),
            str(row["feature"]),
        ),
        axis=1,
    )
    frame["산업 내 위치"] = frame["industry_percentile"].map(format_percentile_label)
    frame["일반 해석 방향"] = frame["feature"].map(
        lambda value: get_feature_direction_label(str(value))
    )
    return frame


def _markdown_table_from_frame(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render a simple markdown table from a dataframe."""
    if frame.empty:
        return "해당 내용이 없습니다."
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row.get(column, "-")) for column in columns) + " |"
        for row in frame.loc[:, columns].to_dict(orient="records")
    ]
    return "\n".join([header, divider, *rows])


def _html_table_from_frame(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render an HTML table from a dataframe."""
    if frame.empty:
        return "<p>해당 내용이 없습니다.</p>"
    header_html = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body_rows = []
    for row in frame.loc[:, columns].to_dict(orient="records"):
        cells = []
        for column in columns:
            value = row.get(column, "-")
            if column == "일반 해석 방향":
                cells.append(f"<td>{render_direction_badge_html(value)}</td>")
            else:
                cells.append(f"<td>{escape(str(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<div class='table-wrap'><table class='report-table'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def _html_shap_bar_rows(frame: pd.DataFrame) -> str:
    """Render compact inline bars for top local SHAP features."""
    if frame.empty:
        return "<p>주요 설명 변수 그래프를 생성할 수 없습니다.</p>"
    max_abs = max(float(frame["abs_shap"].max()), 1e-9)
    rows: list[str] = []
    for row in frame.to_dict(orient="records"):
        width = max(8.0, (float(row["abs_shap"]) / max_abs) * 100.0)
        color = COLOR_RISK if str(row["영향방향"]) == "위험 증가" else COLOR_MITIGATE
        rows.append(
            "<div class='mini-bar-row'>"
            f"<div class='mini-bar-label'>{escape(str(row['표시명']))}</div>"
            "<div class='mini-bar-track'>"
            f"<div class='mini-bar-fill' style='width:{width:.1f}%;background:{color};'></div>"
            "</div>"
            f"<div class='mini-bar-value'>{escape(str(row['SHAP 표시']))}</div>"
            "</div>"
        )
    return "".join(rows)


def _html_percentile_rows(frame: pd.DataFrame) -> str:
    """Render compact percentile bars for peer-comparison context."""
    if frame.empty:
        return "<p>동종업계 비교 그래프를 생성할 수 없습니다.</p>"
    rows: list[str] = []
    for row in frame.to_dict(orient="records"):
        percentile = float(row.get("industry_percentile", 0.0))
        rows.append(
            "<div class='mini-bar-row'>"
            f"<div class='mini-bar-label'>{escape(str(row['표시명']))}</div>"
            "<div class='mini-bar-track'>"
            f"<div class='mini-bar-fill' style='width:{percentile:.1f}%;background:{COLOR_COMPANY};'></div>"
            "</div>"
            f"<div class='mini-bar-value'>{escape(format_percentile_label(percentile))}</div>"
            "</div>"
        )
    return "".join(rows)


def build_html_report(
    *,
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    model: str,
    output_format_label: str,
    sections: dict[str, list[str]],
    report_text: str,
    local_shap: pd.DataFrame,
    peer_slice: pd.DataFrame,
    feature_map: pd.DataFrame,
) -> str:
    """Build a print-friendly detailed HTML report."""
    probability = (
        format_percent(prediction_row.get("prob_speculative"))
        if prediction_row is not None
        else "-"
    )
    predicted_label = (
        to_prediction_label(prediction_row.get("predicted_label"))
        if prediction_row is not None
        else "-"
    )
    risk_band = str(prediction_row.get("risk_band")) if prediction_row is not None else "-"
    threshold = (
        format_scalar(prediction_row.get("threshold")) if prediction_row is not None else "-"
    )
    headline = " ".join(sections.get("한줄 판단", [])).strip() or "심사 요약이 생성되지 않았습니다."
    opinion = " ".join(sections.get("종합 의견", [])).strip()
    local_frame = _prepare_local_driver_report_frame(local_shap, feature_map, top_n=5)
    peer_frame = _prepare_peer_report_frame(peer_slice, feature_map, top_n=5)
    local_table_html = _html_table_from_frame(
        local_frame,
        ["표시명", "실제값", "영향방향", "일반 해석 방향", "SHAP 표시"],
    )
    peer_table_html = _html_table_from_frame(
        peer_frame,
        ["표시명", "선택 기업", "산업 중앙값", "시장 중앙값", "산업 내 위치", "일반 해석 방향"],
    )
    shap_chart_html = _html_shap_bar_rows(local_frame)
    percentile_chart_html = _html_percentile_rows(peer_frame)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI 심사 보고서</title>
  <style>
    @page {{
      size: A4;
      margin: 18mm 16mm;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
      background: #f3f5f9;
      color: #1f2937;
      margin: 0;
      padding: 32px;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}
    .page {{
      max-width: 960px;
      margin: 0 auto;
      background: white;
      border-radius: 20px;
      box-shadow: 0 10px 30px rgba(15,23,42,0.08);
      overflow: hidden;
    }}
    .header {{
      padding: 28px 32px;
      background: linear-gradient(135deg, #e9eefb 0%, #f8fafc 100%);
      border-bottom: 1px solid #e5e7eb;
    }}
    .header-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .brand-mark {{
      width: 46px;
      height: 46px;
      border-radius: 14px;
      background: linear-gradient(135deg, #1d4ed8 0%, #60a5fa 100%);
      color: white;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.02em;
      box-shadow: 0 6px 18px rgba(29, 78, 216, 0.18);
    }}
    .brand-copy {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .brand-title {{
      font-size: 14px;
      font-weight: 800;
      color: #0f172a;
      letter-spacing: 0.02em;
    }}
    .brand-subtitle {{
      font-size: 12px;
      color: #64748b;
      line-height: 1.4;
    }}
    .doc-badges {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }}
    .doc-chip {{
      padding: 7px 11px;
      border-radius: 999px;
      background: rgba(255,255,255,0.82);
      border: 1px solid #d8dfeb;
      color: #475569;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .eyebrow {{
      font-size: 13px;
      font-weight: 700;
      color: #5c6473;
      letter-spacing: 0.02em;
      margin-bottom: 8px;
    }}
    h1 {{
      margin: 0 0 10px 0;
      font-size: 28px;
    }}
    .summary {{
      line-height: 1.7;
      color: #374151;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      padding: 24px 32px 8px 32px;
    }}
    .meta-card {{
      background: #f7f8fb;
      border: 1px solid #e3e7ef;
      border-radius: 14px;
      padding: 14px 16px;
    }}
    .meta-label {{
      font-size: 13px;
      font-weight: 700;
      color: #5c6473;
      margin-bottom: 6px;
    }}
    .meta-value {{
      font-size: 18px;
      font-weight: 700;
      color: #1f2937;
    }}
    .body {{
      padding: 8px 32px 32px 32px;
    }}
    .section {{
      margin-top: 24px;
      padding: 18px 20px;
      border-radius: 16px;
      background: #f9fafb;
      border: 1px solid #e5e7eb;
    }}
    .section h2 {{
      margin: 0 0 12px 0;
      font-size: 18px;
    }}
    .section ul {{
      margin: 0;
      padding-left: 20px;
      line-height: 1.8;
    }}
    .section p {{
      margin: 0;
      line-height: 1.8;
    }}
    .note {{
      white-space: pre-wrap;
      line-height: 1.8;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .report-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .report-table th,
    .report-table td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 10px 8px;
      text-align: left;
      vertical-align: middle;
    }}
    .report-table th {{
      background: #f3f6fb;
      color: #475467;
      font-weight: 700;
    }}
    .mini-bar-row {{
      display: grid;
      grid-template-columns: 180px 1fr 84px;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }}
    .mini-bar-label {{
      font-weight: 700;
      color: #334155;
      font-size: 14px;
    }}
    .mini-bar-track {{
      width: 100%;
      height: 12px;
      background: #edf1f7;
      border-radius: 999px;
      overflow: hidden;
    }}
    .mini-bar-fill {{
      height: 100%;
      border-radius: 999px;
    }}
    .mini-bar-value {{
      text-align: right;
      font-weight: 700;
      color: #1f2937;
      font-size: 13px;
    }}
    @media print {{
      body {{
        background: white;
        padding: 0;
      }}
      .page {{
        max-width: none;
        border-radius: 0;
        box-shadow: none;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div class="header-top">
        <div class="brand">
          <div class="brand-mark">CAS</div>
          <div class="brand-copy">
            <div class="brand-title">기업 신용위험 분석 보고서</div>
            <div class="brand-subtitle">Corporate Analysis System 기반 AI 심사 메모 정리본입니다.</div>
          </div>
        </div>
        <div class="doc-badges">
          <div class="doc-chip">현재 데이터 기준</div>
          <div class="doc-chip">{escape(output_format_label)}</div>
        </div>
      </div>
      <div class="eyebrow">CREDIT RISK MEMO</div>
      <h1>{escape(str(selected_row.get("corp_name")))}</h1>
      <div class="summary">{escape(headline)}</div>
    </div>
    <div class="meta-grid">
      <div class="meta-card"><div class="meta-label">종목코드</div><div class="meta-value">{escape(_stock_code_text(selected_row.get("stock_code")))}</div></div>
      <div class="meta-card"><div class="meta-label">시장</div><div class="meta-value">{escape(to_market_label(selected_row.get("market")))}</div></div>
      <div class="meta-card"><div class="meta-label">산업</div><div class="meta-value">{escape(to_industry_label(selected_row.get("industry_macro_category")))}</div></div>
      <div class="meta-card"><div class="meta-label">규모</div><div class="meta-value">{escape(to_size_label(selected_row.get("firm_size_group")))}</div></div>
      <div class="meta-card"><div class="meta-label">회계연도</div><div class="meta-value">{escape(format_scalar(selected_row.get("fiscal_year")))}</div></div>
      <div class="meta-card"><div class="meta-label">사용 모델</div><div class="meta-value">{escape(model)}</div></div>
      <div class="meta-card"><div class="meta-label">출력 형식</div><div class="meta-value">{escape(output_format_label)}</div></div>
      <div class="meta-card"><div class="meta-label">투기등급 확률</div><div class="meta-value">{escape(probability)}</div></div>
      <div class="meta-card"><div class="meta-label">예측 라벨</div><div class="meta-value">{escape(predicted_label)}</div></div>
      <div class="meta-card"><div class="meta-label">위험 밴드</div><div class="meta-value">{escape(risk_band)}</div></div>
      <div class="meta-card"><div class="meta-label">판정 기준선</div><div class="meta-value">{escape(threshold)}</div></div>
    </div>
    <div class="body">
      <div class="section">
        <h2>핵심 위험 요인</h2>
        <ul>{_html_list(sections.get("핵심 위험 요인", []))}</ul>
      </div>
      <div class="section">
        <h2>완화 요인</h2>
        <ul>{_html_list(sections.get("완화 요인", []))}</ul>
      </div>
      <div class="section">
        <h2>주요 설명 변수 표</h2>
        {local_table_html}
      </div>
      <div class="section">
        <h2>주요 설명 변수 그래프</h2>
        {shap_chart_html}
      </div>
      <div class="section">
        <h2>동종업계 비교 표</h2>
        {peer_table_html}
      </div>
      <div class="section">
        <h2>동종업계 산업 내 위치</h2>
        {percentile_chart_html}
      </div>
      <div class="section">
        <h2>종합 의견</h2>
        <p>{escape(opinion or "종합 의견이 생성되지 않았습니다.")}</p>
      </div>
      <div class="section">
        <h2>AI 심사 메모 원문</h2>
        <div class="note">{escape(report_text.strip())}</div>
      </div>
    </div>
  </div>
</body>
</html>
"""


def build_onepage_html_report(
    *,
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    model: str,
    output_format_label: str,
    sections: dict[str, list[str]],
    local_shap: pd.DataFrame,
    peer_slice: pd.DataFrame,
    feature_map: pd.DataFrame,
) -> str:
    """Build a compact one-page HTML memo."""
    probability = (
        format_percent(prediction_row.get("prob_speculative"))
        if prediction_row is not None
        else "-"
    )
    predicted_label = (
        to_prediction_label(prediction_row.get("predicted_label"))
        if prediction_row is not None
        else "-"
    )
    risk_band = str(prediction_row.get("risk_band")) if prediction_row is not None else "-"
    threshold = (
        format_scalar(prediction_row.get("threshold")) if prediction_row is not None else "-"
    )
    headline = " ".join(sections.get("한줄 판단", [])).strip() or "심사 요약이 생성되지 않았습니다."
    top_local = local_shap.head(3).copy() if not local_shap.empty else pd.DataFrame()
    if not top_local.empty:
        top_local["표시명"] = top_local["feature"].map(
            lambda value: display_name(value, feature_map)
        )
        top_local["실제값"] = top_local.apply(
            lambda row: format_value_with_unit(
                row["feature_value"],
                get_feature_unit(str(row["feature"]), feature_map),
                str(row["feature"]),
            ),
            axis=1,
        )
    peer_summary = peer_slice.copy()
    if not peer_summary.empty:
        peer_summary["distance_from_industry_mid"] = (
            peer_summary["industry_percentile"] - 50.0
        ).abs()
        peer_summary = peer_summary.sort_values("distance_from_industry_mid", ascending=False).head(
            3
        )
        peer_summary["표시명"] = peer_summary["feature"].map(
            lambda value: display_name(value, feature_map)
        )
        peer_summary["산업 대비 차이"] = peer_summary.apply(
            lambda row: format_delta_with_unit(
                row["value"] - row["industry_median"],
                get_feature_unit(str(row["feature"]), feature_map),
                str(row["feature"]),
            ),
            axis=1,
        )

    peer_html = (
        "".join(
            f"<li>{escape(str(row['표시명']))}: 산업 대비 {escape(str(row['산업 대비 차이']))}</li>"
            for row in peer_summary.to_dict(orient="records")
        )
        or "<li>동종업계 비교 데이터가 없습니다.</li>"
    )
    local_table_html = _html_table_from_frame(
        top_local.rename(
            columns={
                "표시명": "지표",
                "실제값": "실제값",
            }
        ),
        ["지표", "실제값"],
    )
    peer_table_html = _html_table_from_frame(
        peer_summary.rename(
            columns={
                "표시명": "지표",
                "산업 대비 차이": "산업 대비 차이",
            }
        ),
        ["지표", "산업 대비 차이"],
    )
    local_chart_html = _html_shap_bar_rows(
        _prepare_local_driver_report_frame(local_shap, feature_map, top_n=3)
    )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>원페이지 심사 메모</title>
  <style>
    @page {{
      size: A4;
      margin: 16mm;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
      background: white;
      color: #1f2937;
      margin: 0;
      padding: 24px;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}
    .page {{
      max-width: 960px;
      margin: 0 auto;
      border: 1px solid #e5e7eb;
      border-radius: 18px;
      overflow: hidden;
    }}
    .header {{
      padding: 22px 24px;
      background: linear-gradient(135deg, #eef4ff 0%, #f8fafc 100%);
      border-bottom: 1px solid #e5e7eb;
    }}
    .header-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 14px;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .brand-mark {{
      width: 40px;
      height: 40px;
      border-radius: 12px;
      background: linear-gradient(135deg, #1d4ed8 0%, #60a5fa 100%);
      color: white;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.02em;
    }}
    .brand-copy {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .brand-title {{
      font-size: 13px;
      font-weight: 800;
      color: #0f172a;
    }}
    .brand-subtitle {{
      font-size: 11px;
      color: #64748b;
      line-height: 1.35;
    }}
    .doc-chip {{
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.82);
      border: 1px solid #d8dfeb;
      color: #475569;
      font-size: 11px;
      font-weight: 700;
      white-space: nowrap;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 24px;
    }}
    .headline {{
      line-height: 1.7;
      color: #374151;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      padding: 16px 24px 0 24px;
    }}
    .meta-card {{
      padding: 12px 14px;
      border-radius: 12px;
      background: #f7f8fb;
      border: 1px solid #e3e7ef;
    }}
    .meta-label {{
      font-size: 12px;
      font-weight: 700;
      color: #5c6473;
      margin-bottom: 4px;
    }}
    .meta-value {{
      font-size: 15px;
      font-weight: 700;
      color: #1f2937;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      padding: 16px 24px 24px 24px;
    }}
    .section {{
      border: 1px solid #e5e7eb;
      background: #fafafa;
      border-radius: 14px;
      padding: 16px;
    }}
    .section h2 {{
      margin: 0 0 10px 0;
      font-size: 17px;
    }}
    .section ul, .section p {{
      margin: 0;
      line-height: 1.75;
    }}
    .full {{
      grid-column: 1 / -1;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .report-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .report-table th,
    .report-table td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 8px 6px;
      text-align: left;
      vertical-align: middle;
    }}
    .report-table th {{
      background: #f3f6fb;
      color: #475467;
      font-weight: 700;
    }}
    .mini-bar-row {{
      display: grid;
      grid-template-columns: 140px 1fr 56px;
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
    }}
    .mini-bar-label {{
      font-weight: 700;
      color: #334155;
      font-size: 13px;
    }}
    .mini-bar-track {{
      width: 100%;
      height: 10px;
      background: #edf1f7;
      border-radius: 999px;
      overflow: hidden;
    }}
    .mini-bar-fill {{
      height: 100%;
      border-radius: 999px;
    }}
    .mini-bar-value {{
      text-align: right;
      font-weight: 700;
      color: #1f2937;
      font-size: 12px;
    }}
    @media print {{
      body {{
        padding: 0;
      }}
      .page {{
        max-width: none;
        border-radius: 0;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div class="header-top">
        <div class="brand">
          <div class="brand-mark">CAS</div>
          <div class="brand-copy">
            <div class="brand-title">원페이지 심사 메모</div>
            <div class="brand-subtitle">핵심 판단과 주요 근거를 한 장으로 정리한 버전입니다.</div>
          </div>
        </div>
        <div class="doc-chip">{escape(output_format_label)}</div>
      </div>
      <h1>{escape(str(selected_row.get("corp_name")))} 원페이지 심사 메모</h1>
      <div class="headline">{escape(headline)}</div>
    </div>
    <div class="meta">
      <div class="meta-card"><div class="meta-label">시장/산업</div><div class="meta-value">{escape(to_market_label(selected_row.get("market")))} / {escape(to_industry_label(selected_row.get("industry_macro_category")))}</div></div>
      <div class="meta-card"><div class="meta-label">규모/회계연도</div><div class="meta-value">{escape(to_size_label(selected_row.get("firm_size_group")))} / {escape(format_scalar(selected_row.get("fiscal_year")))}</div></div>
      <div class="meta-card"><div class="meta-label">투기등급 확률</div><div class="meta-value">{escape(probability)}</div></div>
      <div class="meta-card"><div class="meta-label">예측 라벨</div><div class="meta-value">{escape(predicted_label)} ({escape(risk_band)})</div></div>
      <div class="meta-card"><div class="meta-label">판정 기준선</div><div class="meta-value">{escape(threshold)}</div></div>
      <div class="meta-card"><div class="meta-label">종목코드</div><div class="meta-value">{escape(_stock_code_text(selected_row.get("stock_code")))}</div></div>
      <div class="meta-card"><div class="meta-label">사용 모델</div><div class="meta-value">{escape(model)}</div></div>
      <div class="meta-card"><div class="meta-label">출력 형식</div><div class="meta-value">{escape(output_format_label)}</div></div>
    </div>
    <div class="grid">
      <div class="section">
        <h2>핵심 위험 요인</h2>
        <ul>{_html_list(sections.get("핵심 위험 요인", []))}</ul>
      </div>
      <div class="section">
        <h2>완화 요인</h2>
        <ul>{_html_list(sections.get("완화 요인", []))}</ul>
      </div>
      <div class="section">
        <h2>주요 설명 변수</h2>
        <div style="margin-bottom:10px;">{local_table_html}</div>
        <div>{local_chart_html}</div>
      </div>
      <div class="section">
        <h2>동종업계 비교 핵심 차이</h2>
        <div style="margin-bottom:10px;">{peer_table_html}</div>
        <ul>{peer_html}</ul>
      </div>
      <div class="section full">
        <h2>종합 의견</h2>
        <p>{escape(" ".join(sections.get("종합 의견", [])).strip() or "종합 의견이 생성되지 않았습니다.")}</p>
      </div>
    </div>
  </div>
</body>
</html>
"""


def render_legend_card(
    container: st.delta_generator.DeltaGenerator,
    label: str,
    description: str,
    color: str,
) -> None:
    """Render a compact colored legend card for comparison views."""
    container.markdown(
        (
            f"<div style='min-height:96px;padding:0.85rem 1rem;border-radius:8px;"
            f"background:{COLOR_CARD_BG};border:1px solid {COLOR_CARD_BORDER};"
            "display:flex;flex-direction:column;justify-content:space-between;"
            "margin-bottom:0.35rem;'>"
            "<div style='display:flex;align-items:center;gap:0.5rem;'>"
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:999px;background:{color};'></span>"
            f"<span style='font-size:0.95rem;font-weight:700;color:{COLOR_CARD_LABEL};'>{escape(label)}</span>"
            "</div>"
            f"<div style='font-size:0.95rem;line-height:1.45;color:{COLOR_CARD_VALUE};'>{escape(description)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_accent_summary_card(
    container: st.delta_generator.DeltaGenerator,
    label: str,
    value: object,
    note: str,
    color: str,
) -> None:
    """Render a compact summary card with a colored accent for quick scanning."""
    container.markdown(
        (
            f"<div style='min-height:120px;padding:0.95rem 1rem;border-radius:8px;"
            f"background:{COLOR_CARD_BG};border:1px solid {COLOR_CARD_BORDER};"
            f"border-top:4px solid {color};box-shadow:{CARD_SHADOW};"
            "display:flex;flex-direction:column;justify-content:space-between;"
            "margin-bottom:0.5rem;'>"
            f"<div style='font-size:0.93rem;font-weight:700;color:{COLOR_CARD_LABEL};'>"
            f"{escape(label)}"
            "</div>"
            f"<div style='font-size:1.08rem;line-height:1.45;font-weight:700;color:{COLOR_CARD_VALUE};"
            "word-break:keep-all;'>"
            f"{escape(str(value))}"
            "</div>"
            f"<div style='font-size:0.88rem;line-height:1.45;color:{COLOR_CARD_LABEL};word-break:keep-all;'>"
            f"{escape(note)}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def build_probability_chart(probability: float, threshold: float) -> alt.Chart:
    """Create a simple chart comparing company probability and decision threshold."""
    frame = pd.DataFrame(
        [
            {"label": "기업 위험확률", "value": probability, "kind": "score"},
            {"label": "판정 기준선", "value": threshold, "kind": "threshold"},
        ]
    )
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("label:N", title=""),
            y=alt.Y("value:Q", title="비율", axis=alt.Axis(format="%")),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["score", "threshold"],
                    range=[COLOR_RISK, COLOR_MUTED],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("label:N", title="항목"),
                alt.Tooltip("value:Q", title="값", format=".2%"),
            ],
        )
        .properties(height=260)
    )
    return cast(alt.Chart, chart)


def approximate_percentile(series: pd.Series, new_value: float) -> float | None:
    """Approximate percentile rank if a scenario changes one variable."""
    clean = series.dropna()
    if clean.empty or pd.isna(new_value):
        return None
    augmented = pd.concat([clean, pd.Series([new_value])], ignore_index=True)
    return float(augmented.rank(method="average", pct=True).iloc[-1] * 100.0)


def build_llm_payload(
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    feature_map: pd.DataFrame,
    local_shap: pd.DataFrame,
    peer_slice: pd.DataFrame,
    industry_latest_row: pd.Series | None,
) -> dict[str, object]:
    """Build a concise payload for LLM explanation generation."""
    top_features = feature_map.sort_values("feature").head(0)
    if not local_shap.empty:
        top_shap = local_shap.head(5).copy()
        top_shap["display_name"] = top_shap["feature"].map(
            lambda value: display_name(value, feature_map)
        )
        top_shap["feature_value_display"] = top_shap.apply(
            lambda row: format_value_with_unit(
                row["feature_value"],
                get_feature_unit(str(row["feature"]), feature_map),
                str(row["feature"]),
            ),
            axis=1,
        )
        top_shap["shap_strength_display"] = top_shap["abs_shap"].map(
            lambda value: f"{float(value):.2f}"
        )
        top_shap["direction_korean"] = (
            top_shap["direction"]
            .map(
                {
                    "increase_risk": "위험 증가",
                    "decrease_risk": "위험 완화",
                }
            )
            .fillna("중립")
        )
        top_shap["interpretation_direction"] = top_shap["feature"].map(
            lambda value: get_feature_direction_label(str(value))
        )
        top_shap_records = (
            top_shap.loc[
                :,
                [
                    "display_name",
                    "feature",
                    "feature_value_display",
                    "shap_strength_display",
                    "direction_korean",
                    "interpretation_direction",
                ],
            ]
            .rename(
                columns={
                    "display_name": "korean_name",
                    "feature_value_display": "feature_value_display",
                    "shap_strength_display": "shap_strength_display",
                    "direction_korean": "direction",
                    "interpretation_direction": "interpretation_direction",
                }
            )
            .to_dict(orient="records")
        )
        driver_features = top_shap["feature"].tolist()
        top_features = feature_map.loc[feature_map["feature"].isin(driver_features)].copy()
    else:
        top_shap_records = []

    if top_features.empty:
        top_features = feature_map.head(5).copy()

    feature_records = top_features.loc[
        :,
        ["feature", "korean_name", "value", "unit", "description"],
    ].copy()
    if not feature_records.empty:
        feature_records["value_display"] = feature_records.apply(
            lambda row: format_value_with_unit(row["value"], row["unit"], str(row["feature"])),
            axis=1,
        )
        feature_records["interpretation_direction"] = feature_records["feature"].map(
            lambda value: get_feature_direction_label(str(value))
        )
    feature_records = feature_records.loc[
        :,
        ["feature", "korean_name", "value_display", "description", "interpretation_direction"],
    ].to_dict(orient="records")

    peer_records: list[dict[str, object]] = []
    if not peer_slice.empty:
        peer_slice = peer_slice.copy()
        peer_slice["distance_from_industry_mid"] = (peer_slice["industry_percentile"] - 50.0).abs()
        peer_slice = peer_slice.sort_values("distance_from_industry_mid", ascending=False).head(5)
        peer_slice["korean_name"] = peer_slice["feature"].map(
            lambda value: display_name(value, feature_map)
        )
        peer_slice["value_display"] = peer_slice.apply(
            lambda row: format_value_with_unit(
                row["value"],
                get_feature_unit(str(row["feature"]), feature_map),
                str(row["feature"]),
            ),
            axis=1,
        )
        peer_slice["industry_median_display"] = peer_slice.apply(
            lambda row: format_value_with_unit(
                row["industry_median"],
                get_feature_unit(str(row["feature"]), feature_map),
                str(row["feature"]),
            ),
            axis=1,
        )
        peer_slice["market_median_display"] = peer_slice.apply(
            lambda row: format_value_with_unit(
                row["market_median"],
                get_feature_unit(str(row["feature"]), feature_map),
                str(row["feature"]),
            ),
            axis=1,
        )
        peer_slice["industry_percentile_display"] = peer_slice["industry_percentile"].map(
            format_percentile_label
        )
        peer_slice["market_percentile_display"] = peer_slice["market_percentile"].map(
            format_percentile_label
        )
        peer_slice["industry_delta_display"] = peer_slice.apply(
            lambda row: format_delta_with_unit(
                row["value"] - row["industry_median"],
                get_feature_unit(str(row["feature"]), feature_map),
                str(row["feature"]),
            ),
            axis=1,
        )
        peer_slice["interpretation_direction"] = peer_slice["feature"].map(
            lambda value: get_feature_direction_label(str(value))
        )
        peer_records = peer_slice.loc[
            :,
            [
                "feature",
                "korean_name",
                "value_display",
                "industry_percentile_display",
                "market_percentile_display",
                "industry_median_display",
                "market_median_display",
                "industry_delta_display",
                "interpretation_direction",
            ],
        ].to_dict(orient="records")

    industry_context = None
    if industry_latest_row is not None:
        industry_context = {
            "market": to_market_label(industry_latest_row.get("market")),
            "industry_macro_category": to_industry_label(
                industry_latest_row.get("industry_macro_category")
            ),
            "companies_display": f"{format_scalar(industry_latest_row.get('companies'))}개사",
            "positive_rate_display": format_percent(industry_latest_row.get("positive_rate")),
            "mean_prob_speculative_display": format_percent(
                industry_latest_row.get("mean_prob_speculative")
            ),
            "pred_share_tuned_display": format_percent(industry_latest_row.get("pred_share_tuned")),
        }

    model_output = None
    if prediction_row is not None:
        model_output = {
            "prob_speculative_display": format_percent(prediction_row.get("prob_speculative")),
            "predicted_label": to_prediction_label(prediction_row.get("predicted_label")),
            "threshold_display": format_scalar(prediction_row.get("threshold")),
            "risk_band": prediction_row.get("risk_band"),
        }

    return {
        "company_profile": {
            "corp_name": selected_row.get("corp_name"),
            "stock_code": _stock_code_text(selected_row.get("stock_code")),
            "market": to_market_label(selected_row.get("market")),
            "industry_macro_category": to_industry_label(
                selected_row.get("industry_macro_category")
            ),
            "firm_size_group": to_size_label(selected_row.get("firm_size_group")),
            "fiscal_year": format_scalar(selected_row.get("fiscal_year")),
            "eval_year": format_scalar(selected_row.get("eval_year")),
        },
        "model_output": model_output,
        "key_metrics": feature_records,
        "top_shap": top_shap_records,
        "peer_context": peer_records,
        "industry_context": industry_context,
    }


def render_overview_tab(
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    model_summary: dict[str, object],
    feature_map: pd.DataFrame,
    artifacts: DashboardArtifacts,
) -> None:
    """Render the overview tab."""
    st.subheader("기업 개요")
    col1, col2, col3, col4 = st.columns(4)
    render_bold_value_block(col1, "기업명", str(selected_row["corp_name"]))
    render_bold_value_block(col2, "시장", to_market_label(selected_row["market"]))
    render_bold_value_block(
        col3, "산업", to_industry_label(selected_row["industry_macro_category"])
    )
    render_bold_value_block(col4, "규모", to_size_label(selected_row["firm_size_group"]))
    st.caption(
        "현재 선택한 기업의 기본 정보입니다. 시장, 산업, 규모를 먼저 확인한 뒤 아래의 위험 진단 결과를 함께 읽으면 됩니다."
    )

    st.subheader("모델 결과")
    st.caption(
        "투기등급 확률은 현재 기업이 투기등급으로 분류될 가능성을 뜻하며, 예측 라벨과 위험 밴드는 판정 기준선을 기준으로 함께 해석하면 됩니다."
    )
    if prediction_row is None:
        st.info(
            "현재 리포지토리 패키지에는 기업별 예측확률 파일이 포함되어 있지 않습니다. "
            "아래에는 현재 XGBoost 모델의 전체 test 성능과 선택 기업의 핵심 지표를 함께 표시합니다."
        )
        raw_test_overall_models = model_summary.get("test_overall_models", [])
        test_overall_models = (
            raw_test_overall_models if isinstance(raw_test_overall_models, list) else []
        )
        selected_model_name = model_summary.get("selected_model")
        xgboost_rows = [
            row
            for row in test_overall_models
            if isinstance(row, dict) and row.get("model") == selected_model_name
        ]
        selected_model = xgboost_rows[0] if xgboost_rows else None
        raw_threshold_rows = model_summary.get("xgboost_thresholds", [])
        threshold_rows = raw_threshold_rows if isinstance(raw_threshold_rows, list) else []
        default_threshold = next(
            (
                row
                for row in threshold_rows
                if isinstance(row, dict) and row.get("threshold_type") == "default_0_5"
            ),
            None,
        )
        c1, c2, c3, c4 = st.columns(4)
        render_bold_value_block(
            c1, "모델 PR-AUC", format_scalar(selected_model["pr_auc"] if selected_model else None)
        )
        render_bold_value_block(
            c2,
            "모델 Precision@0.5",
            format_scalar(selected_model["precision_at_0_5"] if selected_model else None),
        )
        render_bold_value_block(
            c3,
            "모델 Recall@0.5",
            format_scalar(selected_model["recall_at_0_5"] if selected_model else None),
        )
        render_bold_value_block(
            c4,
            "모델 Threshold",
            format_scalar(default_threshold["threshold"] if default_threshold else None),
        )
        st.caption(
            "위 수치는 선택 기업의 개별 점수가 아니라, 현재 XGBoost 모델의 전체 테스트 성능을 보여줍니다."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        render_bold_value_block(
            c1, "투기등급 확률", format_percent(prediction_row.get("prob_speculative"))
        )
        render_bold_value_block(
            c2, "예측 라벨", to_prediction_label(prediction_row.get("predicted_label"))
        )
        render_bold_value_block(c3, "판정 기준선", format_scalar(prediction_row.get("threshold")))
        render_badge_value_block(
            c4, "위험 밴드", render_risk_band_badge(prediction_row.get("risk_band"))
        )
        st.caption(
            "기업별 점수는 대시보드 산출물 export 단계에서 현재 XGBoost 학습 레시피를 "
            "재현하여 생성한 값입니다."
        )
        risk_band = str(prediction_row.get("risk_band"))
        probability_text = format_percent(prediction_row["prob_speculative"])
        threshold_text = format_scalar(prediction_row["threshold"])
        label_text = to_prediction_label(prediction_row.get("predicted_label"))
        if risk_band == "고위험":
            summary_text = (
                f"투기등급 확률은 {probability_text}로, 판정 기준선 {threshold_text}를 웃돕니다. "
                f"현재 판단은 {label_text}이며, 주요 위험 요인을 우선 점검할 필요가 있습니다."
            )
            summary_color = COLOR_RISK
        elif risk_band == "관찰":
            summary_text = (
                f"투기등급 확률은 {probability_text}로, 판정 기준선 {threshold_text} 부근에 있습니다. "
                f"현재 판단은 {label_text}이며, 동종업계 대비 취약 지표를 함께 볼 필요가 있습니다."
            )
            summary_color = "#c0841a"
        else:
            summary_text = (
                f"투기등급 확률은 {probability_text}로, 판정 기준선 {threshold_text} 대비 안정적인 수준입니다. "
                f"다만 수익성과 유동성 지표의 변화는 함께 점검할 필요가 있습니다."
            )
            summary_color = COLOR_MITIGATE
        render_summary_banner("한눈에 보기", summary_text, summary_color)
        chart_col, text_col = st.columns([1.2, 0.8])
        with chart_col:
            probability_chart = build_probability_chart(
                float(prediction_row["prob_speculative"]),
                float(prediction_row["threshold"]),
            )
            stretch_altair_chart(probability_chart)
        with text_col:
            st.markdown("**심사 메모**")
            st.markdown(
                f"현재 투기등급 확률은 **{format_percent(prediction_row['prob_speculative'])}** 입니다."
            )
            st.markdown(
                f"현재 판단은 **{label_text}**이며, 위험 밴드는 **{risk_band}** 구간입니다."
            )

    st.subheader("핵심 지표")
    st.caption(
        "현재 기업을 볼 때 먼저 확인하는 핵심 지표입니다. 각 카드에는 현재 값, 지표 설명, 그리고 일반적인 해석 방향이 함께 표시됩니다."
    )
    intro_col1, intro_col2, intro_col3 = st.columns(3)
    render_text_card(
        intro_col1,
        "무엇을 보나요?",
        "유동성, 수익성, 자본건전성, 외부 금융환경처럼 현재 위험 판단에 직접 연결되는 지표를 먼저 보여줍니다.",
    )
    render_text_card(
        intro_col2,
        "어떻게 읽나요?",
        "각 카드의 방향 배지는 일반적으로 높을수록 좋은지, 낮을수록 좋은지를 빠르게 이해할 수 있도록 돕습니다.",
    )
    render_text_card(
        intro_col3,
        "산업 안에서는 어떤가요?",
        "아래 그래프에서는 같은 산업 안에서 현재 기업이 어느 정도 수준에 있는지도 함께 확인할 수 있습니다.",
    )
    overview_features = [
        "cash_ratio",
        "interest_coverage_ratio",
        "capital_impairment_ratio",
        "net_margin",
        "gross_profit",
        "spec_spread",
    ]
    overview_frame = feature_map.loc[feature_map["feature"].isin(overview_features)].copy()
    overview_frame["값"] = overview_frame.apply(
        lambda row: format_value_with_unit(row["value"], row["unit"], str(row["feature"])),
        axis=1,
    )
    overview_frame["일반 해석 방향"] = overview_frame["feature"].map(
        lambda value: get_feature_direction_label(str(value))
    )
    overview_frame = overview_frame.sort_values("korean_name")
    metric_cards = st.columns(3)
    for index, row in enumerate(overview_frame.to_dict(orient="records")):
        render_value_detail_block(
            metric_cards[index % 3],
            str(row["korean_name"]),
            row["값"],
            str(row["description"]),
            render_direction_badge_html(row["일반 해석 방향"]),
        )

    if artifacts.peer_percentiles is not None:
        peer_slice = artifacts.peer_percentiles.loc[
            (
                artifacts.peer_percentiles["stock_code"].map(_stock_code_text)
                == _stock_code_text(selected_row["stock_code"])
            )
            & (artifacts.peer_percentiles["fiscal_year"] == selected_row["fiscal_year"])
            & (artifacts.peer_percentiles["feature"].isin(overview_features))
        ].copy()
        if not peer_slice.empty:
            peer_slice["표시명"] = peer_slice["feature"].map(
                lambda value: display_name(value, feature_map)
            )
            peer_slice["실제값_표시"] = peer_slice.apply(
                lambda row: format_value_with_unit(
                    row["value"],
                    get_feature_unit(str(row["feature"]), feature_map),
                    str(row["feature"]),
                ),
                axis=1,
            )
            percentile_chart = (
                alt.Chart(peer_slice)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X(
                        "industry_percentile:Q",
                        title="산업 내 백분위",
                        scale=alt.Scale(domain=[0, 100]),
                    ),
                    y=alt.Y("표시명:N", sort="-x", title=""),
                    color=alt.value(COLOR_NEUTRAL),
                    tooltip=[
                        alt.Tooltip("표시명:N", title="변수"),
                        alt.Tooltip("industry_percentile:Q", title="산업 백분위", format=".2f"),
                        alt.Tooltip("실제값_표시:N", title="실제값"),
                    ],
                )
                .properties(height=260)
            )
            st.markdown("**핵심 지표가 산업 안에서 어느 수준인지**")
            stretch_altair_chart(percentile_chart)


def render_committee_view_tab(
    *,
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    feature_map: pd.DataFrame,
    local_shap: pd.DataFrame,
    peer_slice: pd.DataFrame,
    developer_mode: bool,
) -> None:
    """Render the Stage 1 model_view and Stage 2 committee_view side by side."""
    st.subheader("외부 근거 중심 위원회 검토")
    st.caption(
        "모델 점수는 다른 탭에서도 충분히 확인할 수 있으므로, 이 탭에서는 뉴스·웹·공시 근거가 "
        "2차 위원회 판단에 어떤 보완 정보를 주는지 더 자세히 보여줍니다."
    )
    selected_output_format = st.selectbox(
        "위원회 검토 출력 방식",
        options=list(LLM_OUTPUT_FORMATS.keys()),
        format_func=lambda value: LLM_OUTPUT_FORMATS.get(value, value),
        index=1,
        key="committee_output_format",
        help="같은 위원회 판단 결과를 짧게, 기본 메모형, 또는 상세 보고서형으로 나누어 볼 수 있습니다.",
    )
    output_format_label = LLM_OUTPUT_FORMATS.get(selected_output_format, selected_output_format)
    format_description = OUTPUT_FORMAT_DESCRIPTIONS.get(
        selected_output_format, "선택한 형식에 맞춰 위원회 판단 결과를 보여줍니다."
    )
    intro_col1, intro_col2, intro_col3 = st.columns(3)
    render_text_card(
        intro_col1,
        "무엇을 자세히 보나요?",
        f"현재는 {output_format_label} 형식입니다. {format_description} 외부 근거의 관련성과 품질을 중심으로 봅니다.",
    )
    render_text_card(
        intro_col2,
        "어떻게 걸러내나요?",
        "기업명·종목코드 직접 관련성, 출처 신뢰도, 위험 키워드가 실제 문맥에서 확인되는지를 나누어 봅니다.",
    )
    render_text_card(
        intro_col3,
        "모델과의 관계",
        "외부 근거는 1차 모델 판단을 덮어쓰는 용도가 아니라, 보류·주의가 필요한 근거를 보완하는 역할입니다.",
    )
    evidence_snapshot = resolve_dashboard_external_evidence(selected_row)

    try:
        committee_context = build_dashboard_committee_context(
            selected_row=selected_row,
            prediction_row=prediction_row,
            local_shap=local_shap,
            peer_slice=peer_slice,
            external_evidence_snapshot=evidence_snapshot,
        )
    except Exception as error:
        LOGGER.exception("dashboard_stage2_committee_context_failed")
        st.error("2차 에이전트 위원회 판단을 생성하는 중 문제가 발생했습니다.")
        st.caption(f"오류 상세: {format_stage2_error_detail(error)}")
        if developer_mode:
            st.exception(error)
        return

    if committee_context is None:
        st.info(
            "선택 기업의 예측확률 파일이 없어 2차 에이전트 위원회 판단을 생성할 수 없습니다. "
            "기업별 prediction_scores.csv가 연결되면 이 탭에서 1차 모델 판단과 2차 위원회 판단을 함께 볼 수 있습니다."
        )
        return

    model_view = _as_plain_dict(committee_context.get("model_view"))
    committee_view = _as_plain_dict(committee_context.get("committee_view"))
    agent_summary = _as_plain_dict(committee_context.get("agent_summary"))
    rule_result = _as_plain_dict(committee_context.get("rule_result"))
    st.session_state["model_view"] = model_view
    st.session_state["committee_view"] = committee_view

    model_label = str(model_view.get("prediction_label") or "-")
    model_display_label = "투기등급(부적격)" if model_label == "부적격" else model_label
    committee_label = str(committee_view.get("final_committee_label") or "보류")
    model_base_label = to_committee_base_label(model_label)
    has_decision_gap = model_base_label != committee_label
    veto_triggered = bool(committee_view.get("veto_triggered", False))
    decision_gap_label = "차이 있음" if has_decision_gap else "일치"
    veto_label = dashboard_veto_status_label(committee_view, evidence_snapshot)
    evidence_panel_colors = EvidencePanelColors(
        risk=COLOR_RISK,
        mitigate=COLOR_MITIGATE,
        neutral=COLOR_NEUTRAL,
    )
    evidence_panel_renderers = EvidencePanelRenderers(
        render_badge_value_block=render_badge_value_block,
        render_bold_value_block=render_bold_value_block,
        render_decision_badge=render_decision_badge,
        render_list_card=render_list_card,
        render_summary_banner=render_summary_banner,
        render_text_card=render_text_card,
    )

    if veto_triggered:
        summary_text = (
            "강제 경고가 발동되었습니다. 강한 외부 위험 신호나 차단 규칙이 감지되어, "
            "위원회 의견은 모델 원판단보다 보수적으로 제시됩니다."
        )
        summary_color = COLOR_RISK
    elif veto_label == "후보 검토":
        summary_text = (
            "강제 경고 후보가 감지되었지만, 최종 발동 기준인 다중 출처·고신뢰 근거 조건은 "
            "아직 충족하지 않았습니다. 따라서 화면에서는 후보 상태로 분리해 표시합니다."
        )
        summary_color = "#c0841a"
    elif has_decision_gap:
        summary_text = (
            f"모델 기준 위원회 대응 라벨은 {model_base_label}이지만, "
            f"2차 위원회는 {committee_label}로 정리했습니다. 아래 판단 차이 설명에서 "
            "왜 차이가 났는지 확인할 수 있습니다."
        )
        summary_color = "#c0841a"
    else:
        summary_text = (
            f"1차 모델 판단과 2차 에이전트 위원회 판단이 {committee_label} 방향으로 일치합니다. "
            "위원회 메모는 모델 판단의 근거와 보완 요인을 설명하는 역할을 합니다."
        )
        summary_color = COLOR_MITIGATE

    render_external_evidence_judgment(
        evidence_snapshot,
        committee_view,
        veto_label=veto_label,
        colors=evidence_panel_colors,
        renderers=evidence_panel_renderers,
    )
    render_external_evidence_items(
        evidence_snapshot,
        expanded=True,
        include_summary=selected_output_format == "detailed",
        renderers=evidence_panel_renderers,
    )

    with st.expander("1차 모델 판단과 비교해서 보기", expanded=False):
        comparison_cols = st.columns(4)
        render_badge_value_block(
            comparison_cols[0],
            "1차 모델 판단",
            render_decision_badge(model_display_label),
        )
        render_badge_value_block(
            comparison_cols[1],
            "2차 위원회 판단",
            render_decision_badge(committee_label),
        )
        render_badge_value_block(
            comparison_cols[2],
            "판단 차이",
            render_decision_badge(decision_gap_label),
        )
        render_badge_value_block(
            comparison_cols[3],
            "강제 경고 상태",
            render_decision_badge(veto_label),
        )

        model_metric_cols = st.columns(3)
        render_bold_value_block(
            model_metric_cols[0],
            "XGBoost 투기등급 확률",
            format_percent(model_view.get("probability_speculative")),
        )
        render_badge_value_block(
            model_metric_cols[1],
            "1차 모델 위험 밴드",
            render_risk_band_badge(format_stage2_risk_band(model_view.get("risk_band"))),
        )
        render_bold_value_block(
            model_metric_cols[2],
            "위원회 신뢰도",
            format_percent(committee_context.get("final_confidence")),
        )
        trigger_cols = st.columns(4)
        secondary_triggered = bool(model_view.get("stage2_secondary_trigger", False))
        review_triggered = bool(model_view.get("stage2_review_trigger", False))
        overwarning_candidate = bool(model_view.get("stage2_overwarning_filter_candidate", False))
        trigger_status = (
            "추가 검토" if secondary_triggered else "1차 위험 검토" if review_triggered else "일반"
        )
        overwarning_status = "완화 검토" if overwarning_candidate else "특이 없음"
        render_badge_value_block(
            trigger_cols[0],
            "2차 검토 트리거",
            render_decision_badge(trigger_status),
        )
        render_bold_value_block(
            trigger_cols[1],
            "45개 보조 변수셋 확률",
            format_percent(model_view.get("probability_speculative_45")),
        )
        render_badge_value_block(
            trigger_cols[2],
            "과민 경고 보조필터",
            render_decision_badge(overwarning_status),
        )
        render_text_card(
            trigger_cols[3],
            "검토 사유",
            str(
                model_view.get("overwarning_filter_reason")
                if overwarning_candidate
                else model_view.get("trigger_reason") or "추가 위원회 검토 트리거 없음"
            ),
        )
        render_summary_banner("판단 차이 해석", summary_text, summary_color)

    full_risk_items = _friendly_committee_items(
        _as_text_list(committee_view.get("key_risk_factors")),
        feature_map,
    )
    full_mitigation_items = _friendly_committee_items(
        _as_text_list(committee_view.get("mitigating_factors")),
        feature_map,
    )
    highlight_risk_items = (
        full_risk_items[:2] if selected_output_format == "brief" else full_risk_items
    )
    highlight_mitigation_items = (
        full_mitigation_items[:2] if selected_output_format == "brief" else full_mitigation_items
    )
    conflict_text = _friendly_committee_text(
        committee_view.get("conflict_resolution")
        or "1차 모델 판단과 2차 위원회 판단이 크게 다르지 않습니다.",
        feature_map,
    )
    final_memo = _friendly_committee_text(
        committee_view.get("final_review_memo")
        or "현재 선택한 기업에 대해 추가로 표시할 검토 의견이 없습니다.",
        feature_map,
    )

    st.subheader("2차 위원회 핵심 판단")
    st.caption(
        "먼저 결론과 가장 중요한 위험·완화 요인을 짧게 확인하고, 아래에서 같은 내용을 상세 문장으로 이어서 볼 수 있습니다."
    )
    render_committee_key_highlights(
        committee_label=committee_label,
        model_display_label=model_display_label,
        decision_gap_label=decision_gap_label,
        veto_label=veto_label,
        final_confidence=committee_context.get("final_confidence"),
        summary_text=summary_text,
        conflict_text=conflict_text,
        final_memo=final_memo,
        risk_items=highlight_risk_items,
        mitigation_items=highlight_mitigation_items,
    )
    render_committee_full_review(
        summary_text=summary_text,
        conflict_text=conflict_text,
        final_memo=final_memo,
        risk_items=full_risk_items,
        mitigation_items=full_mitigation_items,
    )

    if selected_output_format in {"memo", "detailed"}:
        st.subheader("판단에 참고한 근거")
        evidence_frame = _committee_evidence_frame(committee_view.get("evidence_summary"))
        if evidence_frame.empty:
            st.info("아직 화면에 보여줄 근거 요약이 없습니다.")
        else:
            stretch_dataframe(evidence_frame, hide_index=True)

    if selected_output_format == "detailed":
        st.subheader("더 자세히 보기")
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        render_list_card(
            detail_col1,
            "판단 규칙에서 확인한 점",
            _as_text_list(rule_result.get("reasons")),
            COLOR_NEUTRAL,
        )
        render_list_card(
            detail_col2,
            "즉시 주의가 필요한 신호",
            _as_text_list(rule_result.get("blocking_flags")),
            COLOR_RISK,
        )
        render_bold_value_block(
            detail_col3,
            "외부 근거 주의 후보",
            f"{external_veto_candidate_count(evidence_snapshot)}건",
        )
        top_drivers = model_view.get("top_drivers")
        if isinstance(top_drivers, list | tuple) and top_drivers:
            driver_frame = pd.DataFrame(top_drivers).rename(
                columns={
                    "name": "변수",
                    "value": "SHAP 값",
                    "abs_value": "|SHAP|",
                    "feature_value": "실제값",
                }
            )
            stretch_dataframe(driver_frame, hide_index=True)

    if selected_output_format in {"memo", "detailed"}:
        with st.expander(
            "에이전트별로 어떻게 봤는지 보기",
            expanded=selected_output_format == "detailed",
        ):
            agents = _as_plain_dict(agent_summary.get("agents"))
            if not agents:
                st.info("에이전트별 요약이 아직 생성되지 않았습니다.")

            role_env_mapping = {
                "quant_credit": "QUANT_AGENT_MODEL",
                "evidence_audit": "RESEARCH_AGENT_MODEL",
                "chair_report": "MANAGER_AGENT_MODEL",
            }
            for role, raw_agent in agents.items():
                agent = _as_plain_dict(raw_agent)
                role_label = STAGE2_AGENT_ROLE_LABELS.get(str(role), str(role))

                env_key = role_env_mapping.get(str(role))
                used_model = os.environ.get(env_key, "기본 모델") if env_key else "기본 모델"

                st.markdown(
                    f"**{role_label}** "
                    "<span style='"
                    "background-color:var(--cas-neutral-soft); "
                    "color:var(--cas-text); "
                    "border:1px solid var(--cas-neutral-border); "
                    "padding:0.2rem 0.5rem; "
                    "border-radius:0.4rem; "
                    "font-size:0.75rem; "
                    "margin-left:0.5rem;'"
                    ">"
                    f"사용 모델: {used_model}</span>",
                    unsafe_allow_html=True,
                )
                st.write(str(agent.get("summary") or "요약이 없습니다."))
                findings = _as_text_list(agent.get("findings"))
                if findings:
                    st.markdown("\n".join(f"- {item}" for item in findings))
                st.caption(f"검토 신뢰도: {format_percent(agent.get('confidence'))}")

    if developer_mode:
        with st.expander("개발자용 전체 위원회 판단 JSON", expanded=False):
            st.json(committee_context)
        with st.expander("개발자용 규칙 기반 판단 JSON", expanded=False):
            st.json(rule_result)


def render_llm_panel(
    *,
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    feature_map: pd.DataFrame,
    local_shap: pd.DataFrame,
    peer_slice: pd.DataFrame,
    industry_latest_row: pd.Series | None,
    provider: str,
    api_key: str,
    model: str,
    developer_mode: bool,
) -> None:
    """Render an optional LLM explanation section."""
    provider_label = LLM_PROVIDER_LABELS.get(provider, provider)
    st.subheader("AI 심사 메모")
    st.caption("선택 기업의 점수와 비교 결과를 바탕으로, 바로 읽을 수 있는 심사 메모를 생성합니다.")
    intro_col1, intro_col2, intro_col3 = st.columns(3)
    selected_output_format = st.selectbox(
        "출력 형식",
        options=list(LLM_OUTPUT_FORMATS.keys()),
        format_func=lambda value: LLM_OUTPUT_FORMATS.get(value, value),
        index=1,
        help="같은 근거 데이터를 바탕으로 더 짧게, 기본 심사메모형, 또는 조금 더 자세한 보고서형으로 요약할 수 있습니다.",
    )
    output_format_label = LLM_OUTPUT_FORMATS.get(selected_output_format, selected_output_format)
    format_description = {
        "brief": "핵심만 빠르게 읽을 수 있는 짧은 요약 형식입니다.",
        "memo": "가장 균형 잡힌 기본 심사 메모 형식입니다.",
        "detailed": "숫자와 비교 맥락을 조금 더 살린 상세 보고서형입니다.",
    }.get(selected_output_format, "선택한 형식에 맞춰 요약합니다.")
    render_text_card(
        intro_col1,
        "어떤 형식인가요?",
        f"현재는 {output_format_label} 형식으로 보여줍니다. {format_description}",
    )
    render_text_card(
        intro_col2,
        "무엇을 참고하나요?",
        "예측확률, 핵심 지표, SHAP, 동종업계 비교 결과를 함께 참고합니다.",
    )
    render_text_card(
        intro_col3,
        "어떤 모델을 쓰나요?",
        f"현재 선택 모델은 {provider_label}의 {model}이며, API 키를 입력하면 바로 메모를 생성할 수 있습니다.",
    )
    if not api_key.strip():
        st.info(
            f"사이드바의 `AI 메모 설정`에서 {provider_label} API 키를 입력하면 AI 심사 메모를 생성할 수 있습니다."
        )

    payload = build_llm_payload(
        selected_row,
        prediction_row,
        feature_map,
        local_shap,
        peer_slice,
        industry_latest_row,
    )

    cache_key = (
        f"{_stock_code_text(selected_row['stock_code'])}-{selected_row['fiscal_year']}-"
        f"{provider}-{model}-{selected_output_format}"
    )
    if st.button("AI 심사 메모 생성", type="primary"):
        if not api_key.strip():
            st.warning(f"{provider_label} API 키를 입력해야 AI 심사 메모를 생성할 수 있습니다.")
        else:
            try:
                with st.spinner("AI가 심사 메모를 정리하는 중입니다..."):
                    explanation = generate_llm_explanation(
                        provider=provider,
                        api_key=api_key.strip(),
                        model=model.strip(),
                        payload=payload,
                        output_format=selected_output_format,
                    )
                st.session_state[cache_key] = explanation
            except Exception as error:  # pragma: no cover - runtime/network dependent
                st.error(format_llm_error_message(error, provider_label))

    cached = st.session_state.get(cache_key)
    if cached:
        st.success("AI 심사 메모 생성 완료")
        sections = parse_llm_report_sections(cached)
        headline = " ".join(sections["한줄 판단"]).strip() or cached.splitlines()[0].strip()
        render_summary_banner("AI 한줄 판단", headline, COLOR_NEUTRAL)

        risk_badge_items: list[tuple[str, str]] = []
        mitigate_badge_items: list[tuple[str, str]] = []
        if not local_shap.empty:
            shap_view = local_shap.copy().sort_values("abs_shap", ascending=False)
            top_risk_features = (
                shap_view.loc[shap_view["shap_value"] > 0, "feature"].head(3).tolist()
            )
            top_mitigate_features = (
                shap_view.loc[shap_view["shap_value"] < 0, "feature"].head(3).tolist()
            )
            risk_badge_items = [
                (display_name(feature, feature_map), get_feature_direction_label(str(feature)))
                for feature in top_risk_features
            ]
            mitigate_badge_items = [
                (display_name(feature, feature_map), get_feature_direction_label(str(feature)))
                for feature in top_mitigate_features
            ]

        report_col1, report_col2 = st.columns(2)
        render_list_card(report_col1, "핵심 위험 요인", sections["핵심 위험 요인"], COLOR_RISK)
        render_list_card(report_col2, "완화 요인", sections["완화 요인"], COLOR_MITIGATE)
        render_badge_hint_card(
            report_col1,
            "관련 지표 방향",
            risk_badge_items,
            COLOR_RISK,
            "연결할 대표 위험 지표가 없습니다.",
        )
        render_badge_hint_card(
            report_col2,
            "관련 지표 방향",
            mitigate_badge_items,
            COLOR_MITIGATE,
            "연결할 대표 완화 지표가 없습니다.",
        )

        opinion_text = " ".join(sections["종합 의견"]).strip()
        if opinion_text:
            render_text_card(st.container(), "종합 의견", opinion_text)
        export_text = build_exportable_llm_report(
            selected_row=selected_row,
            prediction_row=prediction_row,
            model=f"{provider_label} · {model}",
            output_format_label=output_format_label,
            report_text=cached,
            local_shap=local_shap,
            peer_slice=peer_slice,
            feature_map=feature_map,
        )
        onepage_text = build_onepage_llm_report(
            selected_row=selected_row,
            prediction_row=prediction_row,
            model=f"{provider_label} · {model}",
            output_format_label=output_format_label,
            sections=sections,
            local_shap=local_shap,
            peer_slice=peer_slice,
            feature_map=feature_map,
        )
        html_report = build_html_report(
            selected_row=selected_row,
            prediction_row=prediction_row,
            model=f"{provider_label} · {model}",
            output_format_label=output_format_label,
            sections=sections,
            report_text=cached,
            local_shap=local_shap,
            peer_slice=peer_slice,
            feature_map=feature_map,
        )
        onepage_html = build_onepage_html_report(
            selected_row=selected_row,
            prediction_row=prediction_row,
            model=f"{provider_label} · {model}",
            output_format_label=output_format_label,
            sections=sections,
            local_shap=local_shap,
            peer_slice=peer_slice,
            feature_map=feature_map,
        )
        html_col1, html_col2 = st.columns([1, 1])
        with html_col1:
            stretch_download_button(
                label="보고서형 HTML 다운로드",
                data=html_report,
                file_name=(
                    f"credit_report_{_stock_code_text(selected_row['stock_code'])}_"
                    f"{selected_row['fiscal_year']}.html"
                ),
                mime="text/html",
            )
        with html_col2:
            stretch_download_button(
                label="원페이지 HTML 다운로드",
                data=onepage_html,
                file_name=(
                    f"credit_onepage_{_stock_code_text(selected_row['stock_code'])}_"
                    f"{selected_row['fiscal_year']}.html"
                ),
                mime="text/html",
            )
        utility_col1, utility_col2 = st.columns([1, 1])
        with utility_col1:
            stretch_download_button(
                label="상세 보고서형 다운로드 (.md)",
                data=export_text,
                file_name=(
                    f"credit_report_{_stock_code_text(selected_row['stock_code'])}_"
                    f"{selected_row['fiscal_year']}.md"
                ),
                mime="text/markdown",
            )
        with utility_col2:
            stretch_download_button(
                label="원페이지 요약 다운로드 (.md)",
                data=onepage_text,
                file_name=(
                    f"credit_onepage_{_stock_code_text(selected_row['stock_code'])}_"
                    f"{selected_row['fiscal_year']}.md"
                ),
                mime="text/markdown",
            )
        preview_tab1, preview_tab2, preview_tab3, preview_tab4 = st.tabs(
            ["보고서형 HTML", "원페이지 HTML", "보고서형 미리보기", "원페이지 미리보기"]
        )
        with preview_tab1:
            st.components.v1.html(html_report, height=720, scrolling=True)
        with preview_tab2:
            st.components.v1.html(onepage_html, height=720, scrolling=True)
        with preview_tab3:
            st.text_area(
                "복사용 보고서형 메모",
                value=export_text,
                height=180,
                help="상세 보고서 버전을 그대로 복사해 문서나 메신저에 붙여넣을 수 있습니다.",
            )
        with preview_tab4:
            st.text_area(
                "복사용 원페이지 메모",
                value=onepage_text,
                height=180,
                help="한 장 요약본을 그대로 복사해 발표자료나 요약 메모에 붙여넣을 수 있습니다.",
            )
        with st.expander("원문 보기"):
            st.markdown(
                (
                    f"<div style='padding:1rem 1.05rem;border-radius:8px;"
                    f"background:{COLOR_CARD_BG};border:1px solid {COLOR_CARD_BORDER};"
                    f"border-left:6px solid {COLOR_NEUTRAL};box-shadow:{CARD_SHADOW};"
                    "margin-top:0.25rem;'>"
                    f"<div style='font-size:0.95rem;font-weight:700;color:{COLOR_CARD_LABEL};margin-bottom:0.45rem;'>"
                    "AI 심사 메모 원문"
                    "</div>"
                    f"<div style='font-size:0.98rem;line-height:1.75;color:{COLOR_CARD_VALUE};white-space:pre-wrap;'>"
                    f"{escape(cached)}"
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    if developer_mode:
        with st.expander("AI 입력 payload 보기"):
            st.json(payload)


def render_drivers_tab(
    selected_row: pd.Series,
    artifacts: DashboardArtifacts,
) -> None:
    """Render the drivers tab."""
    st.subheader("주요 영향 요인")
    intro_col1, intro_col2, intro_col3 = st.columns(3)
    render_text_card(
        intro_col1,
        "어떻게 읽나요?",
        "이 기업을 위험하게 보게 만든 요인과 안정적으로 보게 만든 요인을 함께 보여줍니다.",
    )
    render_text_card(
        intro_col2,
        "위험을 높이는 요인",
        "값이 클수록 이 기업을 더 위험하게 보게 만드는 지표를 뜻합니다.",
    )
    render_text_card(
        intro_col3,
        "위험을 낮추는 요인",
        "값이 클수록 위험을 낮추거나 완화하는 쪽으로 작용한 지표를 뜻합니다.",
    )
    if artifacts.local_shap is not None:
        matched = resolve_company_local_shap(selected_row, artifacts.local_shap)
        if not matched.empty:
            st.success("선택 기업 기준 주요 영향 요인을 보여주고 있습니다.")
            local_view = matched.sort_values("abs_shap", ascending=False).head(10)
            feature_map = build_company_feature_map(selected_row, artifacts.feature_dictionary)
            local_view["표시명"] = local_view["feature"].map(
                lambda value: display_name(value, feature_map)
            )
            local_view["영향방향"] = local_view["shap_value"].map(
                lambda value: "위험 증가" if value > 0 else "위험 완화"
            )
            local_view["실제값"] = local_view.apply(
                lambda row: format_value_with_unit(
                    row["feature_value"],
                    get_feature_unit(str(row["feature"]), feature_map),
                    str(row["feature"]),
                ),
                axis=1,
            )
            local_view["일반 해석 방향"] = local_view["feature"].map(
                lambda value: get_feature_direction_label(str(value))
            )
            top_risk = local_view.loc[local_view["shap_value"] > 0].head(1)
            top_mitigate = local_view.loc[local_view["shap_value"] < 0].head(1)
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            render_accent_summary_card(
                summary_col1,
                "가장 큰 위험 요인",
                top_risk.iloc[0]["표시명"] if not top_risk.empty else "없음",
                top_risk.iloc[0]["실제값"]
                if not top_risk.empty
                else "위험 증가 요인이 뚜렷하지 않습니다.",
                COLOR_RISK,
            )
            render_accent_summary_card(
                summary_col2,
                "가장 큰 완화 요인",
                top_mitigate.iloc[0]["표시명"] if not top_mitigate.empty else "없음",
                top_mitigate.iloc[0]["실제값"]
                if not top_mitigate.empty
                else "완화 요인이 뚜렷하지 않습니다.",
                COLOR_MITIGATE,
            )
            render_accent_summary_card(
                summary_col3,
                "상위 SHAP 강도",
                format_scalar(local_view["abs_shap"].head(5).mean()),
                "상위 5개 설명 변수의 평균 |SHAP| 수준입니다.",
                COLOR_NEUTRAL,
            )
            chart = (
                alt.Chart(local_view)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("shap_value:Q", title="SHAP 값"),
                    y=alt.Y(
                        "표시명:N", sort=alt.SortField("abs_shap", order="descending"), title=""
                    ),
                    color=alt.Color(
                        "영향방향:N",
                        scale=alt.Scale(
                            domain=["위험 증가", "위험 완화"],
                            range=[COLOR_RISK, COLOR_MITIGATE],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("표시명:N", title="변수"),
                        alt.Tooltip("shap_value:Q", title="SHAP", format=".2f"),
                        alt.Tooltip("실제값:N", title="실제값"),
                        alt.Tooltip("영향방향:N", title="방향"),
                    ],
                )
                .properties(height=360)
            )
            stretch_altair_chart(chart)
            local_table = local_view.loc[
                :,
                [
                    "rank",
                    "표시명",
                    "실제값",
                    "일반 해석 방향",
                    "shap_value",
                    "abs_shap",
                    "영향방향",
                ],
            ].copy()
            local_table = local_table.rename(
                columns={
                    "rank": "순위",
                    "shap_value": "SHAP 값",
                    "abs_shap": "|SHAP|",
                }
            )
            styled_local = (
                local_table.style.map(style_direction_badge, subset=["일반 해석 방향"])
                .set_properties(subset=["일반 해석 방향"], **{"text-align": "center"})
                .hide(axis="index")
            )
            stretch_dataframe(styled_local, hide_index=True)
            return

    st.info(
        "현재는 전체 모델 기준 주요 영향 요인을 보여주고 있습니다. "
        "기업별 상세 영향 값이 연결되면 이 탭은 자동으로 해당 기업 기준 결과를 우선 표시합니다."
    )
    feature_map = build_company_feature_map(selected_row, artifacts.feature_dictionary)
    merged = artifacts.global_shap_reference.merge(
        feature_map.loc[:, ["feature", "value"]],
        how="left",
        on="feature",
    )
    top_features = merged.sort_values("rank").head(10).copy()
    top_features["표시명"] = top_features["feature"].map(
        lambda value: display_name(value, feature_map)
    )
    top_features["실제값"] = top_features.apply(
        lambda row: format_value_with_unit(row["value"], row.get("unit", ""), str(row["feature"])),
        axis=1,
    )
    top_features["일반 해석 방향"] = top_features["feature"].map(
        lambda value: get_feature_direction_label(str(value))
    )
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    render_accent_summary_card(
        summary_col1,
        "가장 중요한 변수",
        top_features.iloc[0]["표시명"] if not top_features.empty else "없음",
        "기업별 상세 영향 값이 없어 전체 모델 기준으로 보여줍니다.",
        COLOR_NEUTRAL,
    )
    render_accent_summary_card(
        summary_col2,
        "상위 설명축",
        str(top_features.iloc[0]["feature_group"]) if not top_features.empty else "없음",
        "현재 전체 데이터에서 평균적으로 크게 작용하는 변수군입니다.",
        COLOR_COMPANY,
    )
    render_accent_summary_card(
        summary_col3,
        "상위 SHAP 강도",
        format_scalar(top_features["mean_abs_shap"].head(5).mean()),
        "상위 5개 변수의 평균 |SHAP| 수준입니다.",
        COLOR_NEUTRAL,
    )
    chart = (
        alt.Chart(top_features)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=COLOR_NEUTRAL)
        .encode(
            x=alt.X("mean_abs_shap:Q", title="평균 |SHAP|"),
            y=alt.Y("표시명:N", sort=alt.SortField("mean_abs_shap", order="descending"), title=""),
            tooltip=[
                alt.Tooltip("표시명:N", title="변수"),
                alt.Tooltip("mean_abs_shap:Q", title="평균 |SHAP|", format=".2f"),
                alt.Tooltip("실제값:N", title="실제값"),
            ],
        )
        .properties(height=360)
    )
    stretch_altair_chart(chart)
    global_table = top_features.loc[
        :,
        ["rank", "표시명", "feature_group", "일반 해석 방향", "mean_abs_shap", "실제값"],
    ].copy()
    global_table = global_table.rename(
        columns={
            "rank": "순위",
            "feature_group": "변수군",
            "mean_abs_shap": "평균 |SHAP|",
        }
    )
    styled_global = (
        global_table.style.map(style_direction_badge, subset=["일반 해석 방향"])
        .set_properties(subset=["일반 해석 방향"], **{"text-align": "center"})
        .hide(axis="index")
    )
    stretch_dataframe(styled_global, hide_index=True)


def render_peer_tab(
    selected_row: pd.Series,
    artifacts: DashboardArtifacts,
) -> None:
    """Render the peer comparison tab."""
    st.subheader("시장/산업 비교")
    st.caption(
        "선택한 기업의 주요 지표를 같은 산업과 같은 시장의 기준값과 나란히 비교해 보여줍니다."
    )
    peer_slice = resolve_company_peer_slice(selected_row, artifacts.peer_percentiles)
    local_shap = resolve_company_local_shap(selected_row, artifacts.local_shap)

    feature_map = build_company_feature_map(selected_row, artifacts.feature_dictionary)
    peer_slice = peer_slice.merge(
        feature_map.loc[:, ["feature", "korean_name", "feature_group"]],
        how="left",
        on="feature",
    )
    if not local_shap.empty:
        peer_slice = peer_slice.merge(
            local_shap.loc[:, ["feature", "direction", "abs_shap"]],
            how="left",
            on="feature",
        )
    else:
        peer_slice["direction"] = pd.NA
        peer_slice["abs_shap"] = pd.NA
    peer_slice["industry_gap"] = peer_slice["value"] - peer_slice["industry_median"]
    peer_slice["market_gap"] = peer_slice["value"] - peer_slice["market_median"]
    peer_slice["표시명"] = peer_slice["feature"].map(lambda value: display_name(value, feature_map))
    peer_slice["unit"] = peer_slice["feature"].map(
        lambda value: get_feature_unit(str(value), feature_map)
    )
    peer_slice["일반 해석 방향"] = peer_slice["feature"].map(
        lambda value: get_feature_direction_label(str(value))
    )
    peer_slice["선택 기업"] = peer_slice.apply(
        lambda row: format_value_with_unit(
            row["value"],
            str(row["unit"]),
            str(row["feature"]),
        ),
        axis=1,
    )
    peer_slice["산업 중앙값"] = peer_slice.apply(
        lambda row: format_value_with_unit(
            row["industry_median"],
            str(row["unit"]),
            str(row["feature"]),
        ),
        axis=1,
    )
    peer_slice["시장 중앙값"] = peer_slice.apply(
        lambda row: format_value_with_unit(
            row["market_median"],
            str(row["unit"]),
            str(row["feature"]),
        ),
        axis=1,
    )
    peer_slice["산업 내 위치"] = peer_slice["industry_percentile"].map(format_percentile_label)
    peer_slice["시장 내 위치"] = peer_slice["market_percentile"].map(format_percentile_label)

    compare_features: list[str] = st.multiselect(
        "비교할 변수 선택",
        options=peer_slice["feature"].tolist(),
        format_func=lambda value: display_name(value, feature_map),
        default=[
            feature
            for feature in [
                "cash_ratio",
                "interest_coverage_ratio",
                "capital_impairment_ratio",
                "net_margin",
                "short_term_borrowings_share",
            ]
            if feature in peer_slice["feature"].tolist()
        ],
    )
    if compare_features:
        table = peer_slice.loc[peer_slice["feature"].isin(compare_features)].copy()
    else:
        table = peer_slice.head(10).copy()

    def build_peer_summary_line(row: pd.Series) -> str:
        direction_label = "높음" if float(row["industry_gap"]) > 0 else "낮음"
        percentile_text = format_percentile_label(row["industry_percentile"])
        return f"{row['표시명']}: 산업 대비 {direction_label}, 산업 내 {percentile_text}"

    def build_peer_memo_line(row: pd.Series) -> str:
        industry_gap = float(row["industry_gap"])
        gap_text = format_delta_with_unit(abs(industry_gap), str(row["unit"]), str(row["feature"]))
        percentile_text = format_percentile_label(row["industry_percentile"])
        shap_label = (
            "위험을 높이는 쪽" if row["direction"] == "increase_risk" else "위험을 낮추는 쪽"
        )
        if industry_gap > 0:
            level_text = f"산업 중앙값보다 {gap_text} 높은 수준이며"
        elif industry_gap < 0:
            level_text = f"산업 중앙값보다 {gap_text} 낮은 수준이며"
        else:
            level_text = "산업 중앙값과 유사한 수준이며"
        return (
            f"{row['표시명']}은(는) {level_text} 산업 내에서는 {percentile_text} 수준에 해당합니다. "
            f"일반적으로는 '{row['일반 해석 방향']}'으로 해석하며, 현재 모델에서는 이 지표가 {shap_label}으로 작용하는 모습으로 나타납니다."
        )

    vulnerability_lines: list[str] = []
    strength_lines: list[str] = []
    vulnerability_memo_lines: list[str] = []
    strength_memo_lines: list[str] = []
    if not table.empty:
        summary_frame = table.copy()
        summary_frame["industry_distance"] = (summary_frame["industry_percentile"] - 50.0).abs()
        summary_frame["summary_score"] = summary_frame["industry_distance"] * summary_frame[
            "abs_shap"
        ].fillna(0)

        vulnerable = summary_frame.loc[summary_frame["direction"] == "increase_risk"].copy()
        vulnerable = vulnerable.sort_values(
            ["summary_score", "abs_shap", "industry_distance"],
            ascending=[False, False, False],
        ).head(3)
        vulnerability_lines = [build_peer_summary_line(row) for _, row in vulnerable.iterrows()]
        vulnerability_memo_lines = [build_peer_memo_line(row) for _, row in vulnerable.iterrows()]

        strong = summary_frame.loc[summary_frame["direction"] == "decrease_risk"].copy()
        strong = strong.sort_values(
            ["summary_score", "abs_shap", "industry_distance"],
            ascending=[False, False, False],
        ).head(3)
        strength_lines = [build_peer_summary_line(row) for _, row in strong.iterrows()]
        strength_memo_lines = [build_peer_memo_line(row) for _, row in strong.iterrows()]

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    render_bullet_card(
        summary_col1,
        "취약 지표 3개",
        vulnerability_lines,
        COLOR_RISK,
        "현재 선택한 변수 중 위험 증가 방향으로 두드러진 지표가 없습니다.",
    )
    render_bullet_card(
        summary_col2,
        "우수 지표 3개",
        strength_lines,
        COLOR_MITIGATE,
        "현재 선택한 변수 중 위험 완화 방향으로 두드러진 지표가 없습니다.",
    )
    render_text_card(
        summary_col3,
        "해석 기준",
        "취약·우수 지표는 현재 기업의 local SHAP 방향과 산업 내 상대 위치를 함께 반영해 자동으로 정리합니다.",
    )
    with st.expander("자세한 해석 보기"):
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.markdown("**취약 지표 상세 해석**")
            if vulnerability_memo_lines:
                for line in vulnerability_memo_lines:
                    st.markdown(f"- {line}")
            else:
                st.caption("현재 선택한 변수 중 위험 증가 방향으로 두드러진 지표가 없습니다.")
        with detail_col2:
            st.markdown("**우수 지표 상세 해석**")
            if strength_memo_lines:
                for line in strength_memo_lines:
                    st.markdown(f"- {line}")
            else:
                st.caption("현재 선택한 변수 중 위험 완화 방향으로 두드러진 지표가 없습니다.")

    table_units = set(table["unit"].dropna().astype(str).tolist())
    money_only_view = bool(table_units) and table_units == {"KRW thousand"}
    chart_rows: list[dict[str, object]] = []
    for row in table.to_dict(orient="records"):
        label = (
            str(row["korean_name"])
            if pd.notna(row["korean_name"]) and str(row["korean_name"]).strip()
            else str(row["feature"])
        )
        unit = str(row["unit"])
        company_value = (
            float(row["value"]) * 1000 / 100_000_000
            if money_only_view and pd.notna(row["value"])
            else row["value"]
        )
        industry_value = (
            float(row["industry_median"]) * 1000 / 100_000_000
            if money_only_view and pd.notna(row["industry_median"])
            else row["industry_median"]
        )
        market_value = (
            float(row["market_median"]) * 1000 / 100_000_000
            if money_only_view and pd.notna(row["market_median"])
            else row["market_median"]
        )
        chart_rows.extend(
            [
                {
                    "구분": label,
                    "기준": "선택 기업",
                    "값": company_value,
                    "값_표시": format_value_with_unit(row["value"], unit, str(row["feature"])),
                },
                {
                    "구분": label,
                    "기준": "동일 산업 중앙값",
                    "값": industry_value,
                    "값_표시": format_value_with_unit(
                        row["industry_median"], unit, str(row["feature"])
                    ),
                },
                {
                    "구분": label,
                    "기준": "전체 시장 중앙값",
                    "값": market_value,
                    "값_표시": format_value_with_unit(
                        row["market_median"], unit, str(row["feature"])
                    ),
                },
            ]
        )
    value_axis_title = "값 (억 원)" if money_only_view else "값"
    st.markdown("**절대값 비교**")
    legend_col1, legend_col2, legend_col3 = st.columns(3)
    render_legend_card(
        legend_col1, "선택 기업", "현재 선택한 기업의 실제 지표값입니다.", COLOR_COMPANY
    )
    render_legend_card(
        legend_col2, "동일 산업 중앙값", "같은 시장·산업 기업들의 중앙값입니다.", COLOR_INDUSTRY
    )
    render_legend_card(
        legend_col3, "전체 시장 중앙값", "같은 시장 전체 기업들의 중앙값입니다.", COLOR_MARKET
    )
    if len(table_units) <= 1:
        compare_chart = (
            alt.Chart(pd.DataFrame(chart_rows))
            .mark_bar()
            .encode(
                x=alt.X("값:Q", title=value_axis_title),
                y=alt.Y("구분:N", title="", sort="-x"),
                color=alt.Color(
                    "기준:N",
                    scale=alt.Scale(
                        domain=["선택 기업", "동일 산업 중앙값", "전체 시장 중앙값"],
                        range=[COLOR_COMPANY, COLOR_INDUSTRY, COLOR_MARKET],
                    ),
                    legend=alt.Legend(title="비교 기준", orient="top"),
                ),
                xOffset="기준:N",
                tooltip=["구분:N", "기준:N", alt.Tooltip("값_표시:N", title="값")],
            )
            .properties(height=360)
        )
        stretch_altair_chart(compare_chart)
    else:
        st.caption("선택한 변수의 단위가 섞여 있어 변수별 비교 카드로 나누어 표시합니다.")
        detail_cols = st.columns(2)
        for index, row in enumerate(table.to_dict(orient="records")):
            row_chart_data = pd.DataFrame(
                [
                    {"기준": "선택 기업", "값": row["value"], "값_표시": row["선택 기업"]},
                    {
                        "기준": "동일 산업 중앙값",
                        "값": row["industry_median"],
                        "값_표시": row["산업 중앙값"],
                    },
                    {
                        "기준": "전체 시장 중앙값",
                        "값": row["market_median"],
                        "값_표시": row["시장 중앙값"],
                    },
                ]
            )
            if str(row["unit"]) == "KRW thousand":
                row_chart_data["값"] = row_chart_data["값"].astype(float) * 1000 / 100_000_000
                axis_title = "값 (억 원)"
            else:
                axis_title = "값"
            with detail_cols[index % 2]:
                render_text_card(
                    st.container(),
                    str(row["표시명"]),
                    f"산업 중앙값 {row['산업 중앙값']} / 시장 중앙값 {row['시장 중앙값']} 기준으로 비교합니다.",
                )
                mini_chart = (
                    alt.Chart(row_chart_data)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("값:Q", title=axis_title),
                        y=alt.Y("기준:N", title=""),
                        color=alt.Color(
                            "기준:N",
                            scale=alt.Scale(
                                domain=["선택 기업", "동일 산업 중앙값", "전체 시장 중앙값"],
                                range=[COLOR_COMPANY, COLOR_INDUSTRY, COLOR_MARKET],
                            ),
                            legend=None,
                        ),
                        tooltip=["기준:N", alt.Tooltip("값_표시:N", title="값")],
                    )
                    .properties(height=150)
                )
                stretch_altair_chart(mini_chart)

    table["산업 대비 차이"] = table.apply(
        lambda row: format_delta_with_unit(
            row["industry_gap"],
            str(row["unit"]),
            str(row["feature"]),
        ),
        axis=1,
    )
    table["시장 대비 차이"] = table.apply(
        lambda row: format_delta_with_unit(
            row["market_gap"],
            str(row["unit"]),
            str(row["feature"]),
        ),
        axis=1,
    )

    gap_rows: list[dict[str, object]] = []
    percentile_rows: list[dict[str, object]] = []
    for row in table.to_dict(orient="records"):
        label = str(row["표시명"])
        unit = str(row["unit"])
        industry_gap_value = (
            float(row["industry_gap"]) * 1000 / 100_000_000
            if money_only_view and pd.notna(row["industry_gap"])
            else row["industry_gap"]
        )
        market_gap_value = (
            float(row["market_gap"]) * 1000 / 100_000_000
            if money_only_view and pd.notna(row["market_gap"])
            else row["market_gap"]
        )
        gap_rows.extend(
            [
                {
                    "구분": label,
                    "비교": "산업 대비 차이",
                    "값": industry_gap_value,
                    "값_표시": format_delta_with_unit(
                        row["industry_gap"], unit, str(row["feature"])
                    ),
                },
                {
                    "구분": label,
                    "비교": "시장 대비 차이",
                    "값": market_gap_value,
                    "값_표시": format_delta_with_unit(row["market_gap"], unit, str(row["feature"])),
                },
            ]
        )
        percentile_rows.extend(
            [
                {"구분": label, "기준": "산업 내 위치", "백분위": row["industry_percentile"]},
                {"구분": label, "기준": "시장 내 위치", "백분위": row["market_percentile"]},
            ]
        )

    zero_rule = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color=COLOR_MUTED, strokeDash=[4, 4])
        .encode(x="x:Q")
    )
    gap_base = alt.Chart(pd.DataFrame(gap_rows))
    gap_bars = gap_base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X(
            "값:Q",
            title="선택 기업 - 비교 기준 (억 원)" if money_only_view else "선택 기업 - 비교 기준",
        ),
        y=alt.Y("구분:N", title="", sort=table["표시명"].tolist()),
        color=alt.Color(
            "비교:N",
            scale=alt.Scale(
                domain=["산업 대비 차이", "시장 대비 차이"],
                range=[COLOR_INDUSTRY, COLOR_MARKET],
            ),
            legend=alt.Legend(title="차이 기준", orient="top"),
        ),
        xOffset="비교:N",
        tooltip=["구분:N", "비교:N", alt.Tooltip("값_표시:N", title="차이")],
    )
    gap_chart = alt.layer(zero_rule, gap_bars).properties(height=340)

    percentile_base = alt.Chart(pd.DataFrame(percentile_rows))
    percentile_points = percentile_base.mark_circle(size=170).encode(
        x=alt.X("백분위:Q", title="백분위 위치", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("구분:N", title="", sort=table["표시명"].tolist()),
        color=alt.Color(
            "기준:N",
            scale=alt.Scale(
                domain=["산업 내 위치", "시장 내 위치"],
                range=[COLOR_COMPANY, COLOR_SOFT_BLUE],
            ),
            legend=alt.Legend(title="백분위 기준", orient="top"),
        ),
        tooltip=["구분:N", "기준:N", alt.Tooltip("백분위:Q", format=".2f")],
    )
    percentile_mid_rule = (
        alt.Chart(pd.DataFrame({"x": [50]}))
        .mark_rule(color=COLOR_MUTED, strokeDash=[4, 4])
        .encode(x="x:Q")
    )
    percentile_chart = alt.layer(percentile_mid_rule, percentile_points).properties(height=340)

    col_gap, col_percentile = st.columns(2)
    with col_gap:
        st.markdown("**비교 기준 대비 차이**")
        if len(table_units) <= 1:
            stretch_altair_chart(gap_chart)
        else:
            st.caption("단위가 섞여 있어 차이는 표에서 변수별로 읽는 것이 더 적절합니다.")
        st.caption("0보다 크면 선택 기업 값이 비교 기준보다 높고, 0보다 작으면 낮습니다.")
    with col_percentile:
        st.markdown("**산업/시장 내 백분위 위치**")
        stretch_altair_chart(percentile_chart)
        st.caption("50백분위 점선을 기준으로, 오른쪽일수록 상대적으로 높은 수준입니다.")

    table_view = table.loc[
        :,
        [
            "표시명",
            "선택 기업",
            "산업 중앙값",
            "시장 중앙값",
            "산업 대비 차이",
            "시장 대비 차이",
            "일반 해석 방향",
            "산업 내 위치",
            "시장 내 위치",
        ],
    ].copy()
    styled_table = (
        table_view.style.map(style_direction_badge, subset=["일반 해석 방향"])
        .set_properties(subset=["일반 해석 방향"], **{"text-align": "center"})
        .hide(axis="index")
    )
    stretch_dataframe(
        styled_table,
        hide_index=True,
    )
    st.caption(
        "`일반 해석 방향`은 재무 일반론 기준의 안내이며, 실제 평가는 산업 특성과 기업 상황에 따라 달라질 수 있습니다."
    )


def render_industry_tab(
    selected_row: pd.Series,
    artifacts: DashboardArtifacts,
) -> None:
    """Render the industry aggregate tab."""
    default_share_label = "기본 기준선(0.5) 적용 시 고위험 판정 비중"
    tuned_share_label = "조정 기준선 적용 시 고위험 판정 비중"
    st.subheader("산업 흐름 보기")
    st.caption(
        "선택한 기업이 속한 시장과 산업을 기준으로, 현재 수준과 연도별 흐름을 함께 보여줍니다."
    )
    intro_col1, intro_col2, intro_col3 = st.columns(3)
    render_text_card(
        intro_col1,
        "어떤 기준인가요?",
        "선택한 기업과 같은 시장·산업에 속한 기업들을 함께 묶어 보여줍니다.",
    )
    render_text_card(
        intro_col2,
        "현재 산업 수준",
        "각 기업의 가장 최근 연도 자료를 기준으로, 현재 산업 분위기를 보여줍니다.",
    )
    render_text_card(
        intro_col3,
        "시간에 따른 변화",
        "연도별로 위험 수준이 어떻게 달라졌는지, 실제 투기등급 비율과 함께 확인할 수 있습니다.",
    )

    if artifacts.industry_latest_summary is None or artifacts.industry_year_summary is None:
        st.info("산업 집계 파일이 아직 연결되지 않았습니다.")
        return

    market = str(selected_row["market"])
    industry = str(selected_row["industry_macro_category"])

    latest_summary = artifacts.industry_latest_summary.loc[
        (artifacts.industry_latest_summary["market"] == market)
        & (artifacts.industry_latest_summary["industry_macro_category"] == industry)
    ]
    if latest_summary.empty:
        st.warning("선택한 기업의 시장/산업에 해당하는 최신 집계가 없습니다.")
        return

    latest_row = latest_summary.iloc[0]
    shap_summary = None
    if artifacts.industry_shap_summary is not None:
        shap_summary = artifacts.industry_shap_summary.loc[
            (artifacts.industry_shap_summary["market"] == market)
            & (artifacts.industry_shap_summary["industry_macro_category"] == industry)
            & (artifacts.industry_shap_summary["split"] == "test")
        ].sort_values("rank_within_group")

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    render_accent_summary_card(
        summary_col1,
        "현재 산업 위험 수준",
        format_percent(latest_row.get("mean_prob_speculative")),
        "선택 기업과 같은 시장·산업의 최근 평균 위험 수준입니다.",
        COLOR_RISK,
    )
    render_accent_summary_card(
        summary_col2,
        tuned_share_label,
        format_percent(latest_row.get("pred_share_tuned")),
        "기본 0.5 대신 검증 구간에서 정한 조정 기준선을 적용했을 때, 고위험으로 판정되는 기업 비중입니다.",
        COLOR_NEUTRAL,
    )
    render_accent_summary_card(
        summary_col3,
        "산업에서 먼저 보는 지표",
        display_name(
            str(shap_summary.iloc[0]["feature"]),
            build_company_feature_map(selected_row, artifacts.feature_dictionary),
        )
        if shap_summary is not None and not shap_summary.empty
        else "없음",
        "이 산업에서 상대적으로 먼저 확인되는 설명 변수입니다.",
        COLOR_COMPANY,
    )

    c1, c2, c3, c4 = st.columns(4)
    render_bold_value_block(c1, "산업 내 기업 수", format_scalar(latest_row.get("companies")))
    render_bold_value_block(
        c2, "산업 평균 위험확률", format_percent(latest_row.get("mean_prob_speculative"))
    )
    render_bold_value_block(
        c3, "산업 중앙 위험확률", format_percent(latest_row.get("median_prob_speculative"))
    )
    render_bold_value_block(
        c4, tuned_share_label, format_percent(latest_row.get("pred_share_tuned"))
    )

    st.caption(
        f"{to_market_label(market)} / {to_industry_label(industry)} 기준으로 최근 기업 자료를 모아 본 결과입니다."
    )
    st.info(
        "여기서 말하는 '조정 기준선'은 기본 0.5 대신, 모델 성능 균형을 고려해 따로 정한 판정 기준입니다."
    )

    year_summary = artifacts.industry_year_summary.loc[
        (artifacts.industry_year_summary["market"] == market)
        & (artifacts.industry_year_summary["industry_macro_category"] == industry)
    ].copy()
    if not year_summary.empty:
        st.subheader("연도별 산업 흐름")
        trend_long = year_summary.melt(
            id_vars=["fiscal_year"],
            value_vars=["mean_prob_speculative", "pred_share_tuned", "positive_rate"],
            var_name="지표",
            value_name="값",
        )
        trend_long["지표"] = trend_long["지표"].replace(
            {
                "mean_prob_speculative": "산업 평균 위험확률",
                "pred_share_tuned": tuned_share_label,
                "positive_rate": "실제 투기등급 기업 비율",
            }
        )
        trend_chart = (
            alt.Chart(trend_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("fiscal_year:O", title="회계연도"),
                y=alt.Y("값:Q", title="비율", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "지표:N",
                    title="지표",
                    scale=alt.Scale(
                        domain=["산업 평균 위험확률", tuned_share_label, "실제 투기등급 기업 비율"],
                        range=[COLOR_RISK, COLOR_NEUTRAL, COLOR_DARK],
                    ),
                ),
                tooltip=["fiscal_year:O", "지표:N", alt.Tooltip("값:Q", format=".2%")],
            )
            .properties(height=320)
        )
        stretch_altair_chart(trend_chart)
        year_summary_view = year_summary.copy()
        for column in [
            "positive_rate",
            "mean_prob_speculative",
            "median_prob_speculative",
            "pred_share_0_5",
            "pred_share_tuned",
        ]:
            year_summary_view[column] = year_summary_view[column].map(format_percent)
        year_summary_view = year_summary_view.rename(
            columns={
                "market": "시장",
                "industry_macro_category": "산업",
                "fiscal_year": "회계연도",
                "companies": "기업 수",
                "positive_rows": "투기등급 기업 수",
                "positive_rate": "투기등급 비율",
                "mean_prob_speculative": "산업 평균 위험확률",
                "median_prob_speculative": "중앙 위험확률",
                "pred_share_0_5": default_share_label,
                "pred_share_tuned": tuned_share_label,
            }
        )
        year_summary_view["시장"] = year_summary_view["시장"].map(to_market_label)
        year_summary_view["산업"] = year_summary_view["산업"].map(to_industry_label)
        stretch_dataframe(
            year_summary_view.loc[
                :,
                [
                    "시장",
                    "산업",
                    "회계연도",
                    "기업 수",
                    "투기등급 기업 수",
                    "투기등급 비율",
                    "산업 평균 위험확률",
                    "중앙 위험확률",
                    default_share_label,
                    tuned_share_label,
                ],
            ],
            hide_index=True,
        )

    if shap_summary is not None and not shap_summary.empty:
        st.subheader("산업 기준 주요 설명 변수")
        top_shap = shap_summary.head(10).copy()
        feature_map = build_company_feature_map(selected_row, artifacts.feature_dictionary)
        top_shap["표시명"] = top_shap["feature"].map(lambda value: display_name(value, feature_map))
        top_shap["일반 해석 방향"] = top_shap["feature"].map(
            lambda value: get_feature_direction_label(str(value))
        )
        chart = (
            alt.Chart(top_shap)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=COLOR_NEUTRAL)
            .encode(
                x=alt.X("mean_abs_shap:Q", title="평균 |SHAP|"),
                y=alt.Y(
                    "표시명:N", sort=alt.SortField("mean_abs_shap", order="descending"), title=""
                ),
                tooltip=[
                    alt.Tooltip("표시명:N", title="변수"),
                    alt.Tooltip("mean_abs_shap:Q", title="평균 |SHAP|", format=".2f"),
                ],
            )
            .properties(height=320)
        )
        stretch_altair_chart(chart)
        top_shap_view = top_shap.loc[
            :,
            ["rank_within_group", "표시명", "일반 해석 방향", "mean_abs_shap", "mean_signed_shap"],
        ].rename(
            columns={
                "rank_within_group": "순위",
                "표시명": "지표",
                "mean_abs_shap": "평균 |SHAP|",
                "mean_signed_shap": "평균 방향성",
            }
        )
        styled_industry = (
            top_shap_view.style.map(style_direction_badge, subset=["일반 해석 방향"])
            .set_properties(subset=["일반 해석 방향"], **{"text-align": "center"})
            .hide(axis="index")
        )
        stretch_dataframe(styled_industry, hide_index=True)


def render_scenario_tab(
    selected_row: pd.Series,
    artifacts: DashboardArtifacts,
) -> None:
    """Render the scenario tab."""
    st.subheader("가정별 변화 보기")
    st.caption(
        "핵심 지표 값을 가정적으로 바꿔 보면서, 현재 기업의 상대적 위치가 어떻게 달라지는지 살펴봅니다."
    )
    presets = list(artifacts.scenario_presets.keys())
    preset_label_map = {
        "base": "기본",
        "mild_stress": "완만한 스트레스",
        "severe_stress": "강한 스트레스",
    }
    selected_preset = st.selectbox(
        "시나리오 선택", presets, format_func=lambda value: preset_label_map.get(value, value)
    )
    preset_changes = artifacts.scenario_presets[selected_preset]
    intro_col1, intro_col2, intro_col3 = st.columns(3)
    render_text_card(
        intro_col1,
        "현재 시나리오",
        f"현재 선택한 시나리오는 {preset_label_map.get(selected_preset, selected_preset)}입니다.",
    )
    render_text_card(
        intro_col2,
        "어떻게 보나요?",
        "핵심 지표 값을 가정적으로 바꿔 보고, 산업이나 시장 안에서 위치가 어떻게 달라지는지 확인합니다.",
    )
    render_text_card(
        intro_col3,
        "해석할 때 참고할 점",
        "현재는 예측확률을 다시 계산하는 단계가 아니라, 지표 수준 변화와 상대적 위치 변화를 중심으로 보여줍니다.",
    )

    scenario_features = [
        "spec_spread",
        "cash_ratio",
        "net_margin",
        "short_term_borrowings_share",
        "capital_impairment_ratio",
    ]
    rows: list[dict[str, object]] = []
    for feature in scenario_features:
        baseline_value = selected_row.get(feature)
        default_delta = (
            float(preset_changes.get(feature, 0.0)) if isinstance(preset_changes, dict) else 0.0
        )
        feature_map = build_company_feature_map(selected_row, artifacts.feature_dictionary)
        label = display_name(feature, feature_map)
        unit = get_feature_unit(feature, feature_map)
        delta = st.slider(
            f"{label} 얼마나 바꿔볼까요?",
            min_value=-1.0,
            max_value=1.0,
            value=default_delta,
            step=0.01,
        )
        scenario_value = None if pd.isna(baseline_value) else float(baseline_value) + delta
        distribution = (
            artifacts.company_universe.loc[:, feature]
            if feature in artifacts.company_universe
            else pd.Series(dtype=float)
        )
        scenario_percentile = (
            approximate_percentile(distribution, scenario_value)
            if scenario_value is not None
            else None
        )
        rows.append(
            {
                "변수": label,
                "feature": feature,
                "현재값": baseline_value,
                "변화량": delta,
                "시나리오 조정값": scenario_value,
                "시나리오 적용 후 대략적 위치": scenario_percentile,
                "unit": unit,
                "일반 해석 방향": get_feature_direction_label(feature),
            }
        )

    scenario_frame = pd.DataFrame(rows)
    scenario_frame["현재값_표시"] = scenario_frame.apply(
        lambda row: format_value_with_unit(row["현재값"], row["unit"], str(row["feature"])),
        axis=1,
    )
    scenario_frame["시나리오 조정값_표시"] = scenario_frame.apply(
        lambda row: format_value_with_unit(
            row["시나리오 조정값"], row["unit"], str(row["feature"])
        ),
        axis=1,
    )
    scenario_frame["시나리오 적용 후 위치"] = scenario_frame["시나리오 적용 후 대략적 위치"].map(
        format_percentile_label
    )
    strongest_change = (
        scenario_frame.loc[scenario_frame["변화량"].abs().idxmax()]
        if not scenario_frame.empty
        else None
    )
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    render_accent_summary_card(
        summary_col1,
        "현재 시나리오",
        preset_label_map.get(selected_preset, selected_preset),
        "슬라이더 시작값은 이 시나리오를 기준으로 자동 채워집니다.",
        COLOR_NEUTRAL,
    )
    render_accent_summary_card(
        summary_col2,
        "가장 크게 바꾼 지표",
        str(strongest_change["변수"]) if strongest_change is not None else "없음",
        format_delta_with_unit(
            strongest_change["변화량"],
            strongest_change["unit"],
            str(strongest_change["feature"]),
        )
        if strongest_change is not None
        else "-",
        COLOR_RISK,
    )
    render_accent_summary_card(
        summary_col3,
        "바꿔볼 수 있는 지표 수",
        format_scalar(len(scenario_frame)),
        "현재 화면에서 직접 움직여 볼 수 있는 핵심 지표 개수입니다.",
        COLOR_COMPANY,
    )
    st.markdown("**시나리오 적용 전후 보기**")
    for unit_value, unit_frame in scenario_frame.groupby("unit", dropna=False):
        unit_label = describe_unit(str(unit_value))
        st.markdown(f"**{unit_label}**")
        chart_rows: list[dict[str, object]] = []
        money_view = str(unit_value) == "KRW thousand"
        for row in unit_frame.to_dict(orient="records"):
            current_value = (
                float(row["현재값"]) * 1000 / 100_000_000
                if money_view and pd.notna(row["현재값"])
                else row["현재값"]
            )
            scenario_value = (
                float(row["시나리오 조정값"]) * 1000 / 100_000_000
                if money_view and pd.notna(row["시나리오 조정값"])
                else row["시나리오 조정값"]
            )
            chart_rows.extend(
                [
                    {
                        "변수": row["변수"],
                        "구분": "현재 수준",
                        "값": current_value,
                        "값_표시": row["현재값_표시"],
                    },
                    {
                        "변수": row["변수"],
                        "구분": "시나리오 반영값",
                        "값": scenario_value,
                        "값_표시": row["시나리오 조정값_표시"],
                    },
                ]
            )
        scenario_chart = (
            alt.Chart(pd.DataFrame(chart_rows))
            .mark_bar()
            .encode(
                x=alt.X("값:Q", title="값 (억 원)" if money_view else "값"),
                y=alt.Y("변수:N", title=""),
                color=alt.Color(
                    "구분:N",
                    scale=alt.Scale(
                        domain=["현재 수준", "시나리오 반영값"], range=[COLOR_MUTED, COLOR_RISK]
                    ),
                ),
                xOffset="구분:N",
                tooltip=["변수:N", "구분:N", alt.Tooltip("값_표시:N", title="값")],
            )
            .properties(height=max(160, len(unit_frame) * 56))
        )
        stretch_altair_chart(scenario_chart)
    scenario_table = scenario_frame.loc[
        :,
        [
            "변수",
            "현재값_표시",
            "변화량",
            "시나리오 조정값_표시",
            "일반 해석 방향",
            "시나리오 적용 후 위치",
        ],
    ].rename(
        columns={
            "현재값_표시": "현재값",
            "시나리오 조정값_표시": "시나리오 조정값",
        }
    )
    styled_scenario = (
        scenario_table.style.map(style_direction_badge, subset=["일반 해석 방향"])
        .set_properties(subset=["일반 해석 방향"], **{"text-align": "center"})
        .hide(axis="index")
    )
    stretch_dataframe(
        styled_scenario,
        hide_index=True,
    )
    st.warning(
        "현재 시나리오 탭은 지표를 바꿔 보았을 때 상대적 위치가 어떻게 달라지는지 보여줍니다. "
        "기업별 예측확률을 다시 계산하는 기능은 다음 단계에서 추가할 수 있습니다."
    )


def render_footer(artifacts: DashboardArtifacts, *, developer_mode: bool) -> None:
    """Render footer metadata."""
    if developer_mode:
        with st.expander("LLM payload template 보기"):
            st.json(artifacts.llm_payload_template)
        with st.expander("Export manifest 보기"):
            st.json(artifacts.export_manifest)


def format_llm_error_message(error: Exception, provider_label: str) -> str:
    """Convert raw LLM errors into a short, user-friendly message."""
    message = str(error).strip()
    if not message:
        return f"{provider_label} 메모를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요."
    return (
        f"{provider_label} 메모를 불러오지 못했습니다. {message}"
        " 입력한 API 키와 모델명을 다시 확인한 뒤 한 번 더 시도해 주세요."
    )


def format_stage2_error_detail(error: Exception) -> str:
    """Return a short dashboard-safe Stage 2 error detail."""
    message = str(error).strip() or type(error).__name__
    message = re.sub(
        r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+",
        r"\1[redacted]",
        message,
    )
    message = re.sub(r"(sk-[A-Za-z0-9_-]{8})[A-Za-z0-9_-]+", r"\1...[redacted]", message)
    return message[:500]


def main() -> None:
    """Run the credit risk Streamlit dashboard MVP."""
    load_dotenv()
    st.set_page_config(page_title="기업 신용위험 분석 대시보드", layout="wide")
    st.title("기업 신용위험 분석 대시보드")
    st.caption(
        "기업별 위험 진단, 위원회 검토, 동종업계 비교, 산업 집계를 한 화면에서 확인할 수 있는 대시보드입니다. "
        "1차 모델 판단과 2차 에이전트 위원회 판단을 구분해 보여줍니다."
    )
    inject_dashboard_theme()

    preset_info = ARTIFACT_PRESETS["team_43"]
    artifact_dir_input = os.environ.get("CAS_DASHBOARD_ARTIFACT_DIR") or str(
        cast(Path, preset_info["path"])
    )

    st.sidebar.selectbox(
        "금액 표시 방식",
        options=list(MONEY_DISPLAY_MODES.keys()),
        index=0,
        format_func=lambda value: MONEY_DISPLAY_MODES.get(value, value),
        key="money_display_mode",
        help="상세 표기(억·만·원)와 단순 표기(억 원) 중 원하는 방식을 선택합니다.",
    )

    with st.sidebar.expander("에이전트별 모델 설정", expanded=True):
        st.caption("각 에이전트의 LLM을 실시간으로 교체합니다.")

        agent_model_options = [
            "openai:gpt-4o",
            "openai:gpt-4o-mini",
            "anthropic:claude-3-5-sonnet-latest",
            "anthropic:claude-3-haiku-20240307",
            "gemini:gemini-2.5-flash",
            "gemini:gemini-2.0-flash",
        ]

        def get_model_index(env_key: str, default_val: str) -> int:
            val = os.environ.get(env_key, default_val)
            return agent_model_options.index(val) if val in agent_model_options else 0

        quant_sel = st.selectbox(
            "재무 분석 (Quant)",
            options=agent_model_options,
            index=get_model_index("QUANT_AGENT_MODEL", "openai:gpt-4o"),
        )
        research_sel = st.selectbox(
            "리서치 (Research/Macro)",
            options=agent_model_options,
            index=get_model_index("RESEARCH_AGENT_MODEL", "anthropic:claude-3-5-sonnet-latest"),
        )
        manager_sel = st.selectbox(
            "의장 (Manager)",
            options=agent_model_options,
            index=get_model_index("MANAGER_AGENT_MODEL", "gemini:gemini-2.5-flash"),
        )

        os.environ["QUANT_AGENT_MODEL"] = quant_sel
        os.environ["RESEARCH_AGENT_MODEL"] = research_sel
        os.environ["MACRO_AGENT_MODEL"] = research_sel
        os.environ["MANAGER_AGENT_MODEL"] = manager_sel

    with st.sidebar.expander("AI 메모 설정", expanded=False):
        llm_provider = str(
            st.selectbox(
                "LLM 제공자",
                options=list(LLM_PROVIDER_LABELS.keys()),
                format_func=lambda value: LLM_PROVIDER_LABELS.get(value, value),
                key="llm_provider",
                help="AI 심사 메모 탭에서 사용할 LLM 제공자를 선택합니다.",
            )
        )
        model_options: list[tuple[str, str]] = RECOMMENDED_LLM_MODELS[llm_provider]
        model_labels = dict(model_options)
        model_ids = [model for model, _label in model_options]
        llm_model = str(
            st.selectbox(
                "LLM 모델",
                options=model_ids,
                format_func=lambda value: model_labels.get(str(value), str(value)),
                key="llm_model",
                help="추천 모델 중 하나를 선택합니다.",
            )
        )
        api_key_env_var = "OPENAI_API_KEY" if llm_provider == "openai" else "ANTHROPIC_API_KEY"
        llm_api_key = st.text_input(
            f"{LLM_PROVIDER_LABELS.get(llm_provider, llm_provider)} API 키",
            value=os.environ.get(api_key_env_var, ""),
            type="password",
            key=f"llm_api_key_{llm_provider}",
            help=f".env 또는 현재 환경변수의 {api_key_env_var} 값을 기본으로 사용합니다.",
        )
    with st.sidebar.expander("고급 설정", expanded=False):
        developer_mode = st.checkbox(
            "개발자 모드",
            value=False,
            help="개발/디버깅용 메타정보와 payload를 표시합니다.",
        )
        if developer_mode:
            custom_artifact = st.text_input(
                "대시보드 데이터 경로",
                value=artifact_dir_input,
                help="기본값 대신 다른 대시보드 아티팩트 폴더를 열고 싶을 때만 사용합니다.",
            ).strip()
            if custom_artifact:
                artifact_dir_input = custom_artifact
            st.caption("기본값은 현재 연결된 결과 폴더입니다.")
        else:
            st.caption("일반 사용 시에는 기본 설정 그대로 사용하면 됩니다.")

    try:
        artifacts = cached_load_dashboard_artifacts(artifact_dir_input or None)
    except FileNotFoundError as error:
        st.error(f"대시보드 입력 아티팩트를 찾을 수 없습니다: {error}")
        st.stop()

    selected_row = pick_selected_company(artifacts)
    st.session_state["company_selection"] = build_company_selection_from_row(selected_row.to_dict())
    prediction_row = resolve_company_prediction(selected_row, artifacts.prediction_scores)
    feature_map = build_company_feature_map(selected_row, artifacts.feature_dictionary)
    local_shap = resolve_company_local_shap(selected_row, artifacts.local_shap)
    peer_slice = resolve_company_peer_slice(selected_row, artifacts.peer_percentiles)

    (
        overview_tab,
        committee_tab,
        drivers_tab,
        peers_tab,
        industry_tab,
        scenario_tab,
        llm_tab,
    ) = st.tabs(
        [
            "개요",
            "위원회 검토",
            "주요 영향 요인",
            "시장/산업 비교",
            "산업 흐름 보기",
            "가정별 변화 보기",
            "AI 심사 메모",
        ]
    )

    with overview_tab:
        render_overview_tab(
            selected_row, prediction_row, artifacts.model_summary, feature_map, artifacts
        )
    with committee_tab:
        render_committee_view_tab(
            selected_row=selected_row,
            prediction_row=prediction_row,
            feature_map=feature_map,
            local_shap=local_shap,
            peer_slice=peer_slice,
            developer_mode=developer_mode,
        )
    with drivers_tab:
        render_drivers_tab(selected_row, artifacts)
    with peers_tab:
        render_peer_tab(selected_row, artifacts)
    with industry_tab:
        render_industry_tab(selected_row, artifacts)
    with scenario_tab:
        render_scenario_tab(selected_row, artifacts)
    with llm_tab:
        industry_latest_row = resolve_industry_latest_row(
            selected_row,
            artifacts.industry_latest_summary,
        )
        render_llm_panel(
            selected_row=selected_row,
            prediction_row=prediction_row,
            feature_map=feature_map,
            local_shap=local_shap,
            peer_slice=peer_slice,
            industry_latest_row=industry_latest_row,
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model,
            developer_mode=developer_mode,
        )

    render_footer(artifacts, developer_mode=developer_mode)


if __name__ == "__main__":
    main()
