"""Streamlit MVP for TS2000 Core29 dashboard exploration."""

from __future__ import annotations

import os
import re
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from cas.dashboard.data_loader import DashboardArtifacts, load_dashboard_artifacts
from cas.dashboard.llm import generate_openai_explanation

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

COLOR_RISK = "#c85050"
COLOR_MITIGATE = "#2f9e5b"
COLOR_NEUTRAL = "#4f6fad"
COLOR_MUTED = "#9aa3b2"
COLOR_SOFT_BLUE = "#7f93c9"
COLOR_DARK = "#4a4f57"
COLOR_COMPANY = "#1d4ed8"
COLOR_INDUSTRY = "#d97706"
COLOR_MARKET = "#6b7280"
COLOR_CARD_BG = "#f7f8fb"
COLOR_CARD_BORDER = "#e3e7ef"
COLOR_CARD_LABEL = "#5c6473"
COLOR_CARD_VALUE = "#1f2937"
CARD_SHADOW = "0 4px 14px rgba(15, 23, 42, 0.04)"

RECOMMENDED_LLM_MODELS = [
    ("gpt-5.5", "gpt-5.5 | 최고급 추론·요약"),
    ("gpt-5.4-mini", "gpt-5.4-mini | 속도·비용 균형"),
    ("gpt-4.1", "gpt-4.1 | 안정적인 고성능"),
    ("gpt-4.1-mini", "gpt-4.1-mini | 빠른 기본 옵션"),
]

LLM_OUTPUT_FORMATS = {
    "brief": "간단 요약",
    "memo": "기본 심사 메모",
    "detailed": "상세 보고서형",
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


@st.cache_data(show_spinner=False)
def cached_load_dashboard_artifacts(artifact_dir: str | None = None) -> DashboardArtifacts:
    """Cache dashboard artifact loading for Streamlit."""
    path = Path(artifact_dir) if artifact_dir else None
    return load_dashboard_artifacts(path)


def to_market_label(value: object) -> str:
    """Convert a market code into a Korean label."""
    return MARKET_LABELS.get(str(value), str(value))


def to_size_label(value: object) -> str:
    """Convert a firm size code into a Korean label."""
    return SIZE_LABELS.get(str(value), str(value))


def to_industry_label(value: object) -> str:
    """Convert an industry code into a Korean label."""
    return INDUSTRY_LABELS.get(str(value), str(value))


def to_prediction_label(value: object) -> str:
    """Convert a numeric prediction label into a Korean label."""
    try:
        return PREDICTION_LABELS.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def pick_selected_company(artifacts: DashboardArtifacts) -> pd.Series:
    """Render sidebar selectors and return the chosen company snapshot."""
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
        mask = latest["corp_name"].astype(str).str.contains(
            search_query, case=False, na=False
        ) | latest["stock_code"].astype(str).str.contains(search_query, case=False, na=False)
        latest = latest.loc[mask]

    if latest.empty:
        st.sidebar.warning("검색 조건에 맞는 기업이 없습니다. 검색어 또는 필터를 조정해 주세요.")
        st.stop()

    options = latest.assign(
        label=lambda frame: frame["corp_name"]
        + " | "
        + frame["stock_code"].astype(str)
        + " | FY"
        + frame["fiscal_year"].astype(int).astype(str)
    )
    labels = options["label"].tolist()
    selected_label = st.sidebar.selectbox("기업 선택", labels)
    return options.loc[options["label"] == selected_label].iloc[0]


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
    matched = local_shap.loc[
        (local_shap["stock_code"].astype(str) == str(selected_row["stock_code"]))
        & (local_shap["fiscal_year"] == selected_row["fiscal_year"])
    ].copy()
    return matched.sort_values("abs_shap", ascending=False)


def resolve_company_peer_slice(
    selected_row: pd.Series,
    peer_percentiles: pd.DataFrame,
) -> pd.DataFrame:
    """Return peer comparison rows for the selected company-year."""
    return peer_percentiles.loc[
        (peer_percentiles["stock_code"].astype(str) == str(selected_row["stock_code"]))
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
    matched = prediction_scores.loc[
        (prediction_scores["stock_code"].astype(str) == str(selected_row["stock_code"]))
        & (prediction_scores["fiscal_year"] == selected_row["fiscal_year"])
    ]
    if matched.empty:
        return None
    return matched.iloc[0]


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
        return f"{float(value) * 100:.2f}%"
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
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if unit_text == "ratio":
        return f"{number * 100:.2f}%"
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


def format_delta_with_unit(value: object, unit: object) -> str:
    """Format a signed delta using the feature unit for comparison views."""
    if pd.isna(value):
        return "-"

    unit_text = str(unit) if pd.notna(unit) else ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    sign = "+" if number > 0 else ""
    if unit_text == "ratio":
        return f"{sign}{number * 100:.2f}%p"
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
        return f"{float(value):.2f}백분위"
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
        return f"{base_style}background-color:#e8f6ee;color:{COLOR_MITIGATE};"
    if "낮을수록" in text or "아니오가" in text:
        return f"{base_style}background-color:#fff4dd;color:#b7791f;"
    return f"{base_style}background-color:#eef2f7;color:{COLOR_DARK};"


def render_direction_badge_html(value: object) -> str:
    """Render an interpretation direction badge as HTML."""
    text = str(value)
    style = style_direction_badge(text)
    return f"<span style='{style}'>{escape(text)}</span>"


def render_risk_band_badge(risk_band: object) -> str:
    """Render a colored HTML badge for the risk band."""
    label = str(risk_band) if pd.notna(risk_band) else "-"
    style_map = {
        "안정": {"bg": "#e8f6ee", "fg": COLOR_MITIGATE, "border": "#b9e3c8"},
        "관찰": {"bg": "#fff4dd", "fg": "#b7791f", "border": "#f2d39a"},
        "고위험": {"bg": "#fdeaea", "fg": COLOR_RISK, "border": "#f1bcbc"},
    }
    style = style_map.get(label, {"bg": "#eef2f7", "fg": COLOR_DARK, "border": "#d7dfe8"})
    return (
        f"<div style='display:inline-block;padding:0.45rem 0.8rem;border-radius:999px;"
        f"background:{style['bg']};color:{style['fg']};border:1px solid {style['border']};"
        "font-weight:700;font-size:0.95rem;'>"
        f"{label}</div>"
    )


def render_bold_value_block(
    container: st.delta_generator.DeltaGenerator, label: str, value: object
) -> None:
    """Render a bold label and value inside a consistent overview card."""
    container.markdown(
        (
            f"<div style='min-height:104px;padding:0.9rem 1rem;border-radius:14px;"
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
            f"<div style='min-height:104px;padding:0.9rem 1rem;border-radius:14px;"
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
            f"<div style='min-height:136px;padding:0.9rem 1rem;border-radius:14px;"
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
            f"<div style='min-height:120px;padding:0.95rem 1rem;border-radius:14px;"
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
            f"<div style='min-height:168px;padding:0.95rem 1rem;border-radius:14px;"
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
            f"<div style='padding:0.95rem 1.05rem;border-radius:14px;"
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
            f"<div style='min-height:188px;padding:0.95rem 1rem;border-radius:14px;"
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
            f"<div style='min-height:112px;padding:0.85rem 0.95rem;border-radius:14px;"
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
    sections = {
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
        "# TS2000 AI 심사 요약",
        "",
        f"- 기업명: {selected_row.get('corp_name')}",
        f"- 종목코드: {selected_row.get('stock_code')}",
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
            ),
            axis=1,
        )

    lines = [
        "# TS2000 원페이지 심사 메모",
        "",
        "## 기업 개요",
        f"- 기업명: {selected_row.get('corp_name')}",
        f"- 종목코드: {selected_row.get('stock_code')}",
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
  <title>TS2000 AI 심사 보고서</title>
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
          <div class="brand-mark">TS2000</div>
          <div class="brand-copy">
            <div class="brand-title">TS2000 기업 신용위험 분석 보고서</div>
            <div class="brand-subtitle">Corporate Analysis System 기반 AI 심사 메모 정리본입니다.</div>
          </div>
        </div>
        <div class="doc-badges">
          <div class="doc-chip">공식 Core29 기준</div>
          <div class="doc-chip">{escape(output_format_label)}</div>
        </div>
      </div>
      <div class="eyebrow">TS2000 CREDIT RISK MEMO</div>
      <h1>{escape(str(selected_row.get("corp_name")))}</h1>
      <div class="summary">{escape(headline)}</div>
    </div>
    <div class="meta-grid">
      <div class="meta-card"><div class="meta-label">종목코드</div><div class="meta-value">{escape(str(selected_row.get("stock_code")))}</div></div>
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
  <title>TS2000 원페이지 심사 메모</title>
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
          <div class="brand-mark">TS2000</div>
          <div class="brand-copy">
            <div class="brand-title">TS2000 원페이지 심사 메모</div>
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
      <div class="meta-card"><div class="meta-label">종목코드</div><div class="meta-value">{escape(str(selected_row.get("stock_code")))}</div></div>
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
            f"<div style='min-height:96px;padding:0.85rem 1rem;border-radius:14px;"
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
            f"<div style='min-height:120px;padding:0.95rem 1rem;border-radius:14px;"
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
    return (
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
            "stock_code": selected_row.get("stock_code"),
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

    st.subheader("모델 결과")
    if prediction_row is None:
        st.info(
            "현재 리포지토리 패키지에는 기업별 예측확률 파일이 포함되어 있지 않습니다. "
            "아래에는 공식 Core29 XGBoost의 전체 test 성능과 선택 기업의 핵심 지표를 함께 표시합니다."
        )
        xgboost_rows = [
            row
            for row in model_summary["test_overall_models"]
            if row["model"] == model_summary["selected_model"]
        ]
        selected_model = xgboost_rows[0] if xgboost_rows else None
        threshold_rows = model_summary["xgboost_thresholds"]
        default_threshold = next(
            (row for row in threshold_rows if row["threshold_type"] == "default_0_5"),
            None,
        )
        c1, c2, c3, c4 = st.columns(4)
        render_bold_value_block(
            c1, "공식 PR-AUC", format_scalar(selected_model["pr_auc"] if selected_model else None)
        )
        render_bold_value_block(
            c2,
            "공식 Precision@0.5",
            format_scalar(selected_model["precision_at_0_5"] if selected_model else None),
        )
        render_bold_value_block(
            c3,
            "공식 Recall@0.5",
            format_scalar(selected_model["recall_at_0_5"] if selected_model else None),
        )
        render_bold_value_block(
            c4,
            "공식 Threshold",
            format_scalar(default_threshold["threshold"] if default_threshold else None),
        )
        st.caption("위 수치는 기업별 점수가 아니라 공식 Core29 XGBoost test 전체 성능입니다.")
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
            "기업별 점수는 대시보드 산출물 export 단계에서 공식 Core29 XGBoost 학습 레시피를 "
            "재현해 생성한 값입니다."
        )
        risk_band = str(prediction_row.get("risk_band"))
        probability_text = format_percent(prediction_row["prob_speculative"])
        threshold_text = format_scalar(prediction_row["threshold"])
        label_text = to_prediction_label(prediction_row.get("predicted_label"))
        if risk_band == "고위험":
            summary_text = (
                f"현재 투기등급 확률은 {probability_text}이며, tuned 기준선 {threshold_text}를 상회하여 "
                f"{label_text}으로 분류됩니다. 핵심 위험 요인을 우선 점검하는 것이 필요합니다."
            )
            summary_color = COLOR_RISK
        elif risk_band == "관찰":
            summary_text = (
                f"현재 투기등급 확률은 {probability_text}이며, 기준선 {threshold_text} 부근에서 "
                f"{label_text}으로 판정됩니다. 동종업계 대비 취약 지표를 함께 확인하는 것이 좋습니다."
            )
            summary_color = "#c0841a"
        else:
            summary_text = (
                f"현재 투기등급 확률은 {probability_text}이며, tuned 기준선 {threshold_text} 대비 "
                f"안정적인 수준입니다. 다만 주요 수익성과 유동성 지표를 함께 보는 것이 좋습니다."
            )
            summary_color = COLOR_MITIGATE
        render_summary_banner("한눈에 보기", summary_text, summary_color)
        chart_col, text_col = st.columns([1.2, 0.8])
        with chart_col:
            probability_chart = build_probability_chart(
                float(prediction_row["prob_speculative"]),
                float(prediction_row["threshold"]),
            )
            st.altair_chart(probability_chart, use_container_width=True)
        with text_col:
            st.markdown("**리스크 해석**")
            st.write(
                f"- 현재 위험확률은 **{format_percent(prediction_row['prob_speculative'])}** 입니다."
            )
            st.write(
                f"- tuned 기준선 **{format_scalar(prediction_row['threshold'])}** 대비 "
                f"판정 결과는 **{format_scalar(prediction_row['risk_band'])}** 구간입니다."
            )

    st.subheader("핵심 지표")
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
                artifacts.peer_percentiles["stock_code"].astype(str)
                == str(selected_row["stock_code"])
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
            st.markdown("**핵심 지표의 산업 내 위치**")
            st.altair_chart(percentile_chart, use_container_width=True)


def render_llm_panel(
    *,
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    feature_map: pd.DataFrame,
    local_shap: pd.DataFrame,
    peer_slice: pd.DataFrame,
    industry_latest_row: pd.Series | None,
    api_key: str,
    model: str,
    developer_mode: bool,
) -> None:
    """Render an optional LLM explanation section."""
    st.subheader("AI 심사 요약")
    st.caption("선택 기업의 점수와 비교 결과를 바탕으로 심사 메모 형태의 한국어 요약을 생성합니다.")
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
        "출력 형식",
        f"현재 선택한 형식은 {output_format_label}입니다. {format_description}",
    )
    render_text_card(
        intro_col2, "입력 근거", "예측확률, 핵심 지표, SHAP, 동종업계 비교 결과를 함께 참고합니다."
    )
    render_text_card(
        intro_col3,
        "모델 기준",
        f"현재 선택 모델은 {model}이며, API 키가 입력된 경우에만 호출합니다.",
    )
    if not api_key.strip():
        st.info("사이드바의 `AI 요약 설정`에서 OpenAI API 키를 입력하면 요약을 생성할 수 있습니다.")

    payload = build_llm_payload(
        selected_row,
        prediction_row,
        feature_map,
        local_shap,
        peer_slice,
        industry_latest_row,
    )

    cache_key = f"{selected_row['stock_code']}-{selected_row['fiscal_year']}-{model}-{selected_output_format}"
    if st.button("AI 요약 생성", type="primary"):
        if not api_key.strip():
            st.warning("API 키를 입력해야 AI 요약을 생성할 수 있습니다.")
        else:
            try:
                with st.spinner("AI가 심사 메모를 정리하는 중입니다..."):
                    explanation = generate_openai_explanation(
                        api_key=api_key.strip(),
                        model=model.strip(),
                        payload=payload,
                        output_format=selected_output_format,
                    )
                st.session_state[cache_key] = explanation
            except Exception as error:  # pragma: no cover - runtime/network dependent
                st.error(f"AI 요약 생성 중 오류가 발생했습니다: {error}")

    cached = st.session_state.get(cache_key)
    if cached:
        st.success("요약 생성 완료")
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
            model=model,
            output_format_label=output_format_label,
            report_text=cached,
            local_shap=local_shap,
            peer_slice=peer_slice,
            feature_map=feature_map,
        )
        onepage_text = build_onepage_llm_report(
            selected_row=selected_row,
            prediction_row=prediction_row,
            model=model,
            output_format_label=output_format_label,
            sections=sections,
            local_shap=local_shap,
            peer_slice=peer_slice,
            feature_map=feature_map,
        )
        html_report = build_html_report(
            selected_row=selected_row,
            prediction_row=prediction_row,
            model=model,
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
            model=model,
            output_format_label=output_format_label,
            sections=sections,
            local_shap=local_shap,
            peer_slice=peer_slice,
            feature_map=feature_map,
        )
        html_col1, html_col2 = st.columns([1, 1])
        with html_col1:
            st.download_button(
                "보고서형 HTML 다운로드",
                data=html_report,
                file_name=f"ts2000_credit_report_{selected_row['stock_code']}_{selected_row['fiscal_year']}.html",
                mime="text/html",
                use_container_width=True,
            )
        with html_col2:
            st.download_button(
                "원페이지 HTML 다운로드",
                data=onepage_html,
                file_name=f"ts2000_credit_onepage_{selected_row['stock_code']}_{selected_row['fiscal_year']}.html",
                mime="text/html",
                use_container_width=True,
            )
        utility_col1, utility_col2 = st.columns([1, 1])
        with utility_col1:
            st.download_button(
                "상세 보고서형 다운로드 (.md)",
                data=export_text,
                file_name=f"ts2000_credit_report_{selected_row['stock_code']}_{selected_row['fiscal_year']}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with utility_col2:
            st.download_button(
                "원페이지 요약 다운로드 (.md)",
                data=onepage_text,
                file_name=f"ts2000_credit_onepage_{selected_row['stock_code']}_{selected_row['fiscal_year']}.md",
                mime="text/markdown",
                use_container_width=True,
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
                    f"<div style='padding:1rem 1.05rem;border-radius:14px;"
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
    st.subheader("핵심 설명 변수")
    intro_col1, intro_col2, intro_col3 = st.columns(3)
    render_text_card(
        intro_col1,
        "분석 기준",
        "모델이 이 기업을 위험하게 또는 안정적으로 본 핵심 변수를 설명합니다.",
    )
    render_text_card(
        intro_col2, "위험 증가", "SHAP 값이 양수이면 투기등급 확률을 높이는 방향으로 작용합니다."
    )
    render_text_card(
        intro_col3, "위험 완화", "SHAP 값이 음수이면 위험을 낮추는 방향으로 작용합니다."
    )
    if artifacts.local_shap is not None:
        matched = resolve_company_local_shap(selected_row, artifacts.local_shap)
        if not matched.empty:
            st.success("기업별 local SHAP가 연결되어 있습니다.")
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
            st.altair_chart(chart, use_container_width=True)
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
                local_table.style.applymap(style_direction_badge, subset=["일반 해석 방향"])
                .set_properties(subset=["일반 해석 방향"], **{"text-align": "center"})
                .hide(axis="index")
            )
            st.dataframe(styled_local, use_container_width=True, hide_index=True)
            return

    st.info(
        "현재는 global SHAP 기준으로 설명 변수를 보여줍니다. "
        "기업별 local SHAP 파일이 추가되면 이 탭은 자동으로 해당 기업의 local drivers를 표시합니다."
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
        "기업별 local SHAP가 없어서 global SHAP 기준으로 보여줍니다.",
        COLOR_NEUTRAL,
    )
    render_accent_summary_card(
        summary_col2,
        "상위 설명축",
        str(top_features.iloc[0]["feature_group"]) if not top_features.empty else "없음",
        "공식 Core29 전체에서 평균적으로 크게 작용하는 변수군입니다.",
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
    st.altair_chart(chart, use_container_width=True)
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
        global_table.style.applymap(style_direction_badge, subset=["일반 해석 방향"])
        .set_properties(subset=["일반 해석 방향"], **{"text-align": "center"})
        .hide(axis="index")
    )
    st.dataframe(styled_global, use_container_width=True, hide_index=True)


def render_peer_tab(
    selected_row: pd.Series,
    artifacts: DashboardArtifacts,
) -> None:
    """Render the peer comparison tab."""
    st.subheader("시장/산업 비교")
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

    compare_features = st.multiselect(
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
        gap_text = format_delta_with_unit(abs(industry_gap), str(row["unit"]))
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
        st.altair_chart(compare_chart, use_container_width=True)
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
                st.altair_chart(mini_chart, use_container_width=True)

    table["산업 대비 차이"] = table.apply(
        lambda row: format_delta_with_unit(
            row["industry_gap"],
            str(row["unit"]),
        ),
        axis=1,
    )
    table["시장 대비 차이"] = table.apply(
        lambda row: format_delta_with_unit(
            row["market_gap"],
            str(row["unit"]),
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
                    "값_표시": format_delta_with_unit(row["industry_gap"], unit),
                },
                {
                    "구분": label,
                    "비교": "시장 대비 차이",
                    "값": market_gap_value,
                    "값_표시": format_delta_with_unit(row["market_gap"], unit),
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
            st.altair_chart(gap_chart, use_container_width=True)
        else:
            st.caption("단위가 섞여 있어 차이는 표에서 변수별로 읽는 것이 더 적절합니다.")
        st.caption("0보다 크면 선택 기업 값이 비교 기준보다 높고, 0보다 작으면 낮습니다.")
    with col_percentile:
        st.markdown("**산업/시장 내 백분위 위치**")
        st.altair_chart(percentile_chart, use_container_width=True)
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
        table_view.style.applymap(style_direction_badge, subset=["일반 해석 방향"])
        .set_properties(subset=["일반 해석 방향"], **{"text-align": "center"})
        .hide(axis="index")
    )
    st.dataframe(
        styled_table,
        use_container_width=True,
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
    st.subheader("산업별 집계")
    st.caption("선택한 기업이 속한 시장/산업을 기준으로 최신 스냅샷과 연도별 추이를 보여줍니다.")
    intro_col1, intro_col2, intro_col3 = st.columns(3)
    render_text_card(
        intro_col1, "집계 기준", "선택 기업과 같은 시장·산업에 속한 기업들을 기준으로 집계합니다."
    )
    render_text_card(
        intro_col2,
        "최신 스냅샷",
        "기업별 가장 최근 연도 1행만 남겨 현재 시점 산업 분위기를 보여줍니다.",
    )
    render_text_card(
        intro_col3,
        "연도별 추이",
        "연도별 평균 위험확률과 실제 투기등급 비율, 그리고 조정 기준선 적용 시 고위험 판정 비중을 함께 확인합니다.",
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
        "현재 산업 상태",
        format_percent(latest_row.get("mean_prob_speculative")),
        "선택 기업과 같은 시장·산업의 최신 평균 위험확률입니다.",
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
        "산업 기준 주요 변수",
        display_name(
            str(shap_summary.iloc[0]["feature"]),
            build_company_feature_map(selected_row, artifacts.feature_dictionary),
        )
        if shap_summary is not None and not shap_summary.empty
        else "없음",
        "test 구간 산업 SHAP 기준으로 가장 먼저 확인되는 설명 변수입니다.",
        COLOR_COMPANY,
    )

    c1, c2, c3, c4 = st.columns(4)
    render_bold_value_block(c1, "산업 기업 수", format_scalar(latest_row.get("companies")))
    render_bold_value_block(
        c2, "산업 평균 확률", format_percent(latest_row.get("mean_prob_speculative"))
    )
    render_bold_value_block(
        c3, "산업 중앙 확률", format_percent(latest_row.get("median_prob_speculative"))
    )
    render_bold_value_block(
        c4, tuned_share_label, format_percent(latest_row.get("pred_share_tuned"))
    )

    st.caption(
        f"{to_market_label(market)} / {to_industry_label(industry)} 기준 최신 기업 스냅샷 집계입니다."
    )
    st.info(
        "여기서 말하는 '조정 기준선'은 기본 0.5 대신, 검증 구간에서 precision과 recall 균형을 고려해 정한 판정 기준선을 뜻합니다."
    )

    year_summary = artifacts.industry_year_summary.loc[
        (artifacts.industry_year_summary["market"] == market)
        & (artifacts.industry_year_summary["industry_macro_category"] == industry)
    ].copy()
    if not year_summary.empty:
        st.subheader("연도별 산업 추이")
        trend_long = year_summary.melt(
            id_vars=["fiscal_year"],
            value_vars=["mean_prob_speculative", "pred_share_tuned", "positive_rate"],
            var_name="지표",
            value_name="값",
        )
        trend_long["지표"] = trend_long["지표"].replace(
            {
                "mean_prob_speculative": "평균 위험확률",
                "pred_share_tuned": tuned_share_label,
                "positive_rate": "실제 투기등급 비율",
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
                        domain=["평균 위험확률", tuned_share_label, "실제 투기등급 비율"],
                        range=[COLOR_RISK, COLOR_NEUTRAL, COLOR_DARK],
                    ),
                ),
                tooltip=["fiscal_year:O", "지표:N", alt.Tooltip("값:Q", format=".2%")],
            )
            .properties(height=320)
        )
        st.altair_chart(trend_chart, use_container_width=True)
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
                "mean_prob_speculative": "평균 위험확률",
                "median_prob_speculative": "중앙 위험확률",
                "pred_share_0_5": default_share_label,
                "pred_share_tuned": tuned_share_label,
            }
        )
        year_summary_view["시장"] = year_summary_view["시장"].map(to_market_label)
        year_summary_view["산업"] = year_summary_view["산업"].map(to_industry_label)
        st.dataframe(
            year_summary_view.loc[
                :,
                [
                    "시장",
                    "산업",
                    "회계연도",
                    "기업 수",
                    "투기등급 기업 수",
                    "투기등급 비율",
                    "평균 위험확률",
                    "중앙 위험확률",
                    default_share_label,
                    tuned_share_label,
                ],
            ],
            use_container_width=True,
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
        st.altair_chart(chart, use_container_width=True)
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
            top_shap_view.style.applymap(style_direction_badge, subset=["일반 해석 방향"])
            .set_properties(subset=["일반 해석 방향"], **{"text-align": "center"})
            .hide(axis="index")
        )
        st.dataframe(styled_industry, use_container_width=True, hide_index=True)


def render_scenario_tab(
    selected_row: pd.Series,
    artifacts: DashboardArtifacts,
) -> None:
    """Render the scenario tab."""
    st.subheader("시나리오 분석")
    presets = list(artifacts.scenario_presets.keys())
    preset_label_map = {
        "base": "기본",
        "mild_stress": "완만한 스트레스",
        "severe_stress": "강한 스트레스",
    }
    selected_preset = st.selectbox(
        "프리셋", presets, format_func=lambda value: preset_label_map.get(value, value)
    )
    preset_changes = artifacts.scenario_presets[selected_preset]
    intro_col1, intro_col2, intro_col3 = st.columns(3)
    render_text_card(
        intro_col1,
        "선택한 시나리오",
        f"현재 선택한 시나리오는 {preset_label_map.get(selected_preset, selected_preset)}입니다.",
    )
    render_text_card(
        intro_col2,
        "시나리오 적용 방식",
        "핵심 지표 값을 가정적으로 바꿔 보고, 상대적 위치가 어떻게 달라지는지 확인합니다.",
    )
    render_text_card(
        intro_col3,
        "해석 시 유의점",
        "현재는 예측확률을 다시 계산하는 단계가 아니라, 지표 수준 변화와 백분위 이동을 중심으로 보여줍니다.",
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
            f"{label} 변화량",
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
                "시나리오 적용 후 대략적 백분위": scenario_percentile,
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
    scenario_frame["시나리오 적용 후 위치"] = scenario_frame["시나리오 적용 후 대략적 백분위"].map(
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
        "슬라이더 초기값은 이 프리셋을 기준으로 자동 채워집니다.",
        COLOR_NEUTRAL,
    )
    render_accent_summary_card(
        summary_col2,
        "시나리오에서 가장 많이 바뀐 지표",
        str(strongest_change["변수"]) if strongest_change is not None else "없음",
        format_delta_with_unit(strongest_change["변화량"], strongest_change["unit"])
        if strongest_change is not None
        else "-",
        COLOR_RISK,
    )
    render_accent_summary_card(
        summary_col3,
        "시나리오 조정 가능 지표 수",
        format_scalar(len(scenario_frame)),
        "현재 화면에서 직접 조정 가능한 핵심 변수 개수입니다.",
        COLOR_COMPANY,
    )
    st.markdown("**시나리오 적용 전후 비교**")
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
                        "구분": "현재값",
                        "값": current_value,
                        "값_표시": row["현재값_표시"],
                    },
                    {
                        "변수": row["변수"],
                        "구분": "시나리오 조정값",
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
                        domain=["현재값", "시나리오 조정값"], range=[COLOR_MUTED, COLOR_RISK]
                    ),
                ),
                xOffset="구분:N",
                tooltip=["변수:N", "구분:N", alt.Tooltip("값_표시:N", title="값")],
            )
            .properties(height=max(160, len(unit_frame) * 56))
        )
        st.altair_chart(scenario_chart, use_container_width=True)
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
        scenario_table.style.applymap(style_direction_badge, subset=["일반 해석 방향"])
        .set_properties(subset=["일반 해석 방향"], **{"text-align": "center"})
        .hide(axis="index")
    )
    st.dataframe(
        styled_scenario,
        use_container_width=True,
        hide_index=True,
    )
    st.warning(
        "현재 시나리오 탭은 지표를 조정했을 때 상대 위치가 어떻게 달라지는지 보여줍니다. "
        "기업별 예측확률을 다시 계산하는 기능은 다음 단계에서 추가할 수 있습니다."
    )


def render_footer(artifacts: DashboardArtifacts, *, developer_mode: bool) -> None:
    """Render footer metadata."""
    if developer_mode:
        with st.expander("LLM payload template 보기"):
            st.json(artifacts.llm_payload_template)
        with st.expander("Export manifest 보기"):
            st.json(artifacts.export_manifest)


def main() -> None:
    """Run the TS2000 Streamlit dashboard MVP."""
    st.set_page_config(page_title="기업 신용위험 인텔리전스 대시보드", layout="wide")
    st.title("기업 신용위험 인텔리전스 대시보드")
    st.caption(
        "기업별 위험 진단, 동종업계 비교, 산업 집계, AI 심사 메모를 한 화면에서 확인하는 설명형 대시보드입니다. "
        "앞으로 뉴스 리포트와 에이전트 협의 결과까지 함께 연결할 수 있도록 확장할 예정입니다."
    )
    st.markdown(
        """
        <style>
        @media (max-width: 900px) {
          div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    artifact_dir_input = st.sidebar.text_input(
        "아티팩트 경로",
        value="",
        help="비워두면 기본 경로(data/outputs/dashboard/ts2000_core29_mvp)를 사용합니다.",
    )
    st.sidebar.selectbox(
        "금액 표시 방식",
        options=list(MONEY_DISPLAY_MODES.keys()),
        index=0,
        format_func=lambda value: MONEY_DISPLAY_MODES.get(value, value),
        key="money_display_mode",
        help="상세 표기(억·만·원)와 단순 표기(억 원) 중 원하는 방식을 선택합니다.",
    )
    developer_mode = st.sidebar.checkbox(
        "개발자 모드",
        value=False,
        help="개발/디버깅용 메타정보와 payload를 표시합니다.",
    )
    default_api_key = os.environ.get("OPENAI_API_KEY", "")
    default_model = os.environ.get("OPENAI_MODEL", "gpt-5.5")
    model_options = [item[0] for item in RECOMMENDED_LLM_MODELS]
    model_labels = {item[0]: item[1] for item in RECOMMENDED_LLM_MODELS}
    default_model_value = default_model if default_model in model_options else model_options[0]
    with st.sidebar.expander("AI 요약 설정", expanded=False):
        api_key = st.text_input(
            "OpenAI API 키",
            value=default_api_key,
            type="password",
            help="로컬 실행 시에만 사용되며 코드나 파일에는 저장하지 않습니다.",
        )
        selected_model = st.selectbox(
            "추천 모델",
            options=model_options,
            index=model_options.index(default_model_value),
            format_func=lambda value: model_labels.get(value, value),
        )
        custom_model = st.text_input(
            "직접 입력할 모델명 (선택)",
            value="",
            placeholder=selected_model if default_model in model_options else default_model,
            help="필요할 때만 영문 모델 ID를 직접 입력합니다. 비워두면 위 추천 모델을 사용합니다.",
        )
        llm_model = custom_model.strip() or selected_model
        st.caption("API 키는 세션 중 메모리에서만 사용하며 파일에는 저장하지 않습니다.")

    try:
        artifacts = cached_load_dashboard_artifacts(artifact_dir_input or None)
    except FileNotFoundError as error:
        st.error(f"대시보드 입력 아티팩트를 찾을 수 없습니다: {error}")
        st.stop()

    selected_row = pick_selected_company(artifacts)
    prediction_row = resolve_company_prediction(selected_row, artifacts.prediction_scores)
    feature_map = build_company_feature_map(selected_row, artifacts.feature_dictionary)
    local_shap = resolve_company_local_shap(selected_row, artifacts.local_shap)
    peer_slice = resolve_company_peer_slice(selected_row, artifacts.peer_percentiles)
    industry_latest_row = resolve_industry_latest_row(
        selected_row, artifacts.industry_latest_summary
    )

    overview_tab, report_tab, drivers_tab, peers_tab, industry_tab, scenario_tab = st.tabs(
        ["개요", "AI 심사 요약", "주요 요인", "동종업계 비교", "산업 집계", "시나리오"]
    )

    with overview_tab:
        render_overview_tab(
            selected_row, prediction_row, artifacts.model_summary, feature_map, artifacts
        )
    with report_tab:
        render_llm_panel(
            selected_row=selected_row,
            prediction_row=prediction_row,
            feature_map=feature_map,
            local_shap=local_shap,
            peer_slice=peer_slice,
            industry_latest_row=industry_latest_row,
            api_key=api_key,
            model=llm_model,
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

    render_footer(artifacts, developer_mode=developer_mode)


if __name__ == "__main__":
    main()
