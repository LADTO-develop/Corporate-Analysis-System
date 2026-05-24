"""Landing-page company search and selector rendering for the dashboard."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from html import escape

import pandas as pd
import streamlit as st

from cas.dashboard.data_loader import DashboardArtifacts

LANDING_RECOMMENDATION_ORDER = [
    "committee_caught",
    "risk_detected",
    "stable_confirmed",
    "overwarning_softened",
]

LANDING_RECOMMENDATION_LABELS = {
    "committee_caught": "추천 기업",
    "risk_detected": "추천 기업",
    "stable_confirmed": "추천 기업",
    "overwarning_softened": "추천 기업",
}

LANDING_RECOMMENDATION_HELPERS = {
    "committee_caught": "위원회 검토 화면의 흐름을 살펴보기 좋은 예시입니다.",
    "risk_detected": "모델 판단과 외부 근거가 함께 정리되는 방식을 확인할 수 있습니다.",
    "stable_confirmed": "신용도 해석이 어떻게 요약되는지 가볍게 둘러볼 수 있습니다.",
    "overwarning_softened": "위원회 검토가 근거를 어떻게 나누어 설명하는지 확인할 수 있습니다.",
}


@dataclass(frozen=True, slots=True)
class LandingPageFormatters:
    """Display helpers supplied by the host dashboard."""

    stock_code_text: Callable[[object], str]
    format_percent: Callable[[object], str]
    format_scalar: Callable[[object], str]
    to_industry_display_label: Callable[[object], str]
    to_market_display_label: Callable[[object], str]
    to_prediction_label: Callable[[object], str]
    to_size_label: Callable[[object], str]


def pick_selected_company(
    artifacts: DashboardArtifacts,
    *,
    formatters: LandingPageFormatters,
) -> pd.Series:
    """Render company selectors and return the chosen company snapshot."""
    return pick_selected_company_from_market_explorer(artifacts, formatters=formatters)


def pick_selected_company_from_market_explorer(
    artifacts: DashboardArtifacts,
    *,
    formatters: LandingPageFormatters,
) -> pd.Series:
    """Render the market-style company selector and return the chosen company snapshot."""
    explorer_frame = build_company_explorer_frame(artifacts, formatters=formatters)
    if explorer_frame.empty:
        st.warning("분석 가능한 종목 데이터가 없습니다. 대시보드 산출물을 다시 확인해주세요.")
        st.stop()

    current_key = str(st.session_state.get("selected_company_key", ""))
    if current_key:
        matched = explorer_frame.loc[explorer_frame["_company_key"] == current_key]
        if not matched.empty:
            selected_row = matched.iloc[0]
            render_selected_company_detail_header(selected_row, formatters=formatters)
            return selected_row
        st.session_state.pop("selected_company_key", None)

    selected_key = render_company_market_explorer(explorer_frame, formatters=formatters)
    if not selected_key:
        st.info("상단 검색창에서 기업을 선택하면 상세 분석 화면이 열립니다.")
        st.stop()

    matched = explorer_frame.loc[explorer_frame["_company_key"] == selected_key]
    if matched.empty:
        st.session_state.pop("selected_company_key", None)
        st.warning("선택한 종목을 현재 산출물에서 찾을 수 없습니다. 다시 선택해주세요.")
        st.stop()
    return matched.iloc[0]


def render_selected_company_detail_header(
    selected_row: pd.Series,
    *,
    formatters: LandingPageFormatters,
) -> None:
    """Render detail-page navigation once a company has been selected."""
    if st.session_state.pop("scroll_detail_top_once", False):
        st.components.v1.html(
            """
            <script>
            const doc = window.parent.document;
            const main = doc.querySelector('[data-testid="stMain"]');
            if (main && typeof main.scrollTo === 'function') {
              main.scrollTo({ top: 0, behavior: 'smooth' });
            }
            window.parent.scrollTo({ top: 0, behavior: 'smooth' });
            </script>
            """,
            height=0,
        )

    nav_col, title_col = st.columns([0.16, 0.84])
    with nav_col:
        if st.button("← 다른 기업 찾기", use_container_width=True):
            st.session_state.pop("selected_company_key", None)
            st.rerun()

    market = selected_row.get("_display_market") or formatters.to_market_display_label(
        selected_row.get("market")
    )
    industry = selected_row.get("_display_industry") or formatters.to_industry_display_label(
        selected_row.get("industry_macro_category")
    )
    stock_code = formatters.stock_code_text(selected_row.get("stock_code"))
    size_label = selected_row.get("_display_size") or formatters.to_size_label(
        selected_row.get("firm_size_group")
    )
    review_request_key = _company_review_request_key(selected_row, stock_code)

    with title_col:
        info_col, action_col = st.columns([0.64, 0.36], gap="medium")
        with info_col:
            st.markdown(
                (
                    "<div class='selected-company-hero'>"
                    "<div>"
                    "<div class='selected-company-eyebrow'>기업 신용도 해석</div>"
                    f"<div class='selected-company-title'>{escape(str(selected_row.get('corp_name') or '-'))}</div>"
                    "<div class='selected-company-subtitle'>"
                    "선택한 기업의 기본 정보를 먼저 보여드립니다. 아래 탭에서 위원회 검토, "
                    "재무 핵심, 시장·산업 비교를 이어서 확인할 수 있어요."
                    "</div>"
                    "<div class='selected-company-chip-row'>"
                    f"<span class='selected-company-chip'>{escape(str(market))}</span>"
                    f"<span class='selected-company-chip'>{escape(stock_code)}</span>"
                    f"<span class='selected-company-chip'>{escape(str(industry))}</span>"
                    f"<span class='selected-company-chip'>{escape(str(size_label))}</span>"
                    "</div>"
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        with action_col:
            st.markdown(
                (
                    "<div class='selected-company-action-panel'>"
                    "<div class='selected-company-action-label'>정밀 AI 검토</div>"
                    "<div class='selected-company-action-title'>뉴스·공시까지 다시 확인</div>"
                    "<div class='selected-company-action-body'>"
                    "버튼을 누르면 AI 위원회가 외부 근거를 다시 읽고, 아래 위원회 검토 탭에 "
                    "정밀 검토 상태와 결과를 표시합니다."
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            if st.button(
                "정밀 검토 실행",
                key=f"selected_company_stage2_start_{review_request_key}",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state["dashboard_stage2_header_start_request"] = review_request_key
                st.rerun()


def build_company_explorer_frame(
    artifacts: DashboardArtifacts,
    *,
    formatters: LandingPageFormatters,
) -> pd.DataFrame:
    """Build the selectable market overview frame for the dashboard landing area."""
    latest = artifacts.company_latest.copy()
    latest["_stock_code_text"] = latest["stock_code"].map(formatters.stock_code_text)
    latest["_company_key"] = latest.apply(
        lambda row: _company_selection_key(row, formatters=formatters),
        axis=1,
    )

    if artifacts.prediction_scores is not None:
        prediction_frame = artifacts.prediction_scores.copy()
        prediction_frame["_stock_code_text"] = prediction_frame["stock_code"].map(
            formatters.stock_code_text
        )
        prediction_columns = [
            "_stock_code_text",
            "fiscal_year",
            "prob_speculative",
            "predicted_label",
            "risk_band",
            "stage2_review_priority",
            "trigger_reason",
            "external_validation_stage2_effect",
            "external_validation_actual_label",
            "external_validation_credit_rating",
            "external_validation_committee_label",
            "landing_recommendation_bucket",
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
    latest["_display_market"] = latest["market"].map(formatters.to_market_display_label)
    latest["_display_industry"] = latest["industry_macro_category"].map(
        formatters.to_industry_display_label
    )
    latest["_display_size"] = latest["firm_size_group"].map(formatters.to_size_label)
    latest["_display_probability"] = latest["prob_speculative"].map(formatters.format_percent)
    latest["_display_label"] = latest["predicted_label"].map(formatters.to_prediction_label)
    latest["_search_label"] = latest.apply(
        lambda row: _company_search_label(row, formatters=formatters),
        axis=1,
    )
    return latest.sort_values(["market", "corp_name", "fiscal_year"]).reset_index(drop=True)


def _company_selection_key(row: pd.Series, *, formatters: LandingPageFormatters) -> str:
    """Build a stable selection key for one company-year row."""
    fiscal_year = row.get("fiscal_year")
    try:
        fiscal_year_text = str(int(float(str(fiscal_year))))
    except (TypeError, ValueError):
        fiscal_year_text = str(fiscal_year)
    return f"{row.get('market')}-{formatters.stock_code_text(row.get('stock_code'))}-{fiscal_year_text}"


def _company_review_request_key(row: pd.Series, stock_code_text: str) -> str:
    """Build a small key used to request precise review from the detail header."""
    fiscal_year = row.get("fiscal_year")
    try:
        fiscal_year_text = str(int(float(str(fiscal_year))))
    except (TypeError, ValueError):
        fiscal_year_text = str(fiscal_year)
    return f"{stock_code_text}:{fiscal_year_text}"


def _company_search_label(row: pd.Series, *, formatters: LandingPageFormatters) -> str:
    """Return the searchable option label shown in the top selectbox."""
    return (
        f"{row.get('corp_name')} · {formatters.stock_code_text(row.get('stock_code'))} · "
        f"{row.get('_display_market')} · {row.get('_display_industry')} · "
        f"FY{formatters.format_scalar(row.get('fiscal_year'))}"
    )


def render_company_market_explorer(
    explorer_frame: pd.DataFrame,
    *,
    formatters: LandingPageFormatters,
) -> str | None:
    """Render a compact company search landing page."""
    current_key = str(st.session_state.get("selected_company_key", ""))
    all_valid_keys = explorer_frame["_company_key"].astype(str).tolist()
    if current_key and current_key not in all_valid_keys:
        current_key = ""
        st.session_state.pop("selected_company_key", None)

    st.markdown(
        """
        <div class="market-search-panel">
          <div class="market-search-eyebrow">KOSPI · KOSDAQ LISTED COMPANIES</div>
          <h2>어떤 기업의 신용도를 확인할까요?</h2>
          <p>코스피·코스닥 상장기업을 대상으로 재무 데이터, 모델 판단, 뉴스·공시 근거를 함께 살펴봅니다.
          기업을 선택하면 AI 에이전트 위원회가 위험 신호와 완화 근거를 함께 검토해,
          왜 그런 판단이 나왔는지 쉽게 풀어드립니다.</p>
          <div class="market-search-chips">
            <span class="market-search-chip">상장기업 검색</span>
            <span class="market-search-chip">재무·거시 기반 판단</span>
            <span class="market-search-chip">뉴스·공시 근거 검토</span>
            <span class="market-search-chip">에이전트 위원회 의견</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filter_col1, filter_col2 = st.columns(2)
    market_options = ["전체", "KOSPI", "KOSDAQ"]
    selected_market = filter_col1.selectbox(
        "시장 구분",
        options=market_options,
        format_func=lambda value: str(value),
        key="landing_market_filter_v2",
        help="코스피와 코스닥 중 분석할 시장을 먼저 좁혀볼 수 있습니다.",
    )
    filtered_frame = explorer_frame.copy()
    if selected_market != "전체":
        filtered_frame = filtered_frame.loc[filtered_frame["market"].astype(str) == selected_market]

    industry_options = [
        "전체",
        *sorted(filtered_frame["industry_macro_category"].dropna().astype(str).unique().tolist()),
    ]
    selected_industry = filter_col2.selectbox(
        "산업군",
        options=industry_options,
        format_func=lambda value: "전체"
        if value == "전체"
        else formatters.to_industry_display_label(value),
        key="landing_industry_filter_v2",
        help="선택한 시장 안에서 산업군을 한 번 더 좁혀볼 수 있습니다.",
    )
    if selected_industry != "전체":
        filtered_frame = filtered_frame.loc[
            filtered_frame["industry_macro_category"].astype(str) == selected_industry
        ]

    if filtered_frame.empty:
        st.info("선택한 시장/산업 조건에 맞는 기업이 없습니다. 필터를 조정해 주세요.")
        return str(st.session_state.get("selected_company_key", ""))

    selected_market_label = "전체 시장" if selected_market == "전체" else str(selected_market)
    selected_industry_label = (
        "전체 산업"
        if selected_industry == "전체"
        else formatters.to_industry_display_label(selected_industry)
    )
    st.markdown(
        (
            "<div class='landing-filter-summary'>"
            f"현재 <b>{len(filtered_frame):,}개</b> 기업이 선택 조건에 맞습니다. "
            f"필터 기준은 <b>{escape(selected_market_label)}</b>, "
            f"<b>{escape(selected_industry_label)}</b>입니다."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    valid_keys = filtered_frame["_company_key"].astype(str).tolist()
    option_labels = dict(
        zip(
            filtered_frame["_company_key"].astype(str),
            filtered_frame["_search_label"].astype(str),
            strict=False,
        )
    )
    selected_index = valid_keys.index(current_key) if current_key in valid_keys else None
    selected_from_search = st.selectbox(
        "기업명 또는 종목코드로 찾기",
        options=valid_keys,
        index=selected_index,
        format_func=lambda value: option_labels.get(str(value), str(value)),
        placeholder="예: 삼성전자, 005930",
        help="기업명을 입력하거나 종목코드를 입력해 원하는 기업을 바로 선택할 수 있습니다.",
    )
    if selected_from_search and selected_from_search != current_key:
        st.session_state["selected_company_key"] = str(selected_from_search)
        st.session_state["scroll_detail_top_once"] = True
        st.rerun()

    has_validation_recommendations = has_landing_validation_recommendations(filtered_frame)
    if has_validation_recommendations:
        section_title = "가볍게 둘러보기"
        section_caption = (
            "아직 특정 기업이 정해지지 않았다면, 아래 기업을 눌러 "
            "신용도 해석 화면이 어떻게 구성되는지 먼저 확인해보세요."
        )
    else:
        section_title = "가볍게 둘러보기"
        section_caption = (
            "아직 특정 기업이 정해지지 않았다면, 아래 추천 기업을 눌러 "
            "위원회 검토 화면의 흐름을 먼저 확인해보세요."
        )
    st.markdown(
        (
            f'<div class="landing-section-title">{escape(section_title)}</div>'
            f'<div class="landing-section-caption">{escape(section_caption)}</div>'
        ),
        unsafe_allow_html=True,
    )
    render_random_company_cards(filtered_frame, formatters=formatters)
    return str(st.session_state.get("selected_company_key", ""))


def render_top_unsuitable_companies(
    explorer_frame: pd.DataFrame,
    *,
    formatters: LandingPageFormatters,
) -> None:
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
        render_unsuitable_company_card(
            container,
            row,
            badge=f"위험 상위 {index}",
            formatters=formatters,
        )
        if container.button(
            "분석 보기",
            key=f"top_unsuitable_{row['_company_key']}",
            use_container_width=True,
        ):
            st.session_state["selected_company_key"] = str(row["_company_key"])
            st.session_state["scroll_detail_top_once"] = True
            st.rerun()


def render_low_risk_companies(
    explorer_frame: pd.DataFrame,
    *,
    formatters: LandingPageFormatters,
) -> None:
    """Render low-risk companies as a contrast set for demo exploration."""
    ranked = (
        explorer_frame.dropna(subset=["_prob_speculative_number"])
        .sort_values("_prob_speculative_number", ascending=True)
        .head(3)
    )
    if ranked.empty:
        return

    st.markdown("### 안정 가능성 참고 기업")
    columns = st.columns(3)
    for index, row in enumerate(ranked.to_dict(orient="records"), start=1):
        container = columns[index - 1]
        render_unsuitable_company_card(
            container,
            row,
            badge=f"안정 참고 {index}",
            formatters=formatters,
        )
        if container.button(
            "분석 보기",
            key=f"low_risk_{row['_company_key']}",
            use_container_width=True,
        ):
            st.session_state["selected_company_key"] = str(row["_company_key"])
            st.session_state["scroll_detail_top_once"] = True
            st.rerun()


def has_landing_validation_recommendations(explorer_frame: pd.DataFrame) -> bool:
    """Return whether current filters include externally validated 2026 examples."""
    if "landing_recommendation_bucket" not in explorer_frame.columns:
        return False
    buckets = explorer_frame["landing_recommendation_bucket"].fillna("").astype(str).str.strip()
    return bool(buckets.ne("").any())


def select_landing_recommendation_rows(explorer_frame: pd.DataFrame) -> pd.DataFrame:
    """Prefer 2026 external-validation examples, with a deterministic fallback sample."""
    if explorer_frame.empty:
        return explorer_frame

    selected_rows: list[pd.Series] = []
    used_keys: set[str] = set()
    if "landing_recommendation_bucket" in explorer_frame.columns:
        for bucket in LANDING_RECOMMENDATION_ORDER:
            bucket_frame = explorer_frame.loc[
                explorer_frame["landing_recommendation_bucket"].fillna("").astype(str).eq(bucket)
            ].copy()
            if bucket_frame.empty:
                continue
            ascending = bucket in {"stable_confirmed", "overwarning_softened"}
            bucket_frame = bucket_frame.sort_values(
                ["_prob_speculative_number", "corp_name"],
                ascending=[ascending, True],
                na_position="last",
            )
            row = bucket_frame.iloc[0]
            key = str(row.get("_company_key"))
            if key not in used_keys:
                selected_rows.append(row)
                used_keys.add(key)
            if len(selected_rows) >= 3:
                break

    if len(selected_rows) < min(3, len(explorer_frame)):
        fallback = explorer_frame.loc[
            ~explorer_frame["_company_key"].astype(str).isin(used_keys)
        ].sample(
            n=min(3 - len(selected_rows), len(explorer_frame) - len(selected_rows)),
            random_state=43,
        )
        selected_rows.extend([row for _, row in fallback.iterrows()])

    if not selected_rows:
        return explorer_frame.head(0)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def render_random_company_cards(
    explorer_frame: pd.DataFrame,
    *,
    formatters: LandingPageFormatters,
) -> None:
    """Render curated validation examples or a deterministic sample for exploration."""
    if explorer_frame.empty:
        return
    sampled = select_landing_recommendation_rows(explorer_frame)
    sample_size = len(sampled)
    if sample_size == 0:
        return

    columns = st.columns(sample_size)
    for index, row in enumerate(sampled.to_dict(orient="records"), start=1):
        container = columns[index - 1]
        bucket = str(row.get("landing_recommendation_bucket") or "").strip()
        row["_explore_helper_text"] = LANDING_RECOMMENDATION_HELPERS.get(
            bucket,
            "선택 후 위원회 검토에서 모델 판단과 외부근거를 함께 확인합니다.",
        )
        render_unsuitable_company_card(
            container,
            row,
            badge=LANDING_RECOMMENDATION_LABELS.get(bucket, "추천 기업"),
            card_tone="explore",
            show_model_signal=False,
            formatters=formatters,
        )
        if container.button(
            "위원회 검토 보기",
            key=f"random_pick_{row['_company_key']}",
            use_container_width=True,
        ):
            st.session_state["selected_company_key"] = str(row["_company_key"])
            st.session_state["scroll_detail_top_once"] = True
            st.rerun()


def market_card_risk_tone_class(risk_band: object) -> str:
    """Return a CSS tone class for a dashboard landing-card risk band."""
    label = str(risk_band or "").strip()
    if label == "안정":
        return "stable"
    if label == "관찰":
        return "watch"
    if label == "고위험":
        return "high"
    return "neutral"


def render_unsuitable_company_card(
    container: st.delta_generator.DeltaGenerator,
    row: dict[str, object],
    *,
    badge: str,
    formatters: LandingPageFormatters,
    card_tone: str = "",
    show_model_signal: bool = True,
) -> None:
    """Render one company card in the selector landing page."""
    risk_band = row.get("risk_band") or "-"
    risk_tone_class = market_card_risk_tone_class(risk_band)
    probability = row.get("_display_probability") or "-"
    explore_helper_text = str(
        row.get("_explore_helper_text")
        or "선택 후 위원회 검토에서 모델 판단과 외부근거를 함께 확인합니다."
    )
    model_signal_html = (
        "<div class='market-card-prob-label'>1차 모델 투기등급 확률</div>"
        f"<div class='market-card-risk {risk_tone_class}'>{escape(str(probability))}</div>"
        f"<div class='market-card-band {risk_tone_class}'>위험 구간 {escape(str(risk_band))}</div>"
        if show_model_signal
        else (
            "<div class='market-card-meta' style='margin-top:0.55rem;'>"
            f"{escape(explore_helper_text)}"
            "</div>"
        )
    )
    container.markdown(
        (
            f"<div class='market-card {escape(card_tone)}'>"
            f"<div class='market-card-rank {escape(card_tone)}'>{escape(badge)}</div>"
            f"<div class='market-card-title'>{escape(str(row.get('corp_name') or '-'))}</div>"
            "<div class='market-card-meta'>"
            f"<span>{escape(str(row.get('_display_market') or '-'))}</span>"
            f"<span>·</span><span>{escape(formatters.stock_code_text(row.get('stock_code')))}</span>"
            f"<span>·</span><span>{escape(str(row.get('_display_industry') or '-'))}</span>"
            "</div>"
            f"{model_signal_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_market_company_lists(
    explorer_frame: pd.DataFrame,
    *,
    formatters: LandingPageFormatters,
) -> None:
    """Render separate selectable lists for KOSPI and KOSDAQ."""
    st.markdown("### 시장별 종목")
    kospi_col, kosdaq_col = st.columns(2)
    render_market_company_table(
        kospi_col,
        explorer_frame,
        market="KOSPI",
        formatters=formatters,
    )
    render_market_company_table(
        kosdaq_col,
        explorer_frame,
        market="KOSDAQ",
        formatters=formatters,
    )


def render_market_company_table(
    container: st.delta_generator.DeltaGenerator,
    explorer_frame: pd.DataFrame,
    *,
    market: str,
    formatters: LandingPageFormatters,
) -> None:
    """Render one market table and handle row selection."""
    market_frame = explorer_frame.loc[explorer_frame["market"].astype(str) == market].copy()
    market_frame = market_frame.sort_values(
        ["_prob_speculative_number", "corp_name"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    container.markdown(
        f"<div class='market-section-title'>{escape(formatters.to_market_display_label(market))}</div>",
        unsafe_allow_html=True,
    )
    if market_frame.empty:
        container.info(f"{formatters.to_market_display_label(market)} 종목이 없습니다.")
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
                st.session_state["scroll_detail_top_once"] = True
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
