"""Scenario tab rendering and relative-position calculations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import altair as alt
import pandas as pd
import streamlit as st

from cas.dashboard.cards import (
    render_accent_summary_card,
    render_text_card,
    style_direction_badge,
)
from cas.dashboard.chart_data import finite_chart_frame
from cas.dashboard.data_loader import DashboardArtifacts
from cas.dashboard.settings import COLOR_COMPANY, COLOR_MUTED, COLOR_NEUTRAL, COLOR_RISK
from cas.dashboard.streamlit_compat import stretch_altair_chart

DEFAULT_SCENARIO_FEATURES = (
    "spec_spread",
    "cash_ratio",
    "net_margin",
    "short_term_borrowings_share",
    "capital_impairment_ratio",
)


@dataclass(frozen=True)
class ScenarioTabFormatters:
    """Formatting callbacks supplied by the main dashboard module."""

    display_name: Callable[[str, pd.DataFrame], str]
    feature_unit: Callable[[str, pd.DataFrame], str]
    value_with_unit: Callable[[object, object, str | None], str]
    delta_with_unit: Callable[[object, object, str | None], str]
    percentile_label: Callable[[object], str]
    scalar: Callable[[object], str]
    feature_direction_label: Callable[[str], str]
    unit_description: Callable[[str], str]


def approximate_percentile(series: pd.Series, new_value: float) -> float | None:
    """Approximate percentile rank if a scenario changes one variable."""
    clean = series.dropna()
    if clean.empty or pd.isna(new_value):
        return None
    augmented = pd.concat([clean, pd.Series([new_value])], ignore_index=True)
    return float(augmented.rank(method="average", pct=True).iloc[-1] * 100.0)


def build_scenario_frame(
    *,
    selected_row: pd.Series,
    company_universe: pd.DataFrame,
    feature_map: pd.DataFrame,
    deltas: Mapping[str, float],
    formatters: ScenarioTabFormatters,
    scenario_features: Sequence[str] = DEFAULT_SCENARIO_FEATURES,
) -> pd.DataFrame:
    """Build the scenario comparison frame from selected feature deltas."""
    rows: list[dict[str, object]] = []
    for feature in scenario_features:
        baseline_value = selected_row.get(feature)
        scenario_value = _scenario_value(baseline_value, deltas.get(feature, 0.0))
        distribution = (
            company_universe.loc[:, feature]
            if feature in company_universe
            else pd.Series(dtype=float)
        )
        scenario_percentile = (
            approximate_percentile(distribution, scenario_value)
            if scenario_value is not None
            else None
        )
        rows.append(
            {
                "변수": formatters.display_name(feature, feature_map),
                "feature": feature,
                "현재값": baseline_value,
                "변화량": deltas.get(feature, 0.0),
                "시나리오 조정값": scenario_value,
                "시나리오 적용 후 대략적 위치": scenario_percentile,
                "unit": formatters.feature_unit(feature, feature_map),
                "일반 해석 방향": formatters.feature_direction_label(feature),
            }
        )

    scenario_frame = pd.DataFrame(rows)
    scenario_frame["현재값_표시"] = scenario_frame.apply(
        lambda row: formatters.value_with_unit(row["현재값"], row["unit"], str(row["feature"])),
        axis=1,
    )
    scenario_frame["시나리오 조정값_표시"] = scenario_frame.apply(
        lambda row: formatters.value_with_unit(
            row["시나리오 조정값"],
            row["unit"],
            str(row["feature"]),
        ),
        axis=1,
    )
    scenario_frame["시나리오 적용 후 위치"] = scenario_frame["시나리오 적용 후 대략적 위치"].map(
        formatters.percentile_label
    )
    return scenario_frame


type AltairChartLike = (
    alt.Chart | alt.LayerChart | alt.FacetChart | alt.VConcatChart | alt.HConcatChart
)


def apply_altair_cas_theme(
    chart: AltairChartLike,
    theme_mode: str = "light",
) -> AltairChartLike:
    """Apply CAS-controlled light/dark styling to Altair charts."""
    if theme_mode == "dark":
        return (
            chart.properties(background="#080b12")
            .configure_view(
                strokeOpacity=0,
                fill="#080b12",
            )
            .configure_axis(
                labelColor="#f8fafc",
                titleColor="#cbd5e1",
                gridColor="rgba(250, 204, 21, 0.14)",
                domainColor="rgba(250, 204, 21, 0.28)",
                tickColor="rgba(250, 204, 21, 0.28)",
            )
            .configure_legend(
                labelColor="#f8fafc",
                titleColor="#cbd5e1",
            )
            .configure_title(
                color="#f8fafc",
                subtitleColor="#cbd5e1",
            )
        )

    return (
        chart.properties(background="#ffffff")
        .configure_view(
            strokeOpacity=0,
            fill="#ffffff",
        )
        .configure_axis(
            labelColor="#334155",
            titleColor="#64748b",
            gridColor="rgba(148, 163, 184, 0.22)",
            domainColor="rgba(148, 163, 184, 0.34)",
            tickColor="rgba(148, 163, 184, 0.34)",
        )
        .configure_legend(
            labelColor="#334155",
            titleColor="#64748b",
        )
        .configure_title(
            color="#0f172a",
            subtitleColor="#64748b",
        )
    )


def _dashboard_table_palette(theme_mode: str = "light") -> dict[str, str]:
    """Return concrete table colors because pandas Styler does not reliably inherit CSS vars."""
    if str(theme_mode).lower() == "dark":
        return {
            "panel": "#111827",
            "panel_strong": "#172033",
            "card": "#0f172a",
            "text": "#f8fafc",
            "muted": "#cbd5e1",
            "border": "rgba(250, 204, 21, 0.28)",
            "border_soft": "rgba(250, 204, 21, 0.16)",
        }

    return {
        "panel": "#ffffff",
        "panel_strong": "#f8fafc",
        "card": "#ffffff",
        "text": "#0f172a",
        "muted": "#475569",
        "border": "rgba(148, 163, 184, 0.34)",
        "border_soft": "rgba(148, 163, 184, 0.22)",
    }


def apply_dataframe_cas_theme(styler: object, theme_mode: str = "light") -> object:
    """Apply concrete CAS light/dark colors to pandas Styler tables."""
    if not hasattr(styler, "set_table_styles"):
        return styler

    palette = _dashboard_table_palette(theme_mode)

    return styler.set_table_styles(
        [
            {
                "selector": "table",
                "props": [
                    ("width", "100%"),
                    ("border-collapse", "collapse"),
                    ("background-color", palette["panel"]),
                    ("color", palette["text"]),
                    ("border-color", palette["border"]),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("background-color", palette["panel_strong"]),
                    ("color", palette["text"]),
                    ("border", f"1px solid {palette['border']}"),
                    ("font-weight", "700"),
                    ("padding", "0.45rem 0.55rem"),
                    ("white-space", "nowrap"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("background-color", palette["panel"]),
                    ("color", palette["text"]),
                    ("border", f"1px solid {palette['border_soft']}"),
                    ("padding", "0.42rem 0.55rem"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even) td",
                "props": [
                    ("background-color", palette["card"]),
                ],
            },
        ],
        overwrite=False,
    )


def render_themed_dataframe(styler: object, theme_mode: str = "light") -> None:
    """Render a pandas Styler as themed HTML instead of Streamlit's white dataframe grid."""
    themed = apply_dataframe_cas_theme(styler, theme_mode)
    if not hasattr(themed, "to_html"):
        return

    st.markdown(
        f"<div class='cas-themed-table-wrap'>{themed.to_html()}</div>",
        unsafe_allow_html=True,
    )


def render_scenario_tab(
    selected_row: pd.Series,
    artifacts: DashboardArtifacts,
    *,
    feature_map: pd.DataFrame,
    formatters: ScenarioTabFormatters,
    theme_mode: str = "light",
) -> None:
    """Render the scenario tab."""
    st.subheader("가정별 변화 보기")
    st.caption(
        "핵심 지표 값을 가정적으로 바꿔 보면서, 현재 기업의 상대적 위치가 어떻게 달라지는지 살펴봅니다."
    )
    presets = list(artifacts.scenario_presets.keys())
    if not presets:
        st.info("사용할 수 있는 시나리오 프리셋이 없습니다.")
        return

    preset_label_map = {
        "base": "기본",
        "mild_stress": "완만한 스트레스",
        "severe_stress": "강한 스트레스",
    }
    selected_preset = cast(
        str,
        st.selectbox(
            "시나리오 선택",
            presets,
            format_func=lambda value: preset_label_map.get(value, value),
        ),
    )
    raw_preset_changes = artifacts.scenario_presets[selected_preset]
    preset_changes = (
        cast(Mapping[str, object], raw_preset_changes)
        if isinstance(raw_preset_changes, dict)
        else {}
    )
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

    deltas: dict[str, float] = {}
    for feature in DEFAULT_SCENARIO_FEATURES:
        label = formatters.display_name(feature, feature_map)
        default_delta = _float_or_default(preset_changes.get(feature), default=0.0)
        deltas[feature] = float(
            st.slider(
                f"{label} 얼마나 바꿔볼까요?",
                min_value=-1.0,
                max_value=1.0,
                value=default_delta,
                step=0.01,
            )
        )

    scenario_frame = build_scenario_frame(
        selected_row=selected_row,
        company_universe=artifacts.company_universe,
        feature_map=feature_map,
        deltas=deltas,
        formatters=formatters,
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
        formatters.delta_with_unit(
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
        formatters.scalar(len(scenario_frame)),
        "현재 화면에서 직접 움직여 볼 수 있는 핵심 지표 개수입니다.",
        COLOR_COMPANY,
    )
    st.markdown("**시나리오 적용 전후 보기**")
    for unit_value, unit_frame in scenario_frame.groupby("unit", dropna=False):
        unit_label = formatters.unit_description(str(unit_value))
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
        scenario_chart_frame = finite_chart_frame(chart_rows, ["값"])
        scenario_chart = (
            alt.Chart(scenario_chart_frame)
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
                yOffset="구분:N",
                tooltip=["변수:N", "구분:N", alt.Tooltip("값_표시:N", title="값")],
            )
            .properties(height=max(160, len(unit_frame) * 56))
        )
        if scenario_chart_frame.empty:
            st.caption("이 단위 그룹은 차트로 표시할 수 있는 숫자형 값이 없습니다.")
        else:
            scenario_chart = (
                alt.Chart(scenario_chart_frame)
                .mark_bar()
                .encode(
                    y=alt.Y("표시명:N", title=None, sort=None),
                    x=alt.X("값:Q", title="값"),
                    color=alt.Color("구분:N", title="구분"),
                    tooltip=[
                        alt.Tooltip("표시명:N", title="변수"),
                        alt.Tooltip("구분:N", title="구분"),
                        alt.Tooltip("값:Q", title="값"),
                    ],
                )
                .properties(height=max(180, 32 * len(scenario_chart_frame)))
            )

            themed_scenario_chart = apply_altair_cas_theme(scenario_chart, theme_mode)
            stretch_altair_chart(themed_scenario_chart)
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
    render_themed_dataframe(styled_scenario, theme_mode)
    st.warning(
        "현재 시나리오 탭은 지표를 바꿔 보았을 때 상대적 위치가 어떻게 달라지는지 보여줍니다. "
        "기업별 예측확률을 다시 계산하는 기능은 다음 단계에서 추가할 수 있습니다."
    )


def _scenario_value(baseline_value: object, delta: float) -> float | None:
    baseline_number = _float_or_none(baseline_value)
    if baseline_number is None:
        return None
    return baseline_number + delta


def _float_or_default(value: object, *, default: float) -> float:
    parsed = _float_or_none(value)
    return default if parsed is None else parsed


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None
