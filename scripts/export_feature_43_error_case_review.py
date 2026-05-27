from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "data" / "outputs" / "dashboard" / "feature_46_mvp"
DIAGNOSTICS_DIR = ROOT / "data" / "outputs" / "modeling" / "feature_46_xgboost" / "diagnostics"
ERROR_SHAP_DETAILS_PATH = DIAGNOSTICS_DIR / "error_shap_case_details.csv"
PEER_PERCENTILES_PATH = DASHBOARD_DIR / "peer_percentiles.csv"
OUTPUT_DIR = DIAGNOSTICS_DIR

CASE_KEYS = ["market", "stock_code_norm", "corp_name", "fiscal_year", "eval_year"]
TOP_FEATURES_PER_CASE = 5

AMOUNT_CONTEXT_FEATURES = {
    "assets_total",
    "gross_profit",
    "depreciation",
    "firm_size_group",
    "listed_year",
    "industry_macro_category",
    "market",
}
LEVERAGE_FEATURES = {
    "debt_ratio",
    "total_borrowings_ratio",
    "capital_impairment_ratio",
    "short_term_borrowings_share",
    "interest_coverage_ratio",
    "ocf_to_total_borrowings",
}
PROFIT_CASHFLOW_FEATURES = {
    "net_margin",
    "gross_profit",
    "pretax_roa",
    "operating_roa",
    "pretax_roe",
    "cashflow_coverage_ratio",
    "ocf_to_total_liabilities",
    "ocf_to_total_borrowings",
    "ocf_to_sales",
    "is_2y_consecutive_ocf_deficit",
    "is_2y_consecutive_operating_loss",
}
STABILITY_MASK_FEATURES = {
    "assets_total",
    "gross_profit",
    "depreciation",
    "firm_size_group",
    "dividend_payer",
    "listed_year",
}
CURRENT_STABLE_RATIO_FEATURES = {
    "interest_coverage_ratio",
    "cashflow_coverage_ratio",
    "equity_ratio",
    "capital_impairment_ratio",
    "net_margin",
    "total_debt_turnover",
    "ocf_to_total_borrowings",
    "ocf_to_total_liabilities",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build a human-readable FP/FN error review table from SHAP case diagnostics.")
    )
    parser.add_argument("--error-shap-details", type=Path, default=ERROR_SHAP_DETAILS_PATH)
    parser.add_argument("--peer-percentiles", type=Path, default=PEER_PERCENTILES_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def normalize_stock_code(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
    )


def read_error_details(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Error SHAP detail file not found: {path}. "
            "Run scripts/export_feature_43_error_shap_analysis.py first."
        )
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = {
        *CASE_KEYS,
        "stock_code",
        "firm_size_group",
        "industry_macro_category",
        "error_type",
        "actual_label_name",
        "predicted_label_name",
        "prob_speculative",
        "threshold",
        "probability_distance_from_threshold",
        "feature",
        "rank",
        "direction",
        "shap_value",
        "feature_value",
        "korean_name",
        "feature_group",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise KeyError(f"error_shap_case_details.csv is missing columns: {missing_columns}")
    frame = frame.copy()
    frame["stock_code_norm"] = normalize_stock_code(frame["stock_code_norm"])
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame["shap_value"] = pd.to_numeric(frame["shap_value"], errors="coerce")
    frame["prob_speculative"] = pd.to_numeric(frame["prob_speculative"], errors="coerce")
    frame["threshold"] = pd.to_numeric(frame["threshold"], errors="coerce")
    frame["probability_distance_from_threshold"] = pd.to_numeric(
        frame["probability_distance_from_threshold"],
        errors="coerce",
    )
    return frame


def read_peer_percentiles(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = {
        "stock_code",
        "corp_name",
        "fiscal_year",
        "eval_year",
        "market",
        "feature",
        "industry_percentile",
        "market_percentile",
        "overall_percentile",
    }
    if not required_columns.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.copy()
    frame["stock_code_norm"] = normalize_stock_code(frame["stock_code"])
    return frame.loc[
        :,
        [
            "market",
            "stock_code_norm",
            "corp_name",
            "fiscal_year",
            "eval_year",
            "feature",
            "industry_percentile",
            "market_percentile",
            "overall_percentile",
        ],
    ]


def enrich_with_percentiles(details: pd.DataFrame, peer_percentiles: pd.DataFrame) -> pd.DataFrame:
    if peer_percentiles.empty:
        details = details.copy()
        details["industry_percentile"] = np.nan
        details["market_percentile"] = np.nan
        details["overall_percentile"] = np.nan
        return details
    merge_keys = [*CASE_KEYS, "feature"]
    return details.merge(peer_percentiles, on=merge_keys, how="left")


def format_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) >= 1_000_000:
        return f"{numeric:,.0f}"
    if abs(numeric) >= 100:
        return f"{numeric:,.2f}"
    if abs(numeric) >= 1:
        return f"{numeric:.3f}"
    return f"{numeric:.4f}"


def format_percentile(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.0f}p"


def format_probability(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def direction_ko(direction: str) -> str:
    if direction == "increase_risk":
        return "위험↑"
    if direction == "decrease_risk":
        return "위험↓"
    return str(direction)


def select_review_category(
    error_type: str,
    increase_features: set[str],
    decrease_features: set[str],
    top_increase_features: set[str],
    top_decrease_features: set[str],
) -> str:
    if error_type == "false_positive":
        if (top_increase_features & PROFIT_CASHFLOW_FEATURES) and (
            top_increase_features & LEVERAGE_FEATURES
        ):
            return "수익성·상환능력 복합 경고 과대반응"
        if top_increase_features & LEVERAGE_FEATURES:
            return "부채·상환능력 경고 과대반응"
        if top_increase_features & PROFIT_CASHFLOW_FEATURES:
            return "수익성·현금흐름 경고 과대반응"
        if top_increase_features & AMOUNT_CONTEXT_FEATURES:
            return "규모/상장맥락 과민반응"
        if increase_features & AMOUNT_CONTEXT_FEATURES:
            return "규모/상장맥락 보조 신호 과민반응"
        if increase_features & LEVERAGE_FEATURES:
            return "부채·상환능력 경고 과대반응"
        if increase_features & PROFIT_CASHFLOW_FEATURES:
            return "수익성·현금흐름 경고 과대반응"
        return "복합 위험신호 과대반응"

    if top_decrease_features & CURRENT_STABLE_RATIO_FEATURES:
        return "현재 재무비율 안정 신호가 위험을 가림"
    if top_decrease_features & STABILITY_MASK_FEATURES:
        return "규모·우량기업 안정 신호가 위험을 가림"
    if decrease_features & CURRENT_STABLE_RATIO_FEATURES:
        return "현재 재무비율 안정 신호가 위험을 가림"
    if decrease_features & STABILITY_MASK_FEATURES:
        return "규모·우량기업 안정 신호가 위험을 가림"
    if increase_features:
        return "위험 신호는 있었지만 강도가 부족"
    return "재무제표 외부 이벤트 가능성"


def build_hypothesis_and_action(category: str, error_type: str) -> tuple[str, str]:
    if category == "규모/상장맥락 과민반응":
        return (
            "절대금액, 기업규모, 상장연차 같은 맥락 변수가 위험 신호처럼 작동했지만 실제 등급은 투자적격으로 유지된 사례입니다.",
            "절대금액을 단독으로 늘리기보다 산업 내 위치, 규모조정 비율, 외부근거 확인을 함께 쓰는 방향이 적절합니다.",
        )
    if category == "부채·상환능력 경고 과대반응":
        return (
            "부채·상환능력 지표가 경고를 냈지만 실제 신용등급 하락으로 이어지지 않은 사례입니다.",
            "단기 경고와 지속 악화를 구분하도록 2년 이상 지속 여부나 현금흐름 보완 지표를 추가 점검합니다.",
        )
    if category == "수익성·현금흐름 경고 과대반응":
        return (
            "수익성 또는 현금흐름 지표 악화가 모델 위험 판단을 키웠지만 실제 라벨은 안정적이었던 사례입니다.",
            "일시적 부진인지 구조적 악화인지 구분하는 추세 변수와 산업 내 비교 지표를 보강합니다.",
        )
    if category == "수익성·상환능력 복합 경고 과대반응":
        return (
            "수익성 악화와 이자보상·상환능력 경고가 동시에 나타나 모델 확률을 크게 끌어올렸지만 실제 등급은 유지된 사례입니다.",
            "이 조합이 실제 등급 하락으로 이어지는 조건을 외부근거, 지속기간, 산업 내 위치와 함께 재점검합니다.",
        )
    if category == "규모/상장맥락 보조 신호 과민반응":
        return (
            "핵심 위험 요인은 다른 재무지표였지만 절대금액·규모·상장연차가 보조적으로 위험 판단을 키운 사례입니다.",
            "보조 맥락 변수가 FP를 키우는지 세그먼트별로 확인하고, 필요하면 committee_view에서 근거 강도를 낮춰 해석합니다.",
        )
    if category == "규모·우량기업 안정 신호가 위험을 가림":
        return (
            "자산규모, 배당, 매출총이익, 감가상각비 같은 우량·규모성 안정 신호가 실제 투기등급 위험을 낮춰 본 사례입니다.",
            "규모가 큰 기업의 이벤트성 위험은 뉴스, 공시, 등급전망, 감사의견 같은 외부근거 플래그로 보완하는 편이 좋습니다.",
        )
    if category == "현재 재무비율 안정 신호가 위험을 가림":
        return (
            "현재 재무비율은 양호해 보였지만 다음 연도 라벨은 투기등급으로 나타난 사례입니다.",
            "단일 연도 수준값보다 악화 속도, 최근 공시 이슈, 산업 충격 신호를 함께 보도록 보완합니다.",
        )
    if category == "위험 신호는 있었지만 강도가 부족":
        return (
            "일부 위험 요인은 있었지만 안정 신호가 더 강해 최종 확률이 threshold 아래에 머문 사례입니다.",
            "위험 요인의 조합 조건과 외부근거가 있을 때의 위원회 보류/부적격 전환 규칙을 점검합니다.",
        )

    if error_type == "false_positive":
        return (
            "여러 약한 위험 신호가 합쳐져 모델이 투기등급으로 판단했지만 실제 등급은 유지된 사례입니다.",
            "유사 FP를 묶어 threshold 보정 또는 세그먼트별 precision 개선 후보로 관리합니다.",
        )
    return (
        "재무제표만으로 설명하기 어려운 등급 하락 가능성이 있는 사례입니다.",
        "뉴스, DART 공시, 감사의견, 등급전망 같은 외부근거를 우선 확인합니다.",
    )


def confidence_priority(error_type: str, probability: float, threshold: float) -> str:
    distance = abs(probability - threshold)
    if error_type == "false_positive" and probability >= 0.7:
        return "상"
    if error_type == "false_negative" and probability <= 0.1:
        return "상"
    if distance >= 0.25:
        return "상"
    if distance >= 0.10:
        return "중"
    return "하"


def format_feature_list(case_details: pd.DataFrame, direction: str | None = None) -> str:
    frame = case_details.sort_values("rank")
    if direction is not None:
        frame = frame.loc[frame["direction"].eq(direction)]
    pieces = []
    for row in frame.head(TOP_FEATURES_PER_CASE).itertuples():
        percentile = format_percentile(getattr(row, "industry_percentile", np.nan))
        pieces.append(
            f"{row.korean_name}({direction_ko(row.direction)}, "
            f"SHAP={float(row.shap_value):+.3f}, 값={format_value(row.feature_value)}, "
            f"산업백분위={percentile})"
        )
    return "; ".join(pieces) if pieces else "-"


def build_review_table(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered_cases = (
        details.loc[
            :,
            [
                *CASE_KEYS,
                "error_type",
                "probability_distance_from_threshold",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            ["error_type", "probability_distance_from_threshold"],
            ascending=[True, False],
        )
    )

    for case in ordered_cases.itertuples(index=False):
        mask = pd.Series(True, index=details.index)
        for key in CASE_KEYS:
            mask &= details[key].eq(getattr(case, key))
        case_details = details.loc[mask].sort_values("rank")
        first = case_details.iloc[0]
        increase_features = set(
            case_details.loc[case_details["direction"].eq("increase_risk"), "feature"]
        )
        decrease_features = set(
            case_details.loc[case_details["direction"].eq("decrease_risk"), "feature"]
        )
        top_case_details = case_details.loc[case_details["rank"].le(3)]
        top_increase_features = set(
            top_case_details.loc[top_case_details["direction"].eq("increase_risk"), "feature"]
        )
        top_decrease_features = set(
            top_case_details.loc[top_case_details["direction"].eq("decrease_risk"), "feature"]
        )
        category = select_review_category(
            first["error_type"],
            increase_features,
            decrease_features,
            top_increase_features,
            top_decrease_features,
        )
        hypothesis, action = build_hypothesis_and_action(category, first["error_type"])
        priority = confidence_priority(
            first["error_type"],
            float(first["prob_speculative"]),
            float(first["threshold"]),
        )
        error_type_ko = (
            "False Positive: 실제 투자적격인데 모델은 투기등급"
            if first["error_type"] == "false_positive"
            else "False Negative: 실제 투기등급인데 모델은 투자적격"
        )

        rows.append(
            {
                "review_priority": priority,
                "error_type": first["error_type"],
                "error_type_ko": error_type_ko,
                "review_category": category,
                "market": first["market"],
                "stock_code": first["stock_code_norm"],
                "corp_name": first["corp_name"],
                "fiscal_year": int(first["fiscal_year"]),
                "eval_year": int(first["eval_year"]),
                "industry_macro_category": first["industry_macro_category"],
                "firm_size_group": first["firm_size_group"],
                "actual_label_name": first["actual_label_name"],
                "predicted_label_name": first["predicted_label_name"],
                "prob_speculative": float(first["prob_speculative"]),
                "threshold": float(first["threshold"]),
                "probability_gap_from_threshold": float(
                    first["probability_distance_from_threshold"]
                ),
                "top_shap_features": format_feature_list(case_details),
                "risk_increasing_features": format_feature_list(case_details, "increase_risk"),
                "risk_decreasing_features": format_feature_list(case_details, "decrease_risk"),
                "model_misread_hypothesis_ko": hypothesis,
                "recommended_next_action_ko": action,
            }
        )

    priority_order = {"상": 0, "중": 1, "하": 2}
    review = pd.DataFrame(rows)
    review["_priority_order"] = review["review_priority"].map(priority_order).fillna(9)
    return review.sort_values(
        ["_priority_order", "error_type", "probability_gap_from_threshold"],
        ascending=[True, True, False],
    ).drop(columns=["_priority_order"])


def build_category_summary(review: pd.DataFrame) -> pd.DataFrame:
    return (
        review.groupby(["error_type", "review_category"], dropna=False)
        .agg(
            cases=("corp_name", "size"),
            high_priority_cases=("review_priority", lambda x: int((x == "상").sum())),
            mean_probability=("prob_speculative", "mean"),
            mean_gap=("probability_gap_from_threshold", "mean"),
        )
        .reset_index()
        .sort_values(["error_type", "cases"], ascending=[True, False])
    )


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    rows = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.to_dict(orient="records"):
        values = []
        for _, column in columns:
            value = row.get(column)
            if column in {"prob_speculative", "threshold", "probability_gap_from_threshold"}:
                values.append(format_probability(value))
            elif isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value) if value is not None else "")
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def build_report(review: pd.DataFrame, category_summary: pd.DataFrame) -> str:
    fp_count = int(review["error_type"].eq("false_positive").sum())
    fn_count = int(review["error_type"].eq("false_negative").sum())
    high_count = int(review["review_priority"].eq("상").sum())

    fp_examples = review.loc[review["error_type"].eq("false_positive")].head(8)
    fn_examples = review.loc[review["error_type"].eq("false_negative")].head(8)

    return "\n".join(
        [
            "# Feature 43 Error Case Review",
            "",
            "이 리포트는 FP/FN 오류를 단순 목록이 아니라 사람이 해석할 수 있는 "
            "`오류 유형`, `모델 오해 가설`, `다음 개선 액션`으로 정리한 산출물입니다.",
            "",
            "## 1. 요약",
            "",
            f"- False Positive: `{fp_count}`개",
            f"- False Negative: `{fn_count}`개",
            f"- 우선 검토 필요도 `상`: `{high_count}`개",
            "",
            "## 2. 오류 유형별 집계",
            "",
            markdown_table(
                category_summary,
                [
                    ("Error", "error_type"),
                    ("유형", "review_category"),
                    ("Cases", "cases"),
                    ("Priority 상", "high_priority_cases"),
                    ("Mean Prob.", "mean_probability"),
                    ("Mean Gap", "mean_gap"),
                ],
            ),
            "",
            "## 3. 우선 검토 False Positive",
            "",
            markdown_table(
                fp_examples,
                [
                    ("우선순위", "review_priority"),
                    ("기업", "corp_name"),
                    ("시장", "market"),
                    ("연도", "fiscal_year"),
                    ("확률", "prob_speculative"),
                    ("유형", "review_category"),
                    ("모델 오해 가설", "model_misread_hypothesis_ko"),
                    ("다음 액션", "recommended_next_action_ko"),
                ],
            ),
            "",
            "## 4. 우선 검토 False Negative",
            "",
            markdown_table(
                fn_examples,
                [
                    ("우선순위", "review_priority"),
                    ("기업", "corp_name"),
                    ("시장", "market"),
                    ("연도", "fiscal_year"),
                    ("확률", "prob_speculative"),
                    ("유형", "review_category"),
                    ("모델 오해 가설", "model_misread_hypothesis_ko"),
                    ("다음 액션", "recommended_next_action_ko"),
                ],
            ),
            "",
            "## 5. 다음 개선 방향",
            "",
            "- FP는 수익성 악화, 이자보상·상환능력 경고, 현금흐름 약화가 실제 등급 하락으로 이어지는 조건을 더 세밀하게 구분해야 합니다.",
            "- 규모·상장맥락 변수는 단독 교체 실험에서 성능 개선이 없었으므로, 모델 변수 교체보다 에이전트 설명에서 근거 강도를 조절하는 쪽이 안전합니다.",
            "- FN은 재무제표가 안정적으로 보이는 기업의 이벤트성 위험을 외부근거 플래그로 보완하는 방향이 가장 자연스럽습니다.",
            "- 운영 모델 변수셋은 현재 baseline이 가장 강하므로 바로 바꾸지 않고, 오류 리뷰 테이블을 기준으로 외부근거/공시/뉴스 신호를 committee_view에서 먼저 검증하는 편이 안전합니다.",
        ]
    )


def write_outputs(review: pd.DataFrame, category_summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "error_case_review_table.csv"
    summary_path = output_dir / "error_case_review_category_summary.csv"
    report_path = output_dir / "error_case_review_report.md"
    json_path = output_dir / "error_case_review_summary.json"

    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    category_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report_path.write_text(build_report(review, category_summary), encoding="utf-8")

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "false_positive_count": int(review["error_type"].eq("false_positive").sum()),
        "false_negative_count": int(review["error_type"].eq("false_negative").sum()),
        "high_priority_count": int(review["review_priority"].eq("상").sum()),
        "category_summary": category_summary.to_dict(orient="records"),
        "output_files": {
            "review_table": str(review_path.relative_to(ROOT)),
            "category_summary": str(summary_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    details = read_error_details(args.error_shap_details)
    peer_percentiles = read_peer_percentiles(args.peer_percentiles)
    enriched_details = enrich_with_percentiles(details, peer_percentiles)
    review = build_review_table(enriched_details)
    category_summary = build_category_summary(review)
    write_outputs(review, category_summary, args.output_dir)
    print(
        json.dumps(
            {
                "review_rows": len(review),
                "false_positive": int(review["error_type"].eq("false_positive").sum()),
                "false_negative": int(review["error_type"].eq("false_negative").sum()),
                "high_priority": int(review["review_priority"].eq("상").sum()),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
