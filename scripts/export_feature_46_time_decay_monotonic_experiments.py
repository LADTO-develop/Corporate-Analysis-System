from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cas.modeling.stage1_xgboost import (  # noqa: E402
    DEFAULT_ROLLING_EVAL_YEARS,
    DEFAULT_STAGE1_RANDOM_STATE,
    DEFAULT_STAGE1_RECALL_FLOOR,
    build_monotonic_constraints,
    build_time_decay_weights,
    evaluate_calibrated_stage1_split,
    read_stage1_feature_columns,
    read_stage1_master,
    train_stage1_xgboost,
)

INPUT_DIR = ROOT / "data" / "input" / "credit_46_features"
MASTER_PATH = INPUT_DIR / "feature_46_master.csv"
FEATURE_LIST_PATH = INPUT_DIR / "feature_46_list.json"
OUTPUT_DIR = ROOT / "data" / "outputs" / "modeling" / "feature_46_xgboost" / "diagnostics"

RANDOM_STATE = DEFAULT_STAGE1_RANDOM_STATE
RECALL_FLOOR = DEFAULT_STAGE1_RECALL_FLOOR
ROLLING_EVAL_YEARS = DEFAULT_ROLLING_EVAL_YEARS

ROLLING_FOLD_METRICS_FILENAME = "time_decay_monotonic_xgboost_rolling_fold_metrics.csv"
ROLLING_SUMMARY_FILENAME = "time_decay_monotonic_xgboost_rolling_summary.csv"
FINAL_TEST_FILENAME = "time_decay_monotonic_xgboost_final_test.csv"
SUMMARY_FILENAME = "time_decay_monotonic_xgboost_summary.json"
REPORT_FILENAME = "time_decay_monotonic_xgboost_report.md"

BROAD_MONOTONIC_DIRECTIONS: dict[str, int] = {
    "current_ratio": -1,
    "cash_ratio": -1,
    "equity_ratio": -1,
    "debt_ratio": 1,
    "total_borrowings_ratio": 1,
    "capital_impairment_ratio": 1,
    "net_margin": -1,
    "interest_coverage_ratio": -1,
    "pretax_roa": -1,
    "operating_roa": -1,
    "pretax_roe": -1,
    "ocf_to_total_liabilities": -1,
    "ocf_to_total_borrowings": -1,
    "ocf_to_sales": -1,
    "cashflow_coverage_ratio": -1,
    "accruals_ratio": 1,
    "intangible_assets_ratio": 1,
    "spec_spread": 1,
    "short_term_borrowings_share": 1,
    "net_margin_diff": -1,
    "is_2y_consecutive_ocf_deficit": 1,
    "icr_under_1": 1,
    "is_2y_consecutive_operating_loss": 1,
    "gross_profit_industry_year_pct": -1,
}

CORE_MONOTONIC_DIRECTIONS: dict[str, int] = {
    "current_ratio": -1,
    "cash_ratio": -1,
    "equity_ratio": -1,
    "debt_ratio": 1,
    "total_borrowings_ratio": 1,
    "capital_impairment_ratio": 1,
    "interest_coverage_ratio": -1,
    "ocf_to_total_liabilities": -1,
    "ocf_to_total_borrowings": -1,
    "cashflow_coverage_ratio": -1,
    "spec_spread": 1,
    "short_term_borrowings_share": 1,
    "is_2y_consecutive_ocf_deficit": 1,
    "icr_under_1": 1,
    "is_2y_consecutive_operating_loss": 1,
}

LEVERAGE_LIQUIDITY_MONOTONIC_DIRECTIONS: dict[str, int] = {
    "current_ratio": -1,
    "cash_ratio": -1,
    "equity_ratio": -1,
    "debt_ratio": 1,
    "total_borrowings_ratio": 1,
    "capital_impairment_ratio": 1,
    "interest_coverage_ratio": -1,
    "ocf_to_total_borrowings": -1,
    "cashflow_coverage_ratio": -1,
    "short_term_borrowings_share": 1,
}

DISTRESS_GUARDRAIL_MONOTONIC_DIRECTIONS: dict[str, int] = {
    "debt_ratio": 1,
    "total_borrowings_ratio": 1,
    "capital_impairment_ratio": 1,
    "interest_coverage_ratio": -1,
    "cashflow_coverage_ratio": -1,
    "spec_spread": 1,
    "is_2y_consecutive_ocf_deficit": 1,
    "icr_under_1": 1,
    "is_2y_consecutive_operating_loss": 1,
}

MONOTONIC_PROFILES: dict[str, dict[str, int]] = {
    "broad": BROAD_MONOTONIC_DIRECTIONS,
    "core": CORE_MONOTONIC_DIRECTIONS,
    "leverage_liquidity": LEVERAGE_LIQUIDITY_MONOTONIC_DIRECTIONS,
    "distress_guardrail": DISTRESS_GUARDRAIL_MONOTONIC_DIRECTIONS,
}

RISK_PROXY_GROUP_COLUMNS = ["fiscal_year", "industry_macro_category"]
RISK_PROXY_SPECS = [
    ("debt_ratio", "risk_proxy_debt_ratio_industry_year_pct", "direct"),
    ("total_borrowings_ratio", "risk_proxy_total_borrowings_ratio_industry_year_pct", "direct"),
    (
        "short_term_borrowings_share",
        "risk_proxy_short_term_borrowings_share_industry_year_pct",
        "direct",
    ),
    ("capital_impairment_ratio", "risk_proxy_capital_impairment_ratio_industry_year_pct", "direct"),
    ("spec_spread", "risk_proxy_spec_spread_industry_year_pct", "direct"),
    ("accruals_ratio", "risk_proxy_accruals_ratio_industry_year_pct", "direct"),
    ("current_ratio", "risk_proxy_current_ratio_inverse_industry_year_pct", "inverse"),
    ("cash_ratio", "risk_proxy_cash_ratio_inverse_industry_year_pct", "inverse"),
    ("interest_coverage_ratio", "risk_proxy_interest_coverage_inverse_industry_year_pct", "inverse"),
    ("cashflow_coverage_ratio", "risk_proxy_cashflow_coverage_inverse_industry_year_pct", "inverse"),
    ("ocf_to_total_borrowings", "risk_proxy_ocf_to_borrowings_inverse_industry_year_pct", "inverse"),
    ("operating_roa", "risk_proxy_operating_roa_inverse_industry_year_pct", "inverse"),
]
RISK_PROXY_COLUMNS = [proxy for _, proxy, _ in RISK_PROXY_SPECS]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    display_name: str
    note: str
    half_life_years: float | None = None
    use_monotonic_constraints: bool = False
    monotonic_profile: str | None = None
    risk_proxy_pack: bool = False


CANDIDATES = [
    CandidateSpec(
        candidate_id="baseline_current",
        display_name="현재 46 XGBoost",
        note="sample weight와 monotonic constraint를 쓰지 않는 현재 공식 구조",
    ),
    CandidateSpec(
        candidate_id="time_decay_half_life_1y",
        display_name="최근연도 가중 half-life 1년",
        note="학습 연도가 정책연도에서 1년 멀어질 때마다 가중치를 절반으로 감소",
        half_life_years=1.0,
    ),
    CandidateSpec(
        candidate_id="time_decay_half_life_2y",
        display_name="최근연도 가중 half-life 2년",
        note="2021-2022에 가까운 과거 학습 표본을 완만하게 더 반영",
        half_life_years=2.0,
    ),
    CandidateSpec(
        candidate_id="time_decay_half_life_3y",
        display_name="최근연도 가중 half-life 3년",
        note="장기 표본을 크게 버리지 않는 완만한 recency weighting",
        half_life_years=3.0,
    ),
    CandidateSpec(
        candidate_id="monotonic_directional",
        display_name="방향성 단조 제약",
        note="부채/차입/스프레드는 위험 증가, 유동성/수익성/현금흐름은 위험 감소로 제약",
        use_monotonic_constraints=True,
        monotonic_profile="broad",
    ),
    CandidateSpec(
        candidate_id="monotonic_time_decay_2y",
        display_name="단조 제약 + half-life 2년",
        note="방향성 제약과 최근연도 가중치를 함께 적용",
        half_life_years=2.0,
        use_monotonic_constraints=True,
        monotonic_profile="broad",
    ),
    CandidateSpec(
        candidate_id="monotonic_time_decay_3y",
        display_name="단조 제약 + half-life 3년",
        note="방향성 제약과 완만한 최근연도 가중치를 함께 적용",
        half_life_years=3.0,
        use_monotonic_constraints=True,
        monotonic_profile="broad",
    ),
    CandidateSpec(
        candidate_id="monotonic_core_time_decay_3y",
        display_name="핵심 변수 단조 제약 + half-life 3년",
        note="방향성이 가장 명확한 leverage/liquidity/cashflow/guardrail 변수만 제약",
        half_life_years=3.0,
        use_monotonic_constraints=True,
        monotonic_profile="core",
    ),
    CandidateSpec(
        candidate_id="monotonic_leverage_liquidity_time_decay_3y",
        display_name="차입/유동성 단조 제약 + half-life 3년",
        note="수익성/규모성 변수는 풀고 차입, 유동성, 현금흐름 커버리지 중심으로 제약",
        half_life_years=3.0,
        use_monotonic_constraints=True,
        monotonic_profile="leverage_liquidity",
    ),
    CandidateSpec(
        candidate_id="monotonic_guardrail_time_decay_3y",
        display_name="위험 guardrail 단조 제약 + half-life 3년",
        note="명확한 부실 guardrail 변수만 제약해 과도한 수익성 제약을 피함",
        half_life_years=3.0,
        use_monotonic_constraints=True,
        monotonic_profile="distress_guardrail",
    ),
    CandidateSpec(
        candidate_id="risk_proxy_time_decay_3y",
        display_name="위험 proxy pack + half-life 3년",
        note="원변수 대신 산업-연도 내 위험방향 percentile proxy 12개를 추가",
        half_life_years=3.0,
        risk_proxy_pack=True,
    ),
    CandidateSpec(
        candidate_id="risk_proxy_monotonic_time_decay_3y",
        display_name="위험 proxy 단조 제약 + half-life 3년",
        note="위험방향 percentile proxy만 higher-risk 방향으로 단조 제약",
        half_life_years=3.0,
        use_monotonic_constraints=True,
        monotonic_profile="risk_proxy",
        risk_proxy_pack=True,
    ),
    CandidateSpec(
        candidate_id="risk_proxy_core_monotonic_time_decay_3y",
        display_name="위험 proxy + 핵심 단조 제약 + half-life 3년",
        note="위험 proxy에는 +1, 원변수에는 핵심 high-confidence 제약만 적용",
        half_life_years=3.0,
        use_monotonic_constraints=True,
        monotonic_profile="risk_proxy_core",
        risk_proxy_pack=True,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run rolling OOT time-decay sample-weight and monotonic-constraint experiments "
            "for the official 46-feature XGBoost model."
        )
    )
    parser.add_argument("--master-path", type=Path, default=MASTER_PATH)
    parser.add_argument("--feature-list-path", type=Path, default=FEATURE_LIST_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--eval-years", type=int, nargs="+", default=ROLLING_EVAL_YEARS)
    return parser.parse_args()


def read_master(path: Path) -> pd.DataFrame:
    return read_stage1_master(
        path,
        duplicate_keys=["market", "stock_code", "corp_name", "fiscal_year"],
    )


def read_feature_columns(path: Path, master: pd.DataFrame) -> list[str]:
    return read_stage1_feature_columns(path, master)


def add_risk_proxy_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    missing_groups = [column for column in RISK_PROXY_GROUP_COLUMNS if column not in output.columns]
    if missing_groups:
        raise KeyError(f"Missing risk proxy group columns: {missing_groups}")

    grouped = output.groupby(RISK_PROXY_GROUP_COLUMNS, dropna=False)
    for source, proxy, direction in RISK_PROXY_SPECS:
        if source not in output.columns:
            raise KeyError(f"Missing risk proxy source column: {source}")
        percentile = grouped[source].rank(pct=True, method="average")
        output[proxy] = 1.0 - percentile if direction == "inverse" else percentile
    return output


def candidate_feature_columns(base_columns: list[str], spec: CandidateSpec) -> list[str]:
    columns = list(base_columns)
    if spec.risk_proxy_pack:
        columns.extend([column for column in RISK_PROXY_COLUMNS if column not in columns])
    return columns


def candidate_monotonic_directions(spec: CandidateSpec) -> dict[str, int]:
    if not spec.use_monotonic_constraints:
        return {}
    profile = spec.monotonic_profile or "broad"
    if profile == "risk_proxy":
        return {column: 1 for column in RISK_PROXY_COLUMNS}
    if profile == "risk_proxy_core":
        return {**CORE_MONOTONIC_DIRECTIONS, **{column: 1 for column in RISK_PROXY_COLUMNS}}
    try:
        return MONOTONIC_PROFILES[profile]
    except KeyError as error:
        raise KeyError(f"Unknown monotonic profile: {profile}") from error


def candidate_params(spec: CandidateSpec, columns: list[str]) -> dict[str, object]:
    params: dict[str, object] = {}
    if spec.use_monotonic_constraints:
        params["monotone_constraints"] = build_monotonic_constraints(
            columns,
            candidate_monotonic_directions(spec),
        )
    return params


def candidate_train_weights(
    train: pd.DataFrame,
    *,
    spec: CandidateSpec,
    reference_year: int,
) -> object | None:
    if spec.half_life_years is None:
        return None
    return build_time_decay_weights(
        train,
        reference_year=reference_year,
        half_life_years=spec.half_life_years,
    )


def evaluate_candidate_split(
    *,
    train: pd.DataFrame,
    policy: pd.DataFrame,
    evaluation: pd.DataFrame,
    columns: list[str],
    spec: CandidateSpec,
    reference_year: int,
    seed: int,
) -> tuple[dict[str, Any], int | None, float | None, float | None]:
    weights = candidate_train_weights(train, spec=spec, reference_year=reference_year)
    model = train_stage1_xgboost(
        train=train,
        policy=policy,
        columns=columns,
        params=candidate_params(spec, columns),
        seed=seed,
        train_sample_weight=weights,
    )
    metrics, _ = evaluate_calibrated_stage1_split(
        model=model,
        policy=policy,
        evaluation=evaluation,
        columns=columns,
    )
    weight_min = float(weights.min()) if weights is not None else None
    weight_max = float(weights.max()) if weights is not None else None
    return metrics, getattr(model, "best_iteration", None), weight_min, weight_max


def evaluate_rolling_candidate(
    *,
    master: pd.DataFrame,
    base_columns: list[str],
    spec: CandidateSpec,
    eval_years: list[int],
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    columns = candidate_feature_columns(base_columns, spec)
    for eval_year in eval_years:
        policy_year = eval_year - 1
        train = master.loc[master["fiscal_year"] < policy_year].copy()
        policy = master.loc[master["fiscal_year"] == policy_year].copy()
        evaluation = master.loc[master["fiscal_year"] == eval_year].copy()
        if train.empty or policy.empty or evaluation.empty:
            raise ValueError(
                f"Empty rolling split for eval_year={eval_year}: "
                f"train={len(train)}, policy={len(policy)}, evaluation={len(evaluation)}"
            )
        metrics, best_iteration, weight_min, weight_max = evaluate_candidate_split(
            train=train,
            policy=policy,
            evaluation=evaluation,
            columns=columns,
            spec=spec,
            reference_year=policy_year - 1,
            seed=seed,
        )
        rows.append(
            {
                **asdict(spec),
                "feature_count": len(columns),
                "added_risk_proxy_count": len(RISK_PROXY_COLUMNS) if spec.risk_proxy_pack else 0,
                "monotonic_constraint_count": sum(
                    1 for column in columns if candidate_monotonic_directions(spec).get(column, 0)
                ),
                "eval_year": eval_year,
                "policy_year": policy_year,
                "train_rows": len(train),
                "policy_rows": len(policy),
                "eval_rows": len(evaluation),
                "eval_positive_rate": float(evaluation["is_speculative"].mean()),
                "best_iteration": best_iteration,
                "train_weight_min": weight_min,
                "train_weight_max": weight_max,
                **metrics,
            }
        )
    return rows


def evaluate_final_test(
    *,
    master: pd.DataFrame,
    base_columns: list[str],
    spec: CandidateSpec,
    seed: int,
) -> dict[str, Any]:
    columns = candidate_feature_columns(base_columns, spec)
    train = master.loc[master["fiscal_year"] <= 2021].copy()
    policy = master.loc[master["fiscal_year"] == 2022].copy()
    evaluation = master.loc[master["fiscal_year"] >= 2023].copy()
    metrics, best_iteration, weight_min, weight_max = evaluate_candidate_split(
        train=train,
        policy=policy,
        evaluation=evaluation,
        columns=columns,
        spec=spec,
        reference_year=2021,
        seed=seed,
    )
    return {
        **asdict(spec),
        "feature_count": len(columns),
        "added_risk_proxy_count": len(RISK_PROXY_COLUMNS) if spec.risk_proxy_pack else 0,
        "monotonic_constraint_count": sum(
            1 for column in columns if candidate_monotonic_directions(spec).get(column, 0)
        ),
        "train_rows": len(train),
        "policy_rows": len(policy),
        "eval_rows": len(evaluation),
        "eval_positive_rate": float(evaluation["is_speculative"].mean()),
        "best_iteration": best_iteration,
        "train_weight_min": weight_min,
        "train_weight_max": weight_max,
        **metrics,
    }


def summarize_rolling(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "eval_pr_auc",
        "eval_roc_auc",
        "eval_brier",
        "eval_logloss",
        "eval_precision_at_threshold",
        "eval_recall_at_threshold",
        "eval_f1_at_threshold",
        "eval_false_positive_at_threshold",
        "eval_false_negative_at_threshold",
    ]
    rows = []
    for candidate_id, group in fold_metrics.groupby("candidate_id", sort=False):
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "display_name": group["display_name"].iloc[0],
            "note": group["note"].iloc[0],
            "half_life_years": group["half_life_years"].iloc[0],
            "use_monotonic_constraints": bool(group["use_monotonic_constraints"].iloc[0]),
            "monotonic_profile": group["monotonic_profile"].iloc[0],
            "risk_proxy_pack": bool(group["risk_proxy_pack"].iloc[0]),
            "feature_count": int(group["feature_count"].iloc[0]),
            "added_risk_proxy_count": int(group["added_risk_proxy_count"].iloc[0]),
            "monotonic_constraint_count": int(group["monotonic_constraint_count"].iloc[0]),
            "folds": len(group),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
            row[f"{column}_min"] = float(group[column].min())
            row[f"{column}_max"] = float(group[column].max())
        row["total_false_positive"] = int(group["eval_false_positive_at_threshold"].sum())
        row["total_false_negative"] = int(group["eval_false_negative_at_threshold"].sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    return add_baseline_deltas(summary, scope="rolling").sort_values(
        [
            "eval_f1_at_threshold_mean",
            "eval_recall_at_threshold_mean",
            "eval_pr_auc_mean",
            "eval_precision_at_threshold_mean",
        ],
        ascending=False,
    )


def add_baseline_deltas(frame: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    output = frame.copy()
    baseline = output.loc[output["candidate_id"].eq("baseline_current")]
    if baseline.empty:
        return output
    base = baseline.iloc[0]
    if scope == "rolling":
        for column in [
            "eval_pr_auc_mean",
            "eval_precision_at_threshold_mean",
            "eval_recall_at_threshold_mean",
            "eval_f1_at_threshold_mean",
        ]:
            output[f"{column}_delta_vs_baseline"] = output[column] - float(base[column])
        output["total_false_positive_delta_vs_baseline"] = (
            output["total_false_positive"] - int(base["total_false_positive"])
        )
        output["total_false_negative_delta_vs_baseline"] = (
            output["total_false_negative"] - int(base["total_false_negative"])
        )
    else:
        for column in [
            "eval_pr_auc",
            "eval_precision_at_threshold",
            "eval_recall_at_threshold",
            "eval_f1_at_threshold",
        ]:
            output[f"{column}_delta_vs_baseline"] = output[column] - float(base[column])
        output["eval_false_positive_at_threshold_delta_vs_baseline"] = (
            output["eval_false_positive_at_threshold"]
            - int(base["eval_false_positive_at_threshold"])
        )
        output["eval_false_negative_at_threshold_delta_vs_baseline"] = (
            output["eval_false_negative_at_threshold"]
            - int(base["eval_false_negative_at_threshold"])
        )
    return output


def add_final_test_deltas(final_test: pd.DataFrame) -> pd.DataFrame:
    return add_baseline_deltas(final_test, scope="final_test").sort_values(
        [
            "eval_f1_at_threshold",
            "eval_recall_at_threshold",
            "eval_pr_auc",
            "eval_precision_at_threshold",
        ],
        ascending=False,
    )


def select_watch_candidates(rolling_summary: pd.DataFrame) -> pd.DataFrame:
    baseline = rolling_summary.loc[rolling_summary["candidate_id"].eq("baseline_current")].iloc[0]
    candidates = rolling_summary.loc[~rolling_summary["candidate_id"].eq("baseline_current")].copy()
    return candidates.loc[
        candidates["eval_f1_at_threshold_mean"].ge(float(baseline["eval_f1_at_threshold_mean"]))
        & candidates["eval_recall_at_threshold_mean"].ge(
            float(baseline["eval_recall_at_threshold_mean"])
        )
        & candidates["total_false_negative"].le(int(baseline["total_false_negative"]))
    ].sort_values(
        [
            "eval_f1_at_threshold_mean",
            "eval_recall_at_threshold_mean",
            "total_false_negative",
            "total_false_positive",
        ],
        ascending=[False, False, True, True],
    )


def format_metric(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def format_signed(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    numeric = float(value)
    return f"{numeric:+.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(label for label, _, _ in columns) + " |"
    separator = "| " + " | ".join("---:" if kind != "text" else "---" for _, _, kind in columns) + " |"
    rows = [header, separator]
    for _, row in frame.iterrows():
        values = []
        for _, column, kind in columns:
            value = row.get(column, "")
            if kind == "metric":
                values.append(format_metric(value))
            elif kind == "signed":
                values.append(format_signed(value))
            elif kind == "int":
                values.append("" if pd.isna(value) else str(int(value)))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def build_summary_payload(
    *,
    rolling_summary: pd.DataFrame,
    final_test: pd.DataFrame,
    watch_candidates: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    baseline = rolling_summary.loc[rolling_summary["candidate_id"].eq("baseline_current")].iloc[0]
    best = rolling_summary.iloc[0]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": "feature_46_xgboost",
        "dataset": "credit_46_features",
        "base_feature_count": len(feature_columns),
        "risk_proxy_feature_count": len(RISK_PROXY_COLUMNS),
        "eval_years": ROLLING_EVAL_YEARS,
        "threshold_policy": f"max precision with policy-year recall >= {RECALL_FLOOR:.2f}",
        "monotonic_profiles": {
            name: {"direction_count": len(directions), "directions": directions}
            for name, directions in MONOTONIC_PROFILES.items()
        },
        "risk_proxy_specs": [
            {"source": source, "proxy": proxy, "direction": direction}
            for source, proxy, direction in RISK_PROXY_SPECS
        ],
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "rolling_baseline": baseline.to_dict(),
        "rolling_best": best.to_dict(),
        "watch_candidate_count": len(watch_candidates),
        "watch_candidates": watch_candidates.head(5).to_dict(orient="records"),
        "final_test": final_test.to_dict(orient="records"),
        "outputs": {
            "rolling_fold_metrics": ROLLING_FOLD_METRICS_FILENAME,
            "rolling_summary": ROLLING_SUMMARY_FILENAME,
            "final_test": FINAL_TEST_FILENAME,
            "report": REPORT_FILENAME,
        },
    }


def build_report(
    *,
    rolling_summary: pd.DataFrame,
    final_test: pd.DataFrame,
    watch_candidates: pd.DataFrame,
) -> str:
    if watch_candidates.empty:
        recommendation = (
            "rolling 기준에서 F1, Recall, FN 조건을 동시에 만족하는 승격 후보는 없습니다. "
            "공식 모델은 현재 46-feature baseline을 유지하는 편이 안전합니다."
        )
    else:
        top = watch_candidates.iloc[0]
        recommendation = (
            f"rolling 기준 watch 후보는 `{top['candidate_id']}`입니다. "
            "다만 final test와 오류 사례를 함께 확인한 뒤 공식 반영 여부를 결정해야 합니다."
        )

    profile_rows = pd.DataFrame(
        [
            {
                "profile": name,
                "direction_count": len(directions),
                "features": ", ".join(directions),
            }
            for name, directions in MONOTONIC_PROFILES.items()
        ]
        + [
            {
                "profile": "risk_proxy",
                "direction_count": len(RISK_PROXY_COLUMNS),
                "features": ", ".join(RISK_PROXY_COLUMNS),
            },
            {
                "profile": "risk_proxy_core",
                "direction_count": len(CORE_MONOTONIC_DIRECTIONS) + len(RISK_PROXY_COLUMNS),
                "features": ", ".join([*CORE_MONOTONIC_DIRECTIONS, *RISK_PROXY_COLUMNS]),
            },
        ]
    )
    risk_proxy_rows = pd.DataFrame(
        [
            {"source": source, "proxy": proxy, "orientation": direction}
            for source, proxy, direction in RISK_PROXY_SPECS
        ]
    )
    broad_constrained_features = pd.DataFrame(
        [
            {
                "feature": feature,
                "direction": "+1 위험 증가" if direction > 0 else "-1 위험 감소",
            }
            for feature, direction in BROAD_MONOTONIC_DIRECTIONS.items()
        ]
    )

    return "\n".join(
        [
            "# Feature 46 Time-Decay & Monotonic XGBoost Experiment",
            "",
            "## 결론",
            "",
            recommendation,
            "",
            "## Rolling OOT 요약",
            "",
            markdown_table(
                rolling_summary,
                [
                    ("Candidate", "candidate_id", "text"),
                    ("Features", "feature_count", "int"),
                    ("Constraints", "monotonic_constraint_count", "int"),
                    ("PR-AUC", "eval_pr_auc_mean", "metric"),
                    ("Precision", "eval_precision_at_threshold_mean", "metric"),
                    ("Recall", "eval_recall_at_threshold_mean", "metric"),
                    ("F1", "eval_f1_at_threshold_mean", "metric"),
                    ("F1 Δ", "eval_f1_at_threshold_mean_delta_vs_baseline", "signed"),
                    ("FP", "total_false_positive", "int"),
                    ("FN", "total_false_negative", "int"),
                    ("FN Δ", "total_false_negative_delta_vs_baseline", "int"),
                ],
            ),
            "",
            "## Final Test 참고",
            "",
            "Final test는 rolling 선택 이후의 참고 확인용입니다.",
            "",
            markdown_table(
                final_test,
                [
                    ("Candidate", "candidate_id", "text"),
                    ("Features", "feature_count", "int"),
                    ("Constraints", "monotonic_constraint_count", "int"),
                    ("PR-AUC", "eval_pr_auc", "metric"),
                    ("Precision", "eval_precision_at_threshold", "metric"),
                    ("Recall", "eval_recall_at_threshold", "metric"),
                    ("F1", "eval_f1_at_threshold", "metric"),
                    ("F1 Δ", "eval_f1_at_threshold_delta_vs_baseline", "signed"),
                    ("FP", "eval_false_positive_at_threshold", "int"),
                    ("FN", "eval_false_negative_at_threshold", "int"),
                    ("FN Δ", "eval_false_negative_at_threshold_delta_vs_baseline", "int"),
                ],
            ),
            "",
            "## Monotonic Profiles",
            "",
            markdown_table(
                profile_rows,
                [
                    ("Profile", "profile", "text"),
                    ("Directions", "direction_count", "int"),
                    ("Features", "features", "text"),
                ],
            ),
            "",
            "## Risk Proxy Features",
            "",
            "Risk proxy는 모두 값이 클수록 위험이 커지는 방향으로 맞춘 산업-연도 percentile입니다.",
            "",
            markdown_table(
                risk_proxy_rows,
                [
                    ("Source", "source", "text"),
                    ("Proxy", "proxy", "text"),
                    ("Orientation", "orientation", "text"),
                ],
            ),
            "",
            "## Broad Monotonic Constraint Reference",
            "",
            markdown_table(
                broad_constrained_features,
                [
                    ("Feature", "feature", "text"),
                    ("Constraint", "direction", "text"),
                ],
            ),
            "",
            "## 재생성 명령",
            "",
            "```bash",
            "/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_time_decay_monotonic_experiments.py",
            "```",
        ]
    )


def run_experiment(
    *,
    master: pd.DataFrame,
    feature_columns: list[str],
    eval_years: list[int],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], str]:
    fold_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    experiment_frame = add_risk_proxy_features(master)
    for spec in CANDIDATES:
        fold_rows.extend(
            evaluate_rolling_candidate(
                master=experiment_frame,
                base_columns=feature_columns,
                spec=spec,
                eval_years=eval_years,
                seed=seed,
            )
        )
        final_rows.append(
            evaluate_final_test(
                master=experiment_frame,
                base_columns=feature_columns,
                spec=spec,
                seed=seed,
            )
        )

    fold_metrics = pd.DataFrame(fold_rows)
    rolling_summary = summarize_rolling(fold_metrics)
    final_test = add_final_test_deltas(pd.DataFrame(final_rows))
    watch_candidates = select_watch_candidates(rolling_summary)
    summary = build_summary_payload(
        rolling_summary=rolling_summary,
        final_test=final_test,
        watch_candidates=watch_candidates,
        feature_columns=feature_columns,
    )
    report = build_report(
        rolling_summary=rolling_summary,
        final_test=final_test,
        watch_candidates=watch_candidates,
    )
    return fold_metrics, rolling_summary, final_test, watch_candidates, summary, report


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    master = read_master(args.master_path)
    feature_columns = read_feature_columns(args.feature_list_path, master)
    fold_metrics, rolling_summary, final_test, watch_candidates, summary, report = run_experiment(
        master=master,
        feature_columns=feature_columns,
        eval_years=args.eval_years,
        seed=args.seed,
    )

    fold_metrics.to_csv(output_dir / ROLLING_FOLD_METRICS_FILENAME, index=False, encoding="utf-8-sig")
    rolling_summary.to_csv(output_dir / ROLLING_SUMMARY_FILENAME, index=False, encoding="utf-8-sig")
    final_test.to_csv(output_dir / FINAL_TEST_FILENAME, index=False, encoding="utf-8-sig")
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / REPORT_FILENAME).write_text(report, encoding="utf-8")

    baseline = rolling_summary.loc[rolling_summary["candidate_id"].eq("baseline_current")].iloc[0]
    best = rolling_summary.iloc[0]
    print(f"Time-decay/monotonic experiment written to: {output_dir}")
    print(
        "Rolling baseline F1/Recall/FP/FN: "
        f"{baseline['eval_f1_at_threshold_mean']:.4f}/"
        f"{baseline['eval_recall_at_threshold_mean']:.4f}/"
        f"{int(baseline['total_false_positive'])}/"
        f"{int(baseline['total_false_negative'])}"
    )
    print(
        "Rolling best F1/Recall/FP/FN: "
        f"{best['candidate_id']} "
        f"{best['eval_f1_at_threshold_mean']:.4f}/"
        f"{best['eval_recall_at_threshold_mean']:.4f}/"
        f"{int(best['total_false_positive'])}/"
        f"{int(best['total_false_negative'])}"
    )
    print(f"Watch candidates: {len(watch_candidates)}")


if __name__ == "__main__":
    main()
