import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
LOCAL_TS2000_DIR = ROOT / "data" / "external" / "ts2000"
MIRROR_TS2000_DIR = PROJECT_ROOT / "03_Outputs" / "ts2000"

MASTER_DATASET_PATH = MIRROR_TS2000_DIR / "TS2000_Credit_Model_Dataset.csv"
MIRROR_MODEL_V1_DATASET_PATH = MIRROR_TS2000_DIR / "TS2000_Credit_Model_Dataset_Model_V1.csv"
LOCAL_MODEL_V1_DATASET_PATH = LOCAL_TS2000_DIR / "TS2000_Credit_Model_Dataset_Model_V1.csv"
MIRROR_AUDIT_EXPANDED_DATASET_PATH = MIRROR_TS2000_DIR / "TS2000_Credit_Model_Dataset_Model_V1_Audit_Expanded.csv"
LOCAL_AUDIT_EXPANDED_DATASET_PATH = LOCAL_TS2000_DIR / "TS2000_Credit_Model_Dataset_Model_V1_Audit_Expanded.csv"
MIRROR_MODEL_V1_MANIFEST_PATH = MIRROR_TS2000_DIR / "TS2000_Model_V1_Manifest.json"
LOCAL_MODEL_V1_MANIFEST_PATH = LOCAL_TS2000_DIR / "TS2000_Model_V1_Manifest.json"
MIRROR_CORE30_MANIFEST_PATH = MIRROR_TS2000_DIR / "TS2000_Model_Core30_Manifest.json"
LOCAL_CORE30_MANIFEST_PATH = LOCAL_TS2000_DIR / "TS2000_Model_Core30_Manifest.json"
MIRROR_CORE30_FEATURES_PATH = MIRROR_TS2000_DIR / "TS2000_Model_Core30_Features.csv"
LOCAL_CORE30_FEATURES_PATH = LOCAL_TS2000_DIR / "TS2000_Model_Core30_Features.csv"
MIRROR_METADATA_PATH = MIRROR_TS2000_DIR / "column_dictionary" / "ts2000_column_dictionary_metadata.json"
LOCAL_METADATA_PATH = LOCAL_TS2000_DIR / "column_dictionary" / "ts2000_column_dictionary_metadata.json"

AUDIT_EXTENSION_COLUMNS = ["audit_opinion_category"]
MACRO_CORE_COLUMNS = [
    "usd_krw",
    "market_spread",
    "spec_spread",
    "base_rate_diff",
    "market_spread_diff",
    "spec_spread_diff",
]
MACRO_RATE_LEVEL_COLUMNS = ["base_rate", "treasury_3y", "corp_aa_3y", "corp_bbb_3y"]
MARKET_CORE_COLUMNS = ["dividend_payer"]
MARKET_DEFERRED_COLUMNS = ["market_to_book"]
ACTIVITY_WORKING_CAPITAL_LEVEL_COLUMNS = [
    "accounts_receivable_ratio",
    "inventory_ratio",
    "contract_assets_ratio",
    "ar_days",
    "inventory_days",
    "ap_days",
]
ACTIVITY_WORKING_CAPITAL_TREND_COLUMNS = ["ar_days_diff", "inventory_days_diff", "ap_days_diff"]
TREND_DELTA_COLUMNS = [
    "operating_margin_diff",
    "ebitda_margin_diff",
    "equity_ratio_diff",
    "current_ratio_diff",
    "capital_impairment_diff",
    "ocf_to_total_borrowings_diff",
    "net_margin_diff",
    "cash_ratio_diff",
    "total_borrowings_ratio_diff",
    "ocf_to_total_liabilities_diff",
]
TREND_LAG_COLUMNS = ["lag1_current_ratio", "lag1_equity_ratio"]
TREND_TRANSITION_FLAG_COLUMNS = [
    "is_operating_income_turn_negative",
    "is_ocf_turn_negative",
    "is_current_ratio_below_1",
    "is_negative_equity_entry",
]
TREND_PERSISTENCE_FLAG_COLUMNS = [
    "is_2y_consecutive_operating_loss",
    "is_3y_consecutive_operating_loss",
    "is_2y_consecutive_ocf_deficit",
    "is_3y_consecutive_ocf_deficit",
    "is_zombie_3y",
]
TREND_VOLATILITY_COLUMNS = [
    "rolling_3y_cv_operating_margin",
    "rolling_3y_cv_ocf_to_total_borrowings",
    "rolling_3y_cv_revenue_growth",
]
TREND_QUALITY_CAPITAL_COLUMNS = [
    "accruals_ratio",
    "delta_accruals_ratio",
    "non_paid_in_equity_ratio",
    "delta_non_paid_in_equity_ratio",
    "delta_st_borrowings_share",
]

CORE30_FEATURE_GROUPS = {
    "context": [
        "market",
        "firm_size_group",
        "industry_macro_category",
    ],
    "audit_quality": [
        "audit_non_clean_flag",
        "audit_strong_risk_flag",
        "audit_missing_flag",
    ],
    "stability_leverage": [
        "current_ratio",
        "equity_ratio",
        "total_borrowings_ratio",
        "short_term_borrowings_share",
        "capital_impairment_ratio",
    ],
    "profitability_coverage": [
        "operating_margin",
        "interest_burden_ratio",
        "interest_coverage_ratio",
        "ocf_to_sales",
        "ocf_to_total_borrowings",
        "ocf_deficit_flag",
    ],
    "activity_structure": [
        "ar_days",
        "inventory_days",
        "ppe_ratio",
    ],
    "macro": [
        "market_spread",
        "spec_spread",
        "usd_krw",
    ],
    "market_shareholder": [
        "dividend_payer",
    ],
    "trend_early_warning": [
        "revenue_growth",
        "total_borrowings_growth",
        "operating_margin_diff",
        "lag1_current_ratio",
        "is_2y_consecutive_operating_loss",
        "is_2y_consecutive_ocf_deficit",
    ],
}

CORE30_SELECTION_NOTES = {
    "market": "시장별 모집단 차이를 반영하는 기본 범주형",
    "firm_size_group": "기업 규모에 따른 위험수준 차이를 반영",
    "industry_macro_category": "산업 대분류별 구조적 차이를 반영",
    "audit_non_clean_flag": "비적정/한정 등 품질 저하 신호",
    "audit_strong_risk_flag": "부적정·의견거절 등 강한 위험 신호",
    "audit_missing_flag": "감사정보 누락 자체의 리스크 신호",
    "current_ratio": "단기 유동성 핵심 지표",
    "equity_ratio": "자기자본 완충력 핵심 지표",
    "total_borrowings_ratio": "총차입금 부담 수준",
    "short_term_borrowings_share": "차입구조의 단기 의존도",
    "capital_impairment_ratio": "자본잠식 위험 수준",
    "operating_margin": "본업 수익성 대표 변수",
    "interest_burden_ratio": "매출 대비 금융비용 부담",
    "interest_coverage_ratio": "이자 상환능력 대표 지표",
    "ocf_to_sales": "현금창출력 대표 지표",
    "ocf_to_total_borrowings": "차입금 대비 현금상환여력",
    "ocf_deficit_flag": "현금흐름 적자 여부",
    "ar_days": "매출채권 회수 속도",
    "inventory_days": "재고 회전 속도",
    "ppe_ratio": "자산구조상 설비집약도",
    "market_spread": "시장 전체 위험 프리미엄",
    "spec_spread": "투자적격-투기경계 스트레스",
    "usd_krw": "환율 부담",
    "dividend_payer": "배당정책을 통한 기업 성숙도·안정성 신호",
    "revenue_growth": "외형 축소 또는 성장 둔화",
    "total_borrowings_growth": "부채 확대 속도",
    "operating_margin_diff": "수익성 악화 속도",
    "lag1_current_ratio": "직전연도 유동성 상태",
    "is_2y_consecutive_operating_loss": "2년 연속 영업손실 여부",
    "is_2y_consecutive_ocf_deficit": "2년 연속 OCF 적자 여부",
}


def columns_by_role(columns_meta: list[dict[str, str]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in columns_meta:
        grouped.setdefault(row["usage_role"], []).append(row["variable_name"])
    return grouped


def ordered_subset(all_columns: list[str], keep_columns: set[str]) -> list[str]:
    return [column for column in all_columns if column in keep_columns]


def existing_subset(all_columns: list[str], candidates: list[str]) -> list[str]:
    available = set(all_columns)
    return [column for column in candidates if column in available]


def feature_subgroup_maps(all_columns: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    subgroup_candidates = {
        "audit_extension_candidate": AUDIT_EXTENSION_COLUMNS,
        "macro_core": MACRO_CORE_COLUMNS,
        "macro_rate_level": MACRO_RATE_LEVEL_COLUMNS,
        "market_core": MARKET_CORE_COLUMNS,
        "market_deferred": MARKET_DEFERRED_COLUMNS,
        "working_capital_level": ACTIVITY_WORKING_CAPITAL_LEVEL_COLUMNS,
        "working_capital_trend": ACTIVITY_WORKING_CAPITAL_TREND_COLUMNS,
        "trend_delta_change": TREND_DELTA_COLUMNS,
        "trend_lag": TREND_LAG_COLUMNS,
        "trend_transition_flag": TREND_TRANSITION_FLAG_COLUMNS,
        "trend_persistence_flag": TREND_PERSISTENCE_FLAG_COLUMNS,
        "trend_volatility_cv": TREND_VOLATILITY_COLUMNS,
        "trend_earnings_quality_capital": TREND_QUALITY_CAPITAL_COLUMNS,
    }
    group_by_subgroup = {
        "audit_extension_candidate": "audit",
        "macro_core": "macro",
        "macro_rate_level": "macro",
        "market_core": "market_features",
        "market_deferred": "market_features",
        "working_capital_level": "activity",
        "working_capital_trend": "activity",
        "trend_delta_change": "trend",
        "trend_lag": "trend",
        "trend_transition_flag": "trend",
        "trend_persistence_flag": "trend",
        "trend_volatility_cv": "trend",
        "trend_earnings_quality_capital": "trend",
    }

    subgroup_by_column: dict[str, str] = {}
    group_by_column: dict[str, str] = {}
    for subgroup, candidates in subgroup_candidates.items():
        for column in existing_subset(all_columns, candidates):
            subgroup_by_column[column] = subgroup
            group_by_column[column] = group_by_subgroup[subgroup]
    return group_by_column, subgroup_by_column


def write_csv_both(df: pd.DataFrame, *paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json_both(payload: dict, *paths: Path) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def build_core30_feature_columns(all_columns: list[str]) -> list[str]:
    feature_columns = []
    available = set(all_columns)
    for group_columns in CORE30_FEATURE_GROUPS.values():
        for column in group_columns:
            if column not in available:
                raise ValueError(f"Core30 column is missing from master dataset: {column}")
            feature_columns.append(column)
    if len(feature_columns) != 30:
        raise ValueError(f"Core30 must contain exactly 30 columns, found {len(feature_columns)}")
    return feature_columns


def build_exports() -> None:
    master_df = pd.read_csv(MASTER_DATASET_PATH)
    metadata = json.loads(MIRROR_METADATA_PATH.read_text(encoding="utf-8"))
    columns_meta = metadata["columns"]
    role_columns = columns_by_role(columns_meta)
    all_columns = master_df.columns.tolist()
    feature_x_columns = role_columns["feature_x"]

    audit_extension_columns = existing_subset(all_columns, AUDIT_EXTENSION_COLUMNS)
    model_feature_columns = [column for column in feature_x_columns if column not in audit_extension_columns]
    audit_expanded_feature_columns = model_feature_columns + audit_extension_columns

    macro_core_columns = existing_subset(all_columns, MACRO_CORE_COLUMNS)
    macro_rate_level_columns = existing_subset(all_columns, MACRO_RATE_LEVEL_COLUMNS)
    market_core_columns = existing_subset(all_columns, MARKET_CORE_COLUMNS)
    market_deferred_columns = existing_subset(all_columns, MARKET_DEFERRED_COLUMNS)
    working_capital_level_columns = existing_subset(all_columns, ACTIVITY_WORKING_CAPITAL_LEVEL_COLUMNS)
    working_capital_trend_columns = existing_subset(all_columns, ACTIVITY_WORKING_CAPITAL_TREND_COLUMNS)
    trend_delta_columns = existing_subset(all_columns, TREND_DELTA_COLUMNS)
    trend_lag_columns = existing_subset(all_columns, TREND_LAG_COLUMNS)
    trend_transition_flag_columns = existing_subset(all_columns, TREND_TRANSITION_FLAG_COLUMNS)
    trend_persistence_flag_columns = existing_subset(all_columns, TREND_PERSISTENCE_FLAG_COLUMNS)
    trend_volatility_columns = existing_subset(all_columns, TREND_VOLATILITY_COLUMNS)
    trend_quality_capital_columns = existing_subset(all_columns, TREND_QUALITY_CAPITAL_COLUMNS)
    group_by_column, subgroup_by_column = feature_subgroup_maps(all_columns)

    model_keep = set(
        model_feature_columns
        + role_columns["id_reference"]
        + role_columns["time_reference"]
        + role_columns["target_y"]
    )
    audit_expanded_keep = set(
        audit_expanded_feature_columns
        + role_columns["id_reference"]
        + role_columns["time_reference"]
        + role_columns["target_y"]
    )
    model_v1_columns = ordered_subset(all_columns, model_keep)
    audit_expanded_columns = ordered_subset(all_columns, audit_expanded_keep)

    model_v1_df = master_df.loc[:, model_v1_columns]
    audit_expanded_df = master_df.loc[:, audit_expanded_columns]
    write_csv_both(model_v1_df, MIRROR_MODEL_V1_DATASET_PATH, LOCAL_MODEL_V1_DATASET_PATH)
    write_csv_both(
        audit_expanded_df,
        MIRROR_AUDIT_EXPANDED_DATASET_PATH,
        LOCAL_AUDIT_EXPANDED_DATASET_PATH,
    )

    model_v1_manifest = {
        "dataset_file": "TS2000_Credit_Model_Dataset_Model_V1.csv",
        "dataset_variants": {
            "model_v1_baseline": {
                "dataset_file": "TS2000_Credit_Model_Dataset_Model_V1.csv",
                "description": "기본셋. audit_opinion_category를 제외하고 감사의견 flag 4개만 유지합니다.",
            },
            "model_v1_audit_expanded": {
                "dataset_file": "TS2000_Credit_Model_Dataset_Model_V1_Audit_Expanded.csv",
                "description": "기본셋에 audit_opinion_category를 추가한 확장셋입니다.",
                "added_columns": audit_extension_columns,
            },
        },
        "row_count": len(master_df),
        "feature_column_count": len(model_feature_columns),
        "feature_columns": model_feature_columns,
        "audit_expanded_feature_column_count": len(audit_expanded_feature_columns),
        "audit_expanded_feature_columns": audit_expanded_feature_columns,
        "audit_extension_columns": audit_extension_columns,
        "id_columns": role_columns["id_reference"],
        "time_columns": role_columns["time_reference"],
        "target_column": role_columns["target_y"][0],
        "macro_feature_groups": {
            "macro_basic_columns": macro_core_columns,
            "macro_expanded_additional_columns": macro_rate_level_columns,
        },
        "market_feature_groups": {
            "market_core_columns": market_core_columns,
            "market_deferred_columns": market_deferred_columns,
        },
        "activity_feature_groups": {
            "working_capital_level_columns": working_capital_level_columns,
            "working_capital_trend_columns": working_capital_trend_columns,
        },
        "trend_feature_groups": {
            "delta_change_columns": trend_delta_columns,
            "lag_columns": trend_lag_columns,
            "transition_flag_columns": trend_transition_flag_columns,
            "persistence_flag_columns": trend_persistence_flag_columns,
            "volatility_cv_columns": trend_volatility_columns,
            "earnings_quality_capital_columns": trend_quality_capital_columns,
        },
        "feature_groups_by_column": group_by_column,
        "feature_subgroups_by_column": subgroup_by_column,
        "deferred_feature_columns": role_columns["feature_deferred"],
        "excluded_from_model_v1": ["credit_rating"]
        + role_columns["analysis_only_label"]
        + role_columns["feature_deferred"]
        + audit_extension_columns,
        "recommended_experiments": [
            "Model V1 baseline",
            "Model V1 + audit_opinion_category",
            "Model V1 + dividend_payer 유지 / market_to_book 보류 비교",
        ],
        "note": (
            "Model_V1 dataset keeps reference keys for traceability. "
            "When building X, use feature_columns only and keep target_column as y. "
            "audit_opinion_category는 기본셋에서 제외하고 audit 확장셋에서만 비교합니다. "
            "dividend_payer는 공식 입력변수로 승격했고, market_to_book과 ppi_yoy는 feature_deferred로 관리합니다."
        ),
    }
    write_json_both(model_v1_manifest, MIRROR_MODEL_V1_MANIFEST_PATH, LOCAL_MODEL_V1_MANIFEST_PATH)

    core30_feature_columns = build_core30_feature_columns(all_columns)
    core30_manifest = {
        "manifest_name": "TS2000 Core 30",
        "description": (
            "멘토 피드백을 반영해 1차 베이스라인 모델에서 우선 사용할 핵심 30개 feature set입니다. "
            "중복이 큰 raw 규모 변수와 유사 금리 레벨 변수는 제외하고, 비율/스프레드/조기경보 성격의 변수 중심으로 구성했습니다. "
            "이번 개정판에서는 dividend_payer를 공식 포함하고, ppi_yoy는 보류 변수로 이동했습니다."
        ),
        "dataset_file": "TS2000_Credit_Model_Dataset_Model_V1.csv",
        "target_column": "is_speculative",
        "id_columns": role_columns["id_reference"],
        "time_columns": role_columns["time_reference"],
        "recommended_categorical_columns": [
            "market",
            "firm_size_group",
            "industry_macro_category",
        ],
        "feature_groups": CORE30_FEATURE_GROUPS,
        "feature_columns": core30_feature_columns,
        "core30_feature_columns": core30_feature_columns,
        "deferred_candidates_to_compare_next": ["market_to_book", "ppi_yoy"],
        "notes": [
            "1차 XGBoost baseline은 core30_feature_columns만 사용하고, target_column을 y로 둡니다.",
            "raw 금액 변수(revenue, assets_total 등)는 규모 효과와 높은 상관 때문에 Core 30에서는 제외했습니다.",
            "base_rate, treasury_3y, corp_aa_3y, corp_bbb_3y는 상호상관이 높아 1차에서는 스프레드/환율 중심으로 유지했습니다.",
            "audit_opinion_category는 Core 30에서 제외하고, 별도 확장 실험에서만 추가 비교합니다.",
            "dividend_payer는 공식 Core30에 포함하고, market_to_book은 보류 변수로 관리합니다.",
        ],
    }
    write_json_both(core30_manifest, MIRROR_CORE30_MANIFEST_PATH, LOCAL_CORE30_MANIFEST_PATH)

    core30_rows = []
    for feature_group, columns in CORE30_FEATURE_GROUPS.items():
        for column in columns:
            core30_rows.append(
                {
                    "feature_group": feature_group,
                    "feature_name": column,
                    "selection_note": CORE30_SELECTION_NOTES[column],
                }
            )
    core30_df = pd.DataFrame(core30_rows)
    write_csv_both(core30_df, MIRROR_CORE30_FEATURES_PATH, LOCAL_CORE30_FEATURES_PATH)

    metadata["summary"]["full_dataset_file"] = "TS2000_Credit_Model_Dataset.csv"
    metadata["summary"]["model_dataset_file"] = "TS2000_Credit_Model_Dataset_Model_V1.csv"
    metadata["summary"]["audit_expanded_dataset_file"] = "TS2000_Credit_Model_Dataset_Model_V1_Audit_Expanded.csv"
    metadata["summary"]["model_v1_manifest_file"] = "TS2000_Model_V1_Manifest.json"
    metadata["summary"]["core30_manifest_file"] = "TS2000_Model_Core30_Manifest.json"
    metadata["summary"]["core30_feature_file"] = "TS2000_Model_Core30_Features.csv"
    metadata["summary"]["row_count"] = len(master_df)
    metadata["summary"]["full_column_count"] = int(master_df.shape[1])
    metadata["summary"]["model_column_count"] = len(model_v1_columns)
    metadata["summary"]["audit_expanded_column_count"] = len(audit_expanded_columns)
    metadata["summary"]["model_feature_count"] = len(model_feature_columns)
    metadata["summary"]["audit_expanded_feature_count"] = len(audit_expanded_feature_columns)
    metadata["summary"]["feature_deferred_count"] = len(role_columns["feature_deferred"])
    metadata["summary"]["core30_feature_count"] = len(core30_feature_columns)
    metadata["summary"]["dataset_variant_note"] = (
        "TS2000_Credit_Model_Dataset.csv는 전체 마스터입니다. "
        "TS2000_Credit_Model_Dataset_Model_V1.csv는 1차 기본셋으로 "
        "credit_rating, analysis-only labels, feature_deferred, audit_opinion_category를 제외했습니다. "
        "TS2000_Credit_Model_Dataset_Model_V1_Audit_Expanded.csv는 기본셋에 audit_opinion_category를 추가한 비교용 파일입니다. "
        "Core30 실험은 TS2000_Model_Core30_Manifest.json 기준으로 진행합니다."
    )
    metadata["summary"]["two_file_note"] = metadata["summary"]["dataset_variant_note"]

    metadata["source_files"] = [
        {
            "source": "data/external/ts2000/TS2000_Credit_Model_Dataset.csv",
            "purpose": "전체 마스터 데이터셋 (참조/사후분석/보류변수 포함)",
        },
        {
            "source": "data/external/ts2000/TS2000_Credit_Model_Dataset_Model_V1.csv",
            "purpose": "1차 모델 기본셋 (dividend_payer 포함, audit_opinion_category 제외)",
        },
        {
            "source": "data/external/ts2000/TS2000_Credit_Model_Dataset_Model_V1_Audit_Expanded.csv",
            "purpose": "1차 모델 확장셋 (기본셋 + audit_opinion_category)",
        },
        {
            "source": "data/external/ts2000/TS2000_Model_V1_Manifest.json",
            "purpose": "1차 모델의 feature/id/time/target 컬럼 그룹 정의와 변수군 메모",
        },
        {
            "source": "data/external/ts2000/TS2000_Model_Core30_Manifest.json",
            "purpose": "Core30 공식 입력변수 목록과 실험 메모",
        },
        {
            "source": "data/external/ts2000/TS2000_Model_Core30_Features.csv",
            "purpose": "Core30 변수 리스트와 선정 이유",
        },
        {
            "source": "data/external/ts2000/column_dictionary/TS2000_Credit_Model_Column_Dictionary.xlsx",
            "purpose": "컬럼 설명 워크북",
        },
        {
            "source": "data/external/ts2000/column_dictionary/ts2000_column_dictionary_metadata.json",
            "purpose": "컬럼 역할 및 메타데이터 JSON",
        },
    ]

    write_json_both(metadata, MIRROR_METADATA_PATH, LOCAL_METADATA_PATH)

    print(f"Saved master dataset source: {MASTER_DATASET_PATH}")
    print(f"Saved model_v1 dataset: {MIRROR_MODEL_V1_DATASET_PATH}")
    print(f"Saved audit expanded dataset: {MIRROR_AUDIT_EXPANDED_DATASET_PATH}")
    print(f"Saved model_v1 manifest: {MIRROR_MODEL_V1_MANIFEST_PATH}")
    print(f"Saved core30 manifest: {MIRROR_CORE30_MANIFEST_PATH}")


if __name__ == "__main__":
    build_exports()
