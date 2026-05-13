from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input" / "credit_43_features"
FEATURE_MASTER_PATH = INPUT_DIR / "feature_43_master.csv"
FEATURE_SPEC_PATH = INPUT_DIR / "feature_43_list.json"
CANONICAL_INFERENCE_PATH = INPUT_DIR / "feature_43_inference_2026.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and materialize the CAS-internal 2026 inference table. "
            "This script intentionally reads only files inside this repository."
        )
    )
    parser.add_argument("--source", type=Path, default=CANONICAL_INFERENCE_PATH)
    parser.add_argument("--output", type=Path, default=CANONICAL_INFERENCE_PATH)
    parser.add_argument("--target-fiscal-year", type=int, default=2025)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def load_expected_columns() -> tuple[list[str], list[str]]:
    master_columns = pd.read_csv(
        FEATURE_MASTER_PATH,
        nrows=1,
        encoding="utf-8-sig",
    ).columns.tolist()
    expected_columns = [column for column in master_columns if column != "is_speculative"]

    feature_spec = json.loads(FEATURE_SPEC_PATH.read_text(encoding="utf-8"))
    model_features = [str(name) for name in feature_spec["model_features"]]
    missing_model_features = [column for column in model_features if column not in expected_columns]
    if missing_model_features:
        raise ValueError(
            "feature_43_list.json contains model features missing from inference columns: "
            f"{missing_model_features}"
        )

    return expected_columns, model_features


def validate_inference_frame(
    frame: pd.DataFrame,
    *,
    expected_columns: list[str],
    model_features: list[str],
    target_fiscal_year: int,
) -> None:
    if "is_speculative" in frame.columns:
        raise ValueError("Inference table must not contain the labeled target column.")

    missing_columns = [column for column in expected_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Inference table is missing required columns: {missing_columns}")

    extra_columns = [column for column in frame.columns if column not in expected_columns]
    if extra_columns:
        raise ValueError(f"Inference table contains unexpected columns: {extra_columns}")

    if frame.empty:
        raise ValueError("Inference table is empty.")

    fiscal_year = pd.to_numeric(frame["fiscal_year"], errors="coerce")
    eval_year = pd.to_numeric(frame["eval_year"], errors="coerce")
    invalid_fiscal_year = fiscal_year.ne(target_fiscal_year) | fiscal_year.isna()
    invalid_eval_year = eval_year.ne(target_fiscal_year + 1) | eval_year.isna()
    if invalid_fiscal_year.any():
        bad_values = sorted(fiscal_year.loc[invalid_fiscal_year].dropna().astype(int).unique())
        raise ValueError(
            f"Inference table must contain only fiscal_year={target_fiscal_year}; "
            f"found {bad_values}"
        )
    if invalid_eval_year.any():
        bad_values = sorted(eval_year.loc[invalid_eval_year].dropna().astype(int).unique())
        raise ValueError(
            f"Inference table must contain only eval_year={target_fiscal_year + 1}; "
            f"found {bad_values}"
        )

    duplicate_count = int(frame.duplicated(["market", "stock_code", "fiscal_year"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Inference table has duplicate market/stock_code/fiscal_year rows: {duplicate_count}"
        )

    null_model_feature_counts = frame[model_features].isna().sum()
    all_null_features = [
        column for column, count in null_model_feature_counts.items() if int(count) == len(frame)
    ]
    if all_null_features:
        raise ValueError(f"Model features are entirely missing: {all_null_features}")


def main() -> None:
    args = parse_args()
    expected_columns, model_features = load_expected_columns()
    inference = pd.read_csv(args.source, encoding="utf-8-sig")
    validate_inference_frame(
        inference,
        expected_columns=expected_columns,
        model_features=model_features,
        target_fiscal_year=args.target_fiscal_year,
    )

    ordered = inference.loc[:, expected_columns].copy()
    if not args.check_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ordered.to_csv(args.output, index=False, encoding="utf-8-sig")

    action = "Checked" if args.check_only else "Saved"
    print(f"[{action}] {args.output} ({len(ordered):,} rows)")
    print(
        ordered[["market", "stock_code", "corp_name", "fiscal_year", "eval_year"]]
        .head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
