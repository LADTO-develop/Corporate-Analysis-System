from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input" / "credit_43_features"
SOURCE_DATASET_PATH = ROOT / "data" / "raw" / "ts2000" / "TS2000_Credit_Model_Dataset_Model_V1.csv"
FEATURE_SPEC_PATH = INPUT_DIR / "feature_43_list.json"
MASTER_PATH = INPUT_DIR / "feature_43_master.csv"

ID_COLUMNS = [
    "market",
    "stock_code",
    "corp_name",
    "fiscal_year",
    "eval_year",
    "firm_size_group",
    "industry_macro_category",
]
ID_SPLIT_COLUMNS = [*ID_COLUMNS, "label_eval_year"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the credit_43_features input tables from the latest TS2000 Model_V1 dataset."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_DATASET_PATH,
        help="Latest labeled company-year dataset to use as the feature_43 source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory where feature_43 master/split CSVs will be written.",
    )
    return parser.parse_args()


def load_feature_spec(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_master_frame(source: pd.DataFrame, feature_spec: dict[str, object]) -> pd.DataFrame:
    selected_source_features = [
        str(feature)
        for feature in feature_spec["selected_source_features"]  # type: ignore[index]
    ]
    model_features = [str(feature) for feature in feature_spec["model_features"]]  # type: ignore[index]
    categorical = {
        str(feature)
        for feature in feature_spec["categorical_one_hot_columns"]  # type: ignore[index]
    }

    raw_feature_columns = [
        feature for feature in selected_source_features if feature not in categorical
    ]
    required_columns = [*ID_COLUMNS, *raw_feature_columns, "is_speculative"]
    missing_columns = [column for column in required_columns if column not in source.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in source dataset: {missing_columns}")

    master = source.loc[:, required_columns].copy()

    # The 43-feature spec keeps categorical source values on the row while also
    # materializing one-hot columns that the model consumes directly.
    for feature in model_features:
        if feature.startswith("market_"):
            category = feature.removeprefix("market_")
            master[feature] = (master["market"].astype(str) == category).astype(int)
        elif feature.startswith("firm_size_group_"):
            category = feature.removeprefix("firm_size_group_")
            master[feature] = (master["firm_size_group"].astype(str) == category).astype(int)
        elif feature.startswith("industry_macro_category_"):
            category = feature.removeprefix("industry_macro_category_")
            master[feature] = (master["industry_macro_category"].astype(str) == category).astype(
                int
            )

    ordered_columns = [
        *ID_COLUMNS,
        *raw_feature_columns,
        *[feature for feature in model_features if feature not in raw_feature_columns],
        "is_speculative",
    ]
    for column in ordered_columns:
        if column not in master.columns:
            master[column] = pd.NA

    master = (
        master.loc[:, ordered_columns]
        .sort_values(["market", "stock_code", "fiscal_year", "eval_year"])
        .reset_index(drop=True)
    )
    return master


def split_master(
    master: pd.DataFrame, model_features: list[str]
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    split_masks = {
        "train": master["fiscal_year"] <= 2021,
        "valid": master["fiscal_year"] == 2022,
        "test": master["fiscal_year"] >= 2023,
    }
    outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for split_name, mask in split_masks.items():
        frame = master.loc[mask].copy()
        ready = frame.loc[:, [*model_features, "is_speculative"]].copy()
        ids = frame.loc[:, ID_COLUMNS].copy()
        ids["label_eval_year"] = ids["eval_year"]
        outputs[split_name] = (ready, ids.loc[:, ID_SPLIT_COLUMNS])
    return outputs


def write_outputs(
    output_dir: Path,
    master: pd.DataFrame,
    split_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_dir / "feature_43_master.csv", index=False, encoding="utf-8-sig")

    for split_name, (ready, ids) in split_outputs.items():
        ready.to_csv(output_dir / f"xgb_{split_name}.csv", index=False, encoding="utf-8-sig")
        ids.to_csv(output_dir / f"xgb_id_{split_name}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    feature_spec = load_feature_spec(FEATURE_SPEC_PATH)
    source = pd.read_csv(args.source, encoding="utf-8-sig", low_memory=False)
    master = build_master_frame(source, feature_spec)
    model_features = [str(feature) for feature in feature_spec["model_features"]]  # type: ignore[index]
    split_outputs = split_master(master, model_features)
    write_outputs(args.output_dir, master, split_outputs)

    print(f"[Saved] {args.output_dir / 'feature_43_master.csv'} ({len(master):,} rows)")
    for split_name, (ready, _) in split_outputs.items():
        positive = int(ready["is_speculative"].sum())
        print(
            f"[Split] {split_name}: rows={len(ready):,}, "
            f"positive={positive:,}, positive_rate={ready['is_speculative'].mean():.4f}"
        )


if __name__ == "__main__":
    main()
