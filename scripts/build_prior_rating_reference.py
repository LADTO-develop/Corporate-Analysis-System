"""Build a non-leaky prior credit-rating reference for Stage 2 review.

The target-label reference answers "what rating did the company receive in the
evaluation year?"  This prior reference answers a different question: "what was
the latest rating already public as of the financial-statement cutoff date?"

Use this file for agent inputs and boundary-rating context. Do not use future
target labels from the same evaluation year as Stage 2 inputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = ROOT / "data/input/credit_46_features/feature_46_master.csv"
INFERENCE_2026_PATH = ROOT / "data/input/credit_46_features/feature_46_inference_2026.csv"
RATING_HISTORY_PATH = ROOT / "data/evaluation/target_label_reference.csv"
OUTPUT_PATH = ROOT / "data/evaluation/prior_rating_reference.csv"
SUMMARY_PATH = ROOT / "data/evaluation/prior_rating_reference_summary.json"

KEY_COLUMNS = ["market", "stock_code", "corp_name", "fiscal_year", "eval_year"]
EXACT_BOUNDARY_RATINGS = {"BBB-", "BB+"}
NEAR_BOUNDARY_RATINGS = {"BBB", "BB"}

OUTPUT_COLUMNS = [
    "universe",
    "market",
    "stock_code",
    "corp_name",
    "fiscal_year",
    "eval_year",
    "as_of_date",
    "has_prior_rating",
    "prior_credit_rating",
    "prior_credit_rating_rank",
    "prior_is_speculative",
    "prior_rating_boundary_group",
    "prior_rating_date",
    "prior_rating_age_days",
    "prior_rating_agency",
    "prior_rating_agency_group",
    "prior_rating_target",
    "prior_source_eval_year",
    "prior_source_label_set",
    "prior_selection_scope",
    "prior_selection_rule",
    "prior_reference_rule",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument("--inference-2026", type=Path, default=INFERENCE_2026_PATH)
    parser.add_argument("--rating-history", type=Path, default=RATING_HISTORY_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    return parser.parse_args()


def normalize_stock_code(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype("string").str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    )
    is_numeric_code = normalized.str.fullmatch(r"\d{1,6}").fillna(False)
    return normalized.where(~is_numeric_code, normalized.str.zfill(6))


def read_universe(path: Path, universe: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    frame = frame.loc[:, KEY_COLUMNS].copy()
    frame["universe"] = universe
    frame["stock_code"] = normalize_stock_code(frame["stock_code"])
    for column in ["fiscal_year", "eval_year"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame["as_of_date"] = frame["fiscal_year"].astype(str) + "-12-31"
    frame["as_of_date_dt"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    return frame.drop_duplicates(["universe", *KEY_COLUMNS]).reset_index(drop=True)


def read_rating_history(path: Path) -> pd.DataFrame:
    history = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    history = history.copy()
    history["stock_code"] = normalize_stock_code(history["stock_code"])
    history["rating_date_dt"] = pd.to_datetime(history["rating_date"], errors="coerce")
    history["credit_rating_rank"] = pd.to_numeric(
        history["credit_rating_rank"], errors="coerce"
    ).astype("Int64")
    history["eval_year"] = pd.to_numeric(history["eval_year"], errors="coerce").astype("Int64")
    history["is_speculative"] = pd.to_numeric(history["is_speculative"], errors="coerce").astype(
        "Int64"
    )
    return history.dropna(subset=["rating_date_dt", "credit_rating", "credit_rating_rank"])


def classify_rating_boundary(rating: object, rank: object) -> str:
    rating_text = str(rating).strip()
    if rating_text in EXACT_BOUNDARY_RATINGS:
        return "exact_bbb_minus_bb_plus_boundary"
    if rating_text in NEAR_BOUNDARY_RATINGS:
        return "near_bbb_bb_boundary"

    numeric_rank = pd.to_numeric(rank, errors="coerce")
    if pd.isna(numeric_rank):
        return "unknown"
    if int(numeric_rank) <= 10:
        return "investment_grade_non_boundary"
    return "speculative_grade_non_boundary"


def select_prior_ratings(universe: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    universe = universe.copy().reset_index(drop=True)
    universe["_row_id"] = range(len(universe))

    candidates = universe.merge(
        history,
        on="stock_code",
        how="left",
        suffixes=("", "_history"),
        validate="many_to_many",
    )
    candidates = candidates.loc[
        candidates["rating_date_dt"].notna()
        & candidates["as_of_date_dt"].notna()
        & (candidates["rating_date_dt"] <= candidates["as_of_date_dt"])
    ].copy()

    candidates = candidates.sort_values(
        ["_row_id", "rating_date_dt", "credit_rating_rank"],
        ascending=[True, False, False],
    )
    selected = candidates.drop_duplicates("_row_id", keep="first")

    selected = selected.rename(
        columns={
            "credit_rating": "prior_credit_rating",
            "credit_rating_rank": "prior_credit_rating_rank",
            "is_speculative": "prior_is_speculative",
            "rating_date": "prior_rating_date",
            "rating_agency": "prior_rating_agency",
            "rating_agency_group": "prior_rating_agency_group",
            "rating_target": "prior_rating_target",
            "eval_year_history": "prior_source_eval_year",
            "source_label_set": "prior_source_label_set",
            "selection_scope": "prior_selection_scope",
            "selection_rule": "prior_selection_rule",
        }
    )
    prior_columns = [
        "_row_id",
        "prior_credit_rating",
        "prior_credit_rating_rank",
        "prior_is_speculative",
        "prior_rating_date",
        "prior_rating_agency",
        "prior_rating_agency_group",
        "prior_rating_target",
        "prior_source_eval_year",
        "prior_source_label_set",
        "prior_selection_scope",
        "prior_selection_rule",
        "rating_date_dt",
    ]
    selected = selected.loc[:, prior_columns].copy()

    result = universe.merge(selected, on="_row_id", how="left", validate="one_to_one")
    result["has_prior_rating"] = result["prior_credit_rating"].notna()
    result["prior_rating_boundary_group"] = result.apply(
        lambda row: classify_rating_boundary(
            row["prior_credit_rating"], row["prior_credit_rating_rank"]
        )
        if row["has_prior_rating"]
        else "no_prior_rating",
        axis=1,
    )
    result["prior_rating_age_days"] = (
        result["as_of_date_dt"] - result["rating_date_dt"]
    ).dt.days.astype("Int64")
    result["prior_reference_rule"] = result["has_prior_rating"].map(
        {
            True: "latest_public_rating_on_or_before_fiscal_year_end_then_worst_if_tied",
            False: "no_public_rating_on_or_before_fiscal_year_end",
        }
    )

    result = result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["universe", "eval_year", "market", "stock_code"]
    )
    return result.reset_index(drop=True)


def build_summary(
    reference: pd.DataFrame, history: pd.DataFrame, rating_history_path: Path
) -> dict[str, object]:
    matched = reference.loc[reference["has_prior_rating"]]
    rows_by_universe = reference.groupby("universe").size().astype(int).to_dict()
    matched_by_universe = matched.groupby("universe").size().astype(int).to_dict()
    coverage_by_universe = {
        str(universe): round(matched_by_universe.get(universe, 0) / total, 4)
        for universe, total in rows_by_universe.items()
    }
    coverage_by_eval_year = (
        reference.groupby(["universe", "eval_year"])["has_prior_rating"]
        .mean()
        .round(4)
        .rename("coverage")
        .reset_index()
    )

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "purpose": (
            "Non-leaky prior credit-rating reference for Stage 2 agent inputs. "
            "Each row keeps only ratings public on or before fiscal_year-12-31."
        ),
        "as_of_policy": "as_of_date = fiscal_year-12-31",
        "selection_rule": "latest rating_date on/before as_of_date; if tied, choose worst rating",
        "source_rating_history": str(rating_history_path.relative_to(ROOT)),
        "source_rating_history_rows": len(history),
        "rows": len(reference),
        "rows_by_universe": {str(k): int(v) for k, v in rows_by_universe.items()},
        "matched_prior_rows_by_universe": {str(k): int(v) for k, v in matched_by_universe.items()},
        "coverage_by_universe": coverage_by_universe,
        "coverage_by_universe_eval_year": [
            {
                "universe": str(row.universe),
                "eval_year": int(row.eval_year),
                "coverage": float(row.coverage),
            }
            for row in coverage_by_eval_year.itertuples(index=False)
        ],
        "prior_rating_boundary_group_counts": reference["prior_rating_boundary_group"]
        .value_counts()
        .astype(int)
        .to_dict(),
        "prior_rating_agency_group_counts": matched["prior_rating_agency_group"]
        .fillna("UNKNOWN")
        .value_counts()
        .astype(int)
        .to_dict(),
        "columns": OUTPUT_COLUMNS,
        "leakage_policy": (
            "This file may be used as Stage 2 context because it excludes ratings "
            "published after the row-level as_of_date. It is separate from target "
            "labels used to score model or committee correctness."
        ),
    }


def main() -> None:
    args = parse_args()
    master = read_universe(args.master, "model_v1")
    inference_2026 = read_universe(args.inference_2026, "inference_2026")
    universe = pd.concat([master, inference_2026], ignore_index=True)
    history = read_rating_history(args.rating_history)
    reference = select_prior_ratings(universe, history)
    summary = build_summary(reference, history, args.rating_history)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(args.output, index=False, encoding="utf-8-sig")
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        {
            "output": str(args.output),
            "summary": str(args.summary),
            "rows": len(reference),
            "coverage_by_universe": summary["coverage_by_universe"],
            "boundary_group_counts": summary["prior_rating_boundary_group_counts"],
        }
    )


if __name__ == "__main__":
    main()
