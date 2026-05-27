from __future__ import annotations

import argparse
import json
from pathlib import Path

from cas.modeling.stage1_improvement import (
    DEFAULT_PROMOTION_GATES,
    REPORT_FILENAME,
    SUMMARY_FILENAME,
    PromotionGates,
    write_stage1_improvement_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIAGNOSTICS_DIR = (
    ROOT / "data" / "outputs" / "modeling" / "feature_46_xgboost" / "diagnostics"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the consolidated Stage 1 XGBoost improvement report from generated "
            "feature-pack, rolling OOT, and rolling-selection diagnostics."
        )
    )
    parser.add_argument("--diagnostics-dir", type=Path, default=DEFAULT_DIAGNOSTICS_DIR)
    parser.add_argument(
        "--min-rolling-f1-delta",
        type=float,
        default=DEFAULT_PROMOTION_GATES.min_rolling_f1_delta,
    )
    parser.add_argument(
        "--min-rolling-pr-auc-delta",
        type=float,
        default=DEFAULT_PROMOTION_GATES.min_rolling_pr_auc_delta,
    )
    parser.add_argument(
        "--min-final-test-f1-delta",
        type=float,
        default=DEFAULT_PROMOTION_GATES.min_final_test_f1_delta,
    )
    parser.add_argument(
        "--max-final-test-fn-delta",
        type=int,
        default=DEFAULT_PROMOTION_GATES.max_final_test_fn_delta,
    )
    parser.add_argument(
        "--max-final-test-fp-delta",
        type=int,
        default=DEFAULT_PROMOTION_GATES.max_final_test_fp_delta,
    )
    return parser.parse_args()


def gates_from_args(args: argparse.Namespace) -> PromotionGates:
    return PromotionGates(
        min_rolling_f1_delta=args.min_rolling_f1_delta,
        min_rolling_pr_auc_delta=args.min_rolling_pr_auc_delta,
        min_final_test_f1_delta=args.min_final_test_f1_delta,
        max_final_test_fn_delta=args.max_final_test_fn_delta,
        max_final_test_fp_delta=args.max_final_test_fp_delta,
    )


def main() -> None:
    args = parse_args()
    summary = write_stage1_improvement_outputs(
        args.diagnostics_dir,
        gates=gates_from_args(args),
    )
    print(
        json.dumps(
            {
                "candidate_feature_set": summary["candidate_feature_set"],
                "report": str(args.diagnostics_dir / REPORT_FILENAME),
                "summary": str(args.diagnostics_dir / SUMMARY_FILENAME),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
