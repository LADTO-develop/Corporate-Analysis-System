"""Compare deterministic Stage 2 explanations against OpenAI Agno outputs.

The script is intentionally offline: it does not call OpenAI, Agno, or external
evidence APIs.  Run ``run_committee_review_evaluation_batch.py`` first for both
deterministic and Agno modes, then use this script to compare labels, success
flags, and explanation quality heuristics.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_DIR = ROOT / "data/outputs/modeling/feature_46_xgboost/diagnostics/stage2_agents"
DEFAULT_DETERMINISTIC_RESULTS = (
    DIAGNOSTICS_DIR
    / "committee_review_openai_agno_comparison_deterministic/committee_review_batch_results.csv"
)
DEFAULT_AGNO_RESULTS = (
    DIAGNOSTICS_DIR
    / "committee_review_openai_agno_comparison_agno/committee_review_batch_results.csv"
)
DEFAULT_OUTPUT_PREFIX = DIAGNOSTICS_DIR / "stage2_openai_agno_explanation_comparison"

KEY_COLUMNS = ["market", "stock_code", "fiscal_year", "eval_year"]
QUALITY_TERMS = (
    "확률",
    "기준선",
    "재무",
    "부채",
    "유동성",
    "현금흐름",
    "외부근거",
    "뉴스",
    "공시",
    "경계",
    "완화",
    "위험",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic-results", type=Path, default=DEFAULT_DETERMINISTIC_RESULTS)
    parser.add_argument("--agno-results", type=Path, default=DEFAULT_AGNO_RESULTS)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deterministic = read_results(args.deterministic_results)
    agno = read_results(args.agno_results)
    detail = compare_results(deterministic, agno)
    outputs = write_outputs(
        output_prefix=args.output_prefix,
        deterministic_path=args.deterministic_results,
        agno_path=args.agno_results,
        deterministic=deterministic,
        agno=agno,
        detail=detail,
    )
    print(json.dumps({key: _relative(value) for key, value in outputs.items()}, ensure_ascii=False))


def read_results(path: Path) -> pd.DataFrame:
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    if "stock_code" in frame.columns:
        frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    return frame


def compare_results(deterministic: pd.DataFrame, agno: pd.DataFrame) -> pd.DataFrame:
    if deterministic.empty or agno.empty:
        return _empty_detail()
    merged = deterministic.merge(
        agno,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_deterministic", "_agno"),
        validate="one_to_one",
    )
    if merged.empty:
        return _empty_detail()
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        det_memo = str(row.get("final_review_memo_deterministic") or "")
        agno_memo = str(row.get("final_review_memo_agno") or "")
        det_quality = explanation_quality(det_memo)
        agno_quality = explanation_quality(agno_memo)
        rows.append(
            {
                "market": row["market"],
                "stock_code": row["stock_code"],
                "corp_name": row.get("corp_name_deterministic") or row.get("corp_name_agno"),
                "fiscal_year": row["fiscal_year"],
                "eval_year": row["eval_year"],
                "model_error_type": row.get("model_error_type_deterministic")
                or row.get("model_error_type_agno"),
                "actual_label_name": row.get("actual_label_name_deterministic")
                or row.get("actual_label_name_agno"),
                "stage1_label": row.get("model_predicted_label_name_deterministic")
                or row.get("model_predicted_label_name_agno"),
                "deterministic_label": row.get("final_committee_label_deterministic"),
                "agno_label": row.get("final_committee_label_agno"),
                "label_changed": row.get("final_committee_label_deterministic")
                != row.get("final_committee_label_agno"),
                "deterministic_success": _bool_value(row.get("committee_success_deterministic")),
                "agno_success": _bool_value(row.get("committee_success_agno")),
                "success_changed": _bool_value(row.get("committee_success_deterministic"))
                != _bool_value(row.get("committee_success_agno")),
                "deterministic_memo_chars": len(det_memo),
                "agno_memo_chars": len(agno_memo),
                "deterministic_quality_score": det_quality,
                "agno_quality_score": agno_quality,
                "quality_delta": round(agno_quality - det_quality, 4),
            }
        )
    return pd.DataFrame(rows)


def explanation_quality(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    char_score = min(len(stripped) / 360.0, 1.0)
    term_score = min(sum(1 for term in QUALITY_TERMS if term in stripped) / 6.0, 1.0)
    numeric_score = min(len(re.findall(r"\d+(?:\.\d+)?%?", stripped)) / 4.0, 1.0)
    return round(0.35 * char_score + 0.45 * term_score + 0.20 * numeric_score, 4)


def write_outputs(
    *,
    output_prefix: Path,
    deterministic_path: Path,
    agno_path: Path,
    deterministic: pd.DataFrame,
    agno: pd.DataFrame,
    detail: pd.DataFrame,
) -> dict[str, Path]:
    if not output_prefix.is_absolute():
        output_prefix = ROOT / output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = output_prefix.with_name(output_prefix.name + "_details.csv")
    summary_path = output_prefix.with_suffix(".json")
    report_path = output_prefix.with_suffix(".md")

    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary = build_summary(
        deterministic_path=deterministic_path,
        agno_path=agno_path,
        deterministic=deterministic,
        agno=agno,
        detail=detail,
        detail_path=detail_path,
        report_path=report_path,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_report(summary=summary, detail=detail), encoding="utf-8")
    return {"details": detail_path, "summary": summary_path, "report": report_path}


def build_summary(
    *,
    deterministic_path: Path,
    agno_path: Path,
    deterministic: pd.DataFrame,
    agno: pd.DataFrame,
    detail: pd.DataFrame,
    detail_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "deterministic_results_path": _relative(deterministic_path),
        "agno_results_path": _relative(agno_path),
        "deterministic_rows": len(deterministic),
        "agno_rows": len(agno),
        "matched_rows": len(detail),
        "agno_results_available": not agno.empty,
        "outputs": {
            "details": _relative(detail_path),
            "report": _relative(report_path),
        },
    }
    if detail.empty:
        output["status"] = (
            "agno_results_missing_or_unmatched" if agno.empty else "no_common_company_year_rows"
        )
        return output
    output["status"] = "compared"
    output["label_change_count"] = int(detail["label_changed"].sum())
    output["success_change_count"] = int(detail["success_changed"].sum())
    output["deterministic_success_rate"] = round(float(detail["deterministic_success"].mean()), 4)
    output["agno_success_rate"] = round(float(detail["agno_success"].mean()), 4)
    output["deterministic_quality_mean"] = round(
        float(detail["deterministic_quality_score"].mean()),
        4,
    )
    output["agno_quality_mean"] = round(float(detail["agno_quality_score"].mean()), 4)
    output["quality_delta_mean"] = round(float(detail["quality_delta"].mean()), 4)
    return output


def build_report(*, summary: dict[str, Any], detail: pd.DataFrame) -> str:
    lines = [
        "# OpenAI Agno Explanation Comparison",
        "",
        f"- 생성시각(UTC): `{summary['generated_at_utc']}`",
        f"- deterministic rows: `{summary['deterministic_rows']}`",
        f"- Agno rows: `{summary['agno_rows']}`",
        f"- matched rows: `{summary['matched_rows']}`",
        "",
    ]
    if detail.empty:
        lines.extend(
            [
                "## 상태",
                "",
                "- 현재 비교 가능한 OpenAI Agno 결과 파일이 없거나 deterministic 결과와 같은 기업-연도 키가 없습니다.",
                "- 이 스크립트는 외부 API를 직접 호출하지 않습니다. Agno 배치를 별도로 실행한 뒤 같은 경로에 결과를 저장하면 자동으로 비교표를 생성합니다.",
                "- 비교 기준은 최종 라벨 변화, 성공 여부 변화, 메모 길이, 핵심 용어/수치 포함도 기반 설명 품질 점수입니다.",
                "",
            ]
        )
        return "\n".join(lines)
    lines.extend(
        [
            "## 요약",
            "",
            f"- deterministic success rate: `{summary['deterministic_success_rate']:.4f}`",
            f"- Agno success rate: `{summary['agno_success_rate']:.4f}`",
            f"- 평균 설명 품질 점수 변화: `{summary['quality_delta_mean']:.4f}`",
            f"- 최종 라벨 변경 건수: `{summary['label_change_count']}`",
            f"- 성공 여부 변경 건수: `{summary['success_change_count']}`",
            "",
            "## Case Preview",
            "",
            _markdown_table(_preview_columns(detail)),
            "",
        ]
    )
    return "\n".join(lines)


def _preview_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "corp_name",
        "model_error_type",
        "stage1_label",
        "deterministic_label",
        "agno_label",
        "deterministic_quality_score",
        "agno_quality_score",
        "quality_delta",
    ]
    return frame.loc[:, [column for column in columns if column in frame.columns]]


def _markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    preview = frame.head(max_rows).copy()
    columns = [str(column) for column in preview.columns]
    rows = preview.astype(object).where(pd.notna(preview), "").astype(str).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(value.replace("|", "/") for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _empty_detail() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "market",
            "stock_code",
            "corp_name",
            "fiscal_year",
            "eval_year",
            "model_error_type",
            "actual_label_name",
            "stage1_label",
            "deterministic_label",
            "agno_label",
            "label_changed",
            "deterministic_success",
            "agno_success",
            "success_changed",
            "deterministic_memo_chars",
            "agno_memo_chars",
            "deterministic_quality_score",
            "agno_quality_score",
            "quality_delta",
        ]
    )


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
