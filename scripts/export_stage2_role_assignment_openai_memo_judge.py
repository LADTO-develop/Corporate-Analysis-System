"""Use OpenAI to judge memo quality for selected Stage 2 role assignments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ROLE_ASSIGNMENT_DIR = (
    ROOT
    / "data/outputs/modeling/feature_46_xgboost/diagnostics/stage2_agents/"
    "feature46_full_review_trigger_73_role_assignment_20"
)
DEFAULT_CANDIDATE_A = "gemini_quant_claude_evidence_openai_chair"
DEFAULT_CANDIDATE_B = "claude_quant_gemini_evidence_openai_chair"
DEFAULT_MODEL = "gpt-4.1-mini"
USD_PER_1M_INPUT = 0.40
USD_PER_1M_OUTPUT = 1.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-dir", type=Path, default=ROLE_ASSIGNMENT_DIR)
    parser.add_argument("--candidate-a", default=DEFAULT_CANDIDATE_A)
    parser.add_argument("--candidate-b", default=DEFAULT_CANDIDATE_B)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI memo judge.")
    assignment_dir = args.assignment_dir if args.assignment_dir.is_absolute() else ROOT / args.assignment_dir
    output_dir = args.output_dir or assignment_dir / "openai_memo_judge_top2"
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_a = _load_candidate(assignment_dir, args.candidate_a)
    candidate_b = _load_candidate(assignment_dir, args.candidate_b)
    joined = _paired_cases(candidate_a, candidate_b)
    client = OpenAI()

    case_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(_chunks(joined, args.chunk_size), start=1):
        payload = _judge_payload(args.candidate_a, args.candidate_b, chunk)
        result, usage = _judge_chunk(
            client=client,
            model=args.model,
            candidate_a=args.candidate_a,
            candidate_b=args.candidate_b,
            payload=payload,
        )
        for case in result.get("cases", []):
            case_rows.append({"chunk_index": chunk_index, **case})
        usage_rows.append({"chunk_index": chunk_index, **usage})

    case_scores = pd.DataFrame(case_rows)
    usage = pd.DataFrame(usage_rows)
    summary = _candidate_summary(case_scores, usage, args)
    case_scores_path = output_dir / "openai_memo_judge_case_scores.csv"
    summary_path = output_dir / "openai_memo_judge_candidate_summary.csv"
    usage_path = output_dir / "openai_memo_judge_usage.csv"
    report_path = output_dir / "openai_memo_judge_report.md"
    raw_prompt_path = output_dir / "openai_memo_judge_payload_preview.json"
    case_scores.to_csv(case_scores_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    usage.to_csv(usage_path, index=False, encoding="utf-8-sig")
    raw_prompt_path.write_text(
        json.dumps(_judge_payload(args.candidate_a, args.candidate_b, joined.head(2)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(
        _build_report(
            summary=summary,
            case_scores=case_scores,
            usage=usage,
            args=args,
            generated_at=datetime.now(UTC).isoformat(),
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case_scores": _relative(case_scores_path),
                "candidate_summary": _relative(summary_path),
                "usage": _relative(usage_path),
                "report": _relative(report_path),
                "payload_preview": _relative(raw_prompt_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _load_candidate(assignment_dir: Path, assignment_id: str) -> pd.DataFrame:
    path = assignment_dir / "runs" / f"multi_role_{assignment_id}" / "committee_review_batch_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Candidate result not found: {path}")
    frame = pd.read_csv(path)
    frame["case_key"] = (
        frame["market"].astype(str)
        + "|"
        + frame["stock_code"].astype(str)
        + "|"
        + frame["fiscal_year"].astype(str)
        + "|"
        + frame["sample_category"].astype(str)
    )
    return frame


def _paired_cases(candidate_a: pd.DataFrame, candidate_b: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_key",
        "corp_name",
        "fiscal_year",
        "sample_category",
        "actual_label_name",
        "model_predicted_label_name",
        "final_committee_label",
        "committee_decision_type",
        "risk_hold_reason_summary",
        "agent_disagreement_summary",
        "conflict_resolution",
        "final_review_memo",
        "decision_trace",
        "top_evidence_titles",
        "materiality_top_basis",
    ]
    left = candidate_a.loc[:, columns].add_prefix("a_")
    right = candidate_b.loc[:, columns].add_prefix("b_")
    paired = left.merge(right, left_on="a_case_key", right_on="b_case_key", how="inner")
    paired = paired.sort_values(["a_sample_category", "a_corp_name", "a_fiscal_year"]).reset_index(drop=True)
    paired.insert(0, "case_id", [f"case_{index + 1:02d}" for index in range(len(paired))])
    return paired


def _judge_payload(candidate_a: str, candidate_b: str, chunk: pd.DataFrame) -> dict[str, Any]:
    cases = []
    for row in chunk.to_dict(orient="records"):
        cases.append(
            {
                "case_id": row["case_id"],
                "company": row["a_corp_name"],
                "fiscal_year": row["a_fiscal_year"],
                "sample_category": row["a_sample_category"],
                "actual_label": row["a_actual_label_name"],
                "stage1_label": row["a_model_predicted_label_name"],
                "candidate_a": _candidate_memo(row, "a"),
                "candidate_b": _candidate_memo(row, "b"),
            }
        )
    return {
        "candidate_a_id": candidate_a,
        "candidate_b_id": candidate_b,
        "rubric": {
            "scale": "1 to 5, where 5 is best",
            "criteria": [
                "financial_specificity",
                "evidence_grounding",
                "decision_consistency",
                "actionability",
                "clarity",
            ],
            "winner_rule": "Pick the candidate with the better committee memo for this case. Use tie only when genuinely indistinguishable.",
        },
        "cases": cases,
    }


def _candidate_memo(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        "final_label": row[f"{prefix}_final_committee_label"],
        "decision_type": row[f"{prefix}_committee_decision_type"],
        "risk_hold_reason_summary": _trim(row.get(f"{prefix}_risk_hold_reason_summary"), 650),
        "agent_disagreement_summary": _trim(row.get(f"{prefix}_agent_disagreement_summary"), 450),
        "conflict_resolution": _trim(row.get(f"{prefix}_conflict_resolution"), 450),
        "final_review_memo": _trim(row.get(f"{prefix}_final_review_memo"), 1000),
        "decision_trace": _trim(row.get(f"{prefix}_decision_trace"), 650),
        "top_evidence_titles": _trim(row.get(f"{prefix}_top_evidence_titles"), 450),
        "materiality_top_basis": _trim(row.get(f"{prefix}_materiality_top_basis"), 350),
    }


def _judge_chunk(
    *,
    client: OpenAI,
    model: str,
    candidate_a: str,
    candidate_b: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    system = (
        "You are a senior credit committee memo reviewer. Judge only memo quality, "
        "not whether the model is commercially preferable. Respond as strict JSON."
    )
    user = (
        "Compare candidate_a and candidate_b for each case. Score each candidate from 1 to 5 "
        "on financial_specificity, evidence_grounding, decision_consistency, actionability, "
        "and clarity. Also provide overall_score from 1 to 5, winner as candidate_a/candidate_b/tie, "
        "and a short Korean rationale. JSON schema: "
        "{\"cases\":[{\"case_id\":\"case_01\",\"candidate_a\":{\"financial_specificity\":1,"
        "\"evidence_grounding\":1,\"decision_consistency\":1,\"actionability\":1,\"clarity\":1,"
        "\"overall_score\":1},\"candidate_b\":{...},\"winner\":\"candidate_a\","
        "\"rationale_ko\":\"...\"}]}.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    return parsed, {
        "model": model,
        "candidate_a_id": candidate_a,
        "candidate_b_id": candidate_b,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": round(
            prompt_tokens * USD_PER_1M_INPUT / 1_000_000
            + completion_tokens * USD_PER_1M_OUTPUT / 1_000_000,
            6,
        ),
    }


def _candidate_summary(
    case_scores: pd.DataFrame,
    usage: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for label, assignment_id in (("candidate_a", args.candidate_a), ("candidate_b", args.candidate_b)):
        prefix = label + "."
        extracted = pd.json_normalize(case_scores[label])
        winner_count = int(case_scores["winner"].astype(str).eq(label).sum())
        tie_count = int(case_scores["winner"].astype(str).eq("tie").sum())
        rows.append(
            {
                "assignment_id": assignment_id,
                "judge_model": args.model,
                "cases": len(case_scores),
                "mean_overall_score": round(float(extracted["overall_score"].mean()), 4),
                "mean_financial_specificity": round(float(extracted["financial_specificity"].mean()), 4),
                "mean_evidence_grounding": round(float(extracted["evidence_grounding"].mean()), 4),
                "mean_decision_consistency": round(float(extracted["decision_consistency"].mean()), 4),
                "mean_actionability": round(float(extracted["actionability"].mean()), 4),
                "mean_clarity": round(float(extracted["clarity"].mean()), 4),
                "wins": winner_count,
                "ties": tie_count,
                "total_judge_prompt_tokens": int(usage["prompt_tokens"].sum()),
                "total_judge_completion_tokens": int(usage["completion_tokens"].sum()),
                "total_judge_estimated_cost_usd": round(float(usage["estimated_cost_usd"].sum()), 6),
                "score_column_prefix": prefix,
            }
        )
    return pd.DataFrame(rows)


def _build_report(
    *,
    summary: pd.DataFrame,
    case_scores: pd.DataFrame,
    usage: pd.DataFrame,
    args: argparse.Namespace,
    generated_at: str,
) -> str:
    lines = [
        "# OpenAI Memo Judge: Stage 2 Role Assignment Top 2",
        "",
        f"- generated_at_utc: `{generated_at}`",
        f"- judge_model: `{args.model}`",
        f"- candidate_a: `{args.candidate_a}`",
        f"- candidate_b: `{args.candidate_b}`",
        f"- cases: `{len(case_scores)}`",
        f"- estimated_judge_cost_usd: `{usage['estimated_cost_usd'].sum():.6f}`",
        "",
        "## Candidate Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Case Winners",
        "",
        case_scores[["case_id", "winner", "rationale_ko"]].to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


def _chunks(frame: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    return [frame.iloc[start : start + size].copy() for start in range(0, len(frame), size)]


def _trim(value: object, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + " ...[truncated]"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
