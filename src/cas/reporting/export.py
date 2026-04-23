"""Render a final report from ``AgentState``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cas.agents.state import AgentState
from cas.reporting.audit_trail import to_markdown as audit_to_md


def render_report(state: AgentState | dict[str, Any]) -> dict[str, Any]:
    """Build a structured report from the final agent state."""
    s = dict(state)
    company_id = s.get("company_id", "?")
    company_name = s.get("company_name", company_id)
    market = s.get("market", "?")
    analysis_year = s.get("analysis_year", "?")
    recommendation = s.get("final_recommendation", "review")
    confidence = float(s.get("final_confidence", 0.0) or 0.0)
    overall_score = float(s.get("overall_score", 0.0) or 0.0)
    insufficient = bool(s.get("insufficient_data", False))

    base_assessments = s.get("base_assessments") or {}
    market_overlay = s.get("market_overlay") or {}
    qualitative_overlay = s.get("news_overlay") or {}
    reviews = s.get("committee_reviews") or []
    audit = s.get("audit") or []

    md_lines = [
        f"# Corporate Review: {company_name}",
        "",
        f"- **Company ID**: `{company_id}`",
        f"- **Market**: {market}",
        f"- **Analysis Year**: {analysis_year}",
        f"- **Generated At**: {datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')}",
        "",
    ]

    if insufficient:
        md_lines += [
            "## Analysis Deferred",
            "",
            "The profile is missing required inputs, so a full recommendation was not produced.",
            "",
        ]
    else:
        md_lines += [
            "## Final Recommendation",
            "",
            f"- **Recommendation**: `{recommendation}`",
            f"- **Confidence**: {confidence:.3f}",
            f"- **Overall Score**: {overall_score:.3f}",
            "",
            "## Base Assessments",
            "",
        ]
        for name, assessment in base_assessments.items():
            score = assessment.get("score") if isinstance(assessment, dict) else getattr(assessment, "score", None)
            summary = assessment.get("summary") if isinstance(assessment, dict) else getattr(assessment, "summary", "")
            md_lines.append(f"- `{name}`: {score:.3f} - {summary}" if score is not None else f"- `{name}`: n/a")
        md_lines.append("")

        md_lines += [
            "## Context Adjustments",
            "",
            f"- **Market**: {market_overlay.get('adjustment', 0.0):+.3f} - {market_overlay.get('rationale', '')}",
            f"- **Qualitative**: {qualitative_overlay.get('adjustment', 0.0):+.3f} - {qualitative_overlay.get('rationale', '')}",
            "",
            "## Committee Reviews",
            "",
        ]
        if not reviews:
            md_lines.append("_(No committee reviews)_")
        else:
            md_lines.append("| Perspective | Recommendation | Confidence | Rationale |")
            md_lines.append("|---|---|---|---|")
            for review in reviews:
                review_dict = review if isinstance(review, dict) else review.model_dump()
                md_lines.append(
                    f"| {review_dict.get('perspective','')} | "
                    f"`{review_dict.get('recommendation','')}` | {float(review_dict.get('confidence',0.0)):.3f} | "
                    f"{str(review_dict.get('rationale','')).replace('|', chr(92)+'|')} |"
                )
        md_lines.append("")

    md_lines += [
        "## Audit Trail",
        "",
        audit_to_md(audit),
        "",
        "---",
        "_This report is a decision-support artifact, not an automatic investment decision._",
    ]

    return {
        "company_id": company_id,
        "company_name": company_name,
        "market": market,
        "analysis_year": analysis_year,
        "final_recommendation": recommendation,
        "final_confidence": confidence,
        "overall_score": overall_score,
        "insufficient_data": insufficient,
        "base_assessments": {
            key: (value if isinstance(value, dict) else value.model_dump())
            for key, value in base_assessments.items()
        },
        "market_overlay": market_overlay,
        "news_overlay": qualitative_overlay,
        "committee_reviews": [
            (review if isinstance(review, dict) else review.model_dump()) for review in reviews
        ],
        "audit": [(entry if isinstance(entry, dict) else entry.model_dump()) for entry in audit],
        "markdown": "\n".join(md_lines),
    }
