"""Render a single firm's final report from its ``AgentState``.

Produces a dict with ``markdown``, ``json``, and metadata fields that the
``report_node`` then writes to disk.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from bfd.agents.state import AgentState
from bfd.reporting.audit_trail import to_markdown as audit_to_md


def render_report(state: AgentState | dict[str, Any]) -> dict[str, Any]:
    """Build a structured report from the final agent state.

    The markdown block is designed to be copy-pasted into a PR comment or a
    lightweight credit memo.
    """
    s = dict(state)
    corp_code = s.get("corp_code", "?")
    market = s.get("market", "?")
    fy = s.get("fiscal_year", "?")
    verdict = s.get("final_verdict", "uncertain")
    confidence = float(s.get("final_confidence", 0.0) or 0.0)
    ensemble_proba = float(s.get("ensemble_proba", 0.0) or 0.0)
    insufficient = bool(s.get("insufficient_data", False))

    base_preds = s.get("base_predictions") or {}
    macro = s.get("macro_overlay") or {}
    news = s.get("news_overlay") or {}
    opinions = s.get("committee_opinions") or []
    audit = s.get("audit") or []

    md_lines = [
        f"# 한계기업 간이 진단 보고서",
        "",
        f"- **기업코드**: `{corp_code}`",
        f"- **시장**: {market}",
        f"- **회계연도**: FY{fy}",
        f"- **대상 신용등급 연도**: {fy + 1 if isinstance(fy, int) else '?'}",
        f"- **생성 시각**: {datetime.utcnow().isoformat(timespec='seconds')}Z",
        "",
    ]

    if insufficient:
        md_lines += [
            "## ⚠️ 판정 보류",
            "",
            "필수 데이터가 부족하여 정식 판정을 내리지 못했습니다. 감사 트레일을 확인해 주세요.",
            "",
        ]
    else:
        md_lines += [
            "## 최종 판정",
            "",
            f"- **판정**: `{verdict}`",
            f"- **신뢰도**: {confidence:.3f}",
            f"- **앙상블 P(borderline)**: {ensemble_proba:.3f}",
            "",
            "## 베이스 모델 예측",
            "",
        ]
        for name, pred in base_preds.items():
            p = pred.get("proba_borderline") if isinstance(pred, dict) else getattr(pred, "proba_borderline", None)
            md_lines.append(f"- `{name}`: P(borderline) = {p:.3f}" if p is not None else f"- `{name}`: n/a")
        md_lines.append("")

        md_lines += [
            "## 오버레이",
            "",
            f"- **거시 조정**: {macro.get('adjustment', 0.0):+.3f} — {macro.get('rationale_ko', '')}",
            f"- **뉴스 조정**: {news.get('adjustment', 0.0):+.3f} — {news.get('rationale_ko', '')}",
            "",
            "## 위원회 의견",
            "",
        ]
        if not opinions:
            md_lines.append("_(위원회 의견 없음)_")
        else:
            md_lines.append("| 기법 | 역할 | 판정 | 신뢰도 | 논거 |")
            md_lines.append("|---|---|---|---|---|")
            for op in opinions:
                op_d = op if isinstance(op, dict) else op.model_dump()
                md_lines.append(
                    f"| {op_d.get('technique','')} | {op_d.get('role','')} | "
                    f"`{op_d.get('verdict','')}` | {float(op_d.get('confidence',0.0)):.3f} | "
                    f"{str(op_d.get('rationale_ko','')).replace('|', chr(92)+'|')} |"
                )
        md_lines.append("")

    md_lines += [
        "## 감사 트레일",
        "",
        audit_to_md(audit),
        "",
        "---",
        "_본 보고서는 보조 의사결정 도구입니다. 투자/대출 의사결정에 본 보고서를 단독 근거로 사용해서는 안 됩니다._",
    ]

    return {
        "corp_code": corp_code,
        "market": market,
        "fiscal_year": fy,
        "final_verdict": verdict,
        "final_confidence": confidence,
        "ensemble_proba": ensemble_proba,
        "insufficient_data": insufficient,
        "base_predictions": {
            k: (v if isinstance(v, dict) else v.model_dump())
            for k, v in (base_preds or {}).items()
        },
        "macro_overlay": macro,
        "news_overlay": news,
        "committee_opinions": [
            (o if isinstance(o, dict) else o.model_dump()) for o in opinions
        ],
        "audit": [
            (a if isinstance(a, dict) else a.model_dump()) for a in audit
        ],
        "markdown": "\n".join(md_lines),
    }
