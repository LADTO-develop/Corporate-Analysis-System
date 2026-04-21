# Architecture

이 문서는 Borderline Firm Detector(이하 BFD)의 전체 파이프라인 구조와 각 레이어의
책임 범위를 정리합니다. 원본 프로젝트 기획은 `TEMPMODELPLAN.pdf`에 있으며, 이 문서는
그 구현 청사진입니다.

## 전체 데이터 흐름

```
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ TS2000 재무제표 5본 │   │ ECOS 거시경제지표 │   │ DART/언론 뉴스·공시│
│ (BS, IS, CF, SCE,  │   │ (금리, FX, CPI,  │   │                     │
│  주석)             │   │  회사채 금리)    │   │                     │
└─────────┬──────────┘   └─────────┬──────────┘   └─────────┬──────────┘
          │                         │                         │
          ▼                         ▼                         ▼
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ features/*.py       │   │ features/macro.py  │   │ rag/llm_features   │
│ (재무 파생변수)     │   │ (거시 스냅샷)      │   │ (Claude 구조화출력)│
└─────────┬──────────┘   └─────────┬──────────┘   └─────────┬──────────┘
          └────────────┬────────────┘                         │
                       ▼                                      ▼
          ┌────────────────────────────┐        ┌────────────────────┐
          │ data/splitters.py          │        │  agents/nodes/     │
          │ (t → t+1 매핑, 누수 차단)  │        │  news_overlay_node │
          └─────────────┬──────────────┘        └────────────────────┘
                        ▼
          ┌────────────────────────────┐
          │ validation/leakage.py      │  ← CI에서 실패 시 머지 차단
          │ (assert_next_year_mapping) │
          └─────────────┬──────────────┘
                        ▼
          ┌────────────────────────────┐
          │ 시장별 분리 모델           │
          │  KOSPI ensemble            │
          │  KOSDAQ ensemble           │
          │   (TabPFN + LGBM + EBM)    │
          └─────────────┬──────────────┘
                        ▼
          ┌────────────────────────────┐
          │ agents/nodes/committee     │
          │  (악마의 대변인, 6모자,    │
          │   명목집단, 스텝래더)      │
          └─────────────┬──────────────┘
                        ▼
          ┌────────────────────────────┐
          │ reporting/export.py        │
          │  (JSON + Markdown 보고서)  │
          └────────────────────────────┘
```

## 레이어 책임

| 레이어 | 경로 | 핵심 책임 |
|--------|------|-----------|
| Data | `bfd/data/` | TS2000/ECOS/뉴스/평가사 파일 로드, 스키마 검증, t→t+1 정합 |
| Features | `bfd/features/` | 재무제표 5본·거시·주석에서 파생변수 생성 + 카탈로그 |
| Ratings | `bfd/ratings/` | 평가사별 원본 등급 → 공통 22-notch 정규화 + 이진화 |
| Models | `bfd/models/` | TabPFN/LGBM/EBM + OOF 스태킹 + 캘리브레이션 |
| RAG | `bfd/rag/` | 뉴스 RAG + LLM 구조화 출력 + 내부정보 보안 정책 |
| Agents | `bfd/agents/` | LangGraph State·노드·위원회 전략·툴 카탈로그 |
| Validation | `bfd/validation/` | 누수 단언·메트릭·walk-forward·드리프트 |
| Reporting | `bfd/reporting/` | 감사 트레일·EBM 설명·최종 리포트 렌더 |

## 핵심 설계 결정

### 시장별 분리 모델
KOSPI와 KOSDAQ은 재무 구조, 지배적 평가사, 데이터 결측률이 크게 다릅니다.
동일 모델을 강제할 경우 KOSDAQ 소형주 특성이 KOSPI 대형주 통계에 묻혀버리므로,
`configs/market/{kospi,kosdaq}.yaml`에서 피처 서브셋·하이퍼파라미터를 분리합니다.

### 타깃 누수 방지를 코드로 강제
재무(t년) ↔ 등급(t+1년) 매핑은 `bfd.utils.time.target_rating_year`가 유일한 SoT이고,
`splitters.map_financials_to_next_year_rating`이 이를 사용하여 조인하며,
`validation.leakage.assert_next_year_mapping`이 CI에서 어서션합니다.
세 지점이 모두 같은 한 함수를 참조하므로 부분적 변경이 불가능합니다.

### EBM을 "해석 전용 채널"로 항상 유지
앙상블 기여와 별개로 `bfd.reporting.explanations`는 항상 EBM의 global importance와
local shape function을 꺼내 리포트에 싣습니다. 블랙박스 앙상블의 판정 근거가 필요할 때
별도의 post-hoc SHAP 작업 없이 바로 꺼낼 수 있습니다.

### 위원회 LLM은 앙상블에 참여하지 않은 별도 모델
`configs/agent/committee.yaml`의 `committee_llm`은 분석 파이프라인에 사용되지 않은
LLM을 지정해야 합니다. 같은 모델이 예측과 검증을 모두 수행하면 편향이 순환하기 때문입니다.

### 내부정보 채널은 기본 비활성
`BFD_INTERNAL_INFO_MODE`가 `disabled`이면 내부 기밀 입력 경로 자체가 작동하지 않습니다.
활성화할 경우 `bfd.rag.internal_info.redaction`으로 PII 마스킹이, `policy`로 감사로그가
강제됩니다. 자세한 내용은 [security_policy.md](security_policy.md) 참고.

## LangGraph State 요약

```python
class AgentState(TypedDict, total=False):
    corp_code: str
    market: Literal["KOSPI", "KOSDAQ"]
    fiscal_year: int
    target_year: int

    raw_financials: dict[str, Any]
    features: dict[str, float | int | bool]

    base_predictions: Annotated[dict[str, BasePrediction], merge_dict]
    macro_overlay: MacroOverlay
    news_overlay: NewsOverlay
    ensemble_proba: float

    committee_opinions: Annotated[list[CommitteeOpinion], append_opinions]
    final_verdict: Literal["borderline", "healthy", "uncertain"]
    final_confidence: float

    audit: Annotated[list[AuditEntry], append_audit]
    artifacts: Annotated[dict[str, str], merge_dict]
    insufficient_data: bool
```

- `append_audit`, `append_opinions`: 병렬 실행된 노드들의 출력이 덮어써지지 않고
  누적되도록 하는 reducer입니다.
- `merge_dict`: `base_predictions`처럼 여러 노드/모델이 동일 dict의 서로 다른 키를
  채우는 경우에 사용됩니다.

## 노드 토폴로지

```
START
  └─ data
       ├─ [insufficient] → report (조기 종료)
       └─ [enough]
            └─ feature
                 └─ base_prediction
                      ├─ macro_overlay ─┐
                      └─ news_overlay ──┴─ committee ─ report ─ END
```

`macro_overlay`와 `news_overlay`는 `base_prediction` 이후 병렬로 실행되고
(fan-out), `committee`에서 합류합니다 (fan-in).
`data_node`가 데이터 부족 판정을 내리면 `feature` 이후 단계를 모두 건너뛰고
`report_node`로 직행합니다.

## 관련 문서

- [data_dictionary_ts2000.md](data_dictionary_ts2000.md) — TS2000 5본 컬럼 사전
- [rating_normalization.md](rating_normalization.md) — 평가사 정규화 규칙
- [feature_spec.md](feature_spec.md) — 파생변수 목록과 해석
- [agent_graph.md](agent_graph.md) — Mermaid 다이어그램
- [security_policy.md](security_policy.md) — 내부정보 보안 정책
