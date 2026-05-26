# Live Agno/OpenAI External API Runbook

작성일: 2026-05-21

## 문제 원인

현재 Stage 2 Agno 기본 모드는 OpenAI 단일 provider 실행(`CAS_STAGE2_AGNO_MODE=single`, `CAS_STAGE2_MODEL_PROVIDER=openai`)이다. 이 모드는 `OPENAI_API_KEY`만 있으면 preflight가 통과하도록 맞춰져 있다. `single`은 OpenAI 한 provider를 쓰되 QuantCredit/EvidenceAudit/ChairReport 세 역할 agent를 분리 실행하는 모드다. Agno live 실행에서는 특정 조건에만 ReviewQAAgent와 RiskRecallQAAgent가 사후 검수로 추가될 수 있다. 실제 live latency를 측정할 때는 캐시 재사용을 피하기 위해 `--no-stage2-llm-cache`를 붙인다.

여러 LLM 관점을 비교하는 `multi_llm_committee`는 선택 사항이다. 이 모드는 역할별로 Claude, GPT, Gemini를 함께 쓰므로 `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` 또는 `GEMINI_API_KEY`가 모두 필요할 수 있다.

## Codex 실행 환경 제한

Codex 작업환경에서는 private workspace-derived holdout/evaluation data를 외부 API로 전송하는 실행이 정책상 차단될 수 있다. 사용자가 대화에서 외부 전송을 승인해도, 상위 실행 정책이 차단하면 Codex가 우회할 수 없다.

따라서 실제 기업-회계연도 데이터를 OpenAI API로 보내는 live batch는 로컬 터미널에서 직접 실행한다. 아래 명령은 같은 저장소와 같은 Python 환경을 사용하므로 결과 파일 경로는 Codex에서 만든 산출물과 동일하게 남는다.

## Preflight

```bash
cd "/Users/inji/Documents/금융 데이터 분석/Project/Corporate-Analysis-System"

/usr/bin/env \
  CAS_STAGE2_RUNNER=agno \
  CAS_STAGE2_AGNO_MODE=single \
  CAS_STAGE2_MODEL_PROVIDER=openai \
  CAS_STAGE2_MODEL=gpt-4.1-mini \
  /opt/anaconda3/envs/aura/bin/python scripts/check_agno_stage2.py
```

예상 결과:

```text
Agno Stage 2 preflight passed.
```

## OpenAI API Live Batch

외부 뉴스/공시 API 없이 OpenAI/Agno 위원회만 확인할 때:

```bash
/usr/bin/env \
  CAS_STAGE2_FALLBACK_ON_ERROR=0 \
  CAS_STAGE2_MAX_TOKENS=6000 \
  CAS_STAGE2_AGENT_TIMEOUT_SECONDS=30 \
  CAS_STAGE2_AGENT_RETRIES=2 \
  CAS_STAGE2_PROVIDER_MAX_RETRIES=0 \
  /opt/anaconda3/envs/aura/bin/python scripts/run_committee_review_evaluation_batch.py \
  --samples data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_holdout_unseen_8_samples.csv \
  --output-dir data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_holdout_unseen_agno_openai_live_batch \
  --policy balanced_current_45_or_near_threshold_0_10 \
  --per-category 2 \
  --max-cases 8 \
  --stage2-runner agno \
  --stage2-agno-mode single \
  --stage2-model-provider openai \
  --stage2-model gpt-4.1-mini \
  --no-stage2-llm-cache \
  --workers 2 \
  --retry-failed-attempts 2 \
  --retry-failed-workers 1 \
  --retry-failed-delay-seconds 2
```

외부 뉴스/공시 수집까지 함께 켤 때는 `OPENDART_API_KEY`, `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`, `TAVILY_API_KEY`를 `.env`에 설정한 뒤 `--live-external-evidence`를 추가한다.

OpenAI/Claude Agno 호출 지연 outlier를 줄일 때는 `CAS_STAGE2_AGENT_TIMEOUT_SECONDS`를 설정한다. 이 값은 개별 agent HTTP 요청 timeout이며, timeout 또는 일시 오류가 나면 `CAS_STAGE2_AGENT_RETRIES` 횟수만큼 CAS retry 루프가 다시 시도한다. provider SDK 내부 재시도와 CAS 재시도가 겹치면 지연시간이 길어질 수 있으므로, 속도 측정에서는 `CAS_STAGE2_PROVIDER_MAX_RETRIES=0`을 권장한다. timeout을 끄려면 `CAS_STAGE2_AGENT_TIMEOUT_SECONDS=0` 또는 `off`를 사용한다.

## API 실패행 자동 재시도

`run_committee_review_evaluation_batch.py`는 배치 1차 실행 후 실패행만 자동 재시도할 수 있다.
재시도 대상은 다음 중 하나에 해당하는 행이다.

- `error_message`가 있는 행
- `stage2_error_message`가 있는 행
- 최종 위원회 라벨이 비어 있는 행
- `committee_effect` 또는 `committee_review_safe_effect`가 `run_failed`인 행
- 외부근거 수집 전체 상태가 `error` 또는 `failed`인 행

권장 설정은 본 실행은 병렬로 돌리고, 재시도는 API TPM burst를 피하기 위해 `workers=1`로
낮추는 방식이다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/run_committee_review_evaluation_batch.py \
  --samples data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_mixed_hard_40_timeout30_speed_gate_v3_samples.csv \
  --output-dir data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_mixed_hard_40_agno_openai_live_with_retry \
  --policy mixed_hard_40_timeout30_speed_gate_v3 \
  --per-category 12 \
  --max-cases 40 \
  --live-external-evidence \
  --stage2-runner agno \
  --stage2-agno-mode single \
  --stage2-model-provider openai \
  --stage2-model gpt-4.1-mini \
  --no-stage2-llm-cache \
  --workers 4 \
  --retry-failed-attempts 2 \
  --retry-failed-workers 1 \
  --retry-failed-delay-seconds 2
```

최종 `committee_review_batch_results.csv`는 재시도 성공행이 병합된 combined 결과다.
재시도된 행에는 `retry_attempt` 컬럼이 채워진다. 감사 추적용 원본 재시도 샘플과
재시도 결과는 `output_dir/retry_artifacts/retry_attempt_N_samples.csv`,
`output_dir/retry_artifacts/retry_attempt_N_results.csv`에 남는다. 불필요하면
`--no-retry-failed-artifacts`를 붙인다.

OpenDART 공시 분류는 `external_evidence_v7` 캐시 버전을 사용한다. 이 버전부터
소송/계약해지/자금조달/거래정지 공시를 모두 같은 위험으로 보지 않고,
`disclosure_severity`, `disclosure_event_class`, `disclosure_materiality`를 함께 남긴다.
일정금액 미만 소송, 자율공시 단일 계약해지, SPAC 합병 절차성 거래정지는
`caution/procedural_or_one_off`로 낮춰 EvidenceAuditAgent가 참고 맥락으로만 다루고,
상장폐지·감사의견 거절·관리종목·영업정지 같은 실질 부실 사건은 `adverse/veto`로 유지한다.

또한 v5부터는 `단일판매ㆍ공급계약해지`와 `영업정지` 후보에 한해 OpenDART 상세 공시를
추가 조회한다. 영업정지는 `bsnSp.json`의 `sl_vs` 또는 `bsnsp_amt/rsl`을 사용하고,
계약해지는 `document.xml` 원문 zip에서 매출 대비 비율을 파싱해 `materiality_ratio`,
`materiality_basis`, `materiality_source`를 남긴다. 매출 대비 3% 미만은
`procedural_or_one_off`, 3~10%는 `watch_context`, 10% 이상은 `substantive_adverse`로
분류한다. 이 상세중요도 보강은 기본 활성화이며, 속도 비교에서 끄고 싶으면
`CAS_OPENDART_DETAIL_MATERIALITY_ENABLED=0`을 추가한다.

v6에서는 `bsnSp.json`에 비율이 없거나 매칭되는 영업정지 행이 없을 때
`document.xml` 원문 fallback을 한 번 더 수행한다. 이 fallback은 종속회사 영업정지
공시에서 `최근매출액 대비`, `영업정지금액`, `최근매출액` 같은 원문 표 값을 찾아
모회사 신용위험으로 볼 만큼 중대한지 다시 분류한다.

v7에서는 상세 공시 materiality 범위를 확장했다. 자금조달 공시는 발행금액/자기자본과
희석률, 채무보증은 보증금액/자기자본, 소송은 청구금액/자기자본 또는 매출액을 파싱한다.
3% 미만은 낮은 중요도, 3~10%는 watch context, 10% 이상은 `substantive_adverse`로
분류해 제목만으로 위험을 확정하지 않도록 한다.

이후 committee guardrail은 자금조달·채무보증의 10% 이상 materiality를 단독 치명
근거로 쓰지 않는다. `veto_candidate`, `critical_context_confirmed`, 자본잠식·부도·
상장폐지 같은 hard distress 문맥, 또는 현금흐름/이자보상/손익/레버리지 중 2축 이상의
재무 스트레스가 함께 있을 때 `risk_hold` 근거로 유지한다. 반대로 방어적인 TN 케이스에서
자금조달·채무보증 비율만 큰 경우에는 hidden-tail-risk와 RiskRecallQA의 위험 확정을
막아 `eligible` 또는 `boundary_hold` 쪽으로 남길 수 있다.
반복 채무보증처럼 규모성 공시와 일부 재무약점이 함께 있어 보류 자체는 유지할 필요가
있지만, 현금흐름 악화나 치명 문맥이 없는 경우에는 hidden-tail-risk를 `risk_hold`가 아닌
`review_hold`로 낮춰 표시한다.

배치 결과 CSV에는 Stage 2 실행 진단 컬럼이 함께 남는다. 주요 컬럼은 `stage2_backend_name`, `stage2_llm_cache_hit`, `stage2_total_elapsed_seconds`, `stage2_agent_elapsed_seconds_sum`, `stage2_quant_credit_elapsed_seconds`, `stage2_evidence_audit_elapsed_seconds`, `stage2_chair_report_elapsed_seconds`, `stage2_review_qa_elapsed_seconds`, `stage2_review_qa_triggered`, `stage2_review_qa_trigger_reasons`, `stage2_review_qa_recommended_action`, `stage2_review_qa_advisory_applied`, `stage2_review_qa_advisory_apply_reason`, `stage2_risk_recall_qa_elapsed_seconds`, `stage2_risk_recall_qa_triggered`, `stage2_risk_recall_qa_trigger_reasons`, `stage2_risk_recall_qa_recommended_action`, `stage2_risk_recall_qa_advisory_applied`, `stage2_risk_recall_qa_advisory_apply_reason`, `stage2_parallel_independent_agents`다. 실제 API 속도를 측정할 때는 `stage2_llm_cache_hit=False`인 행을 기준으로 보고, 캐시 재사용 여부를 제거하려면 위 예시처럼 `--no-stage2-llm-cache`를 붙인다.

같은 CSV에는 materiality 확인용 컬럼도 남는다. `materiality_event_count`,
`materiality_substantive_count`, `materiality_watch_count`, `materiality_max_ratio`,
`materiality_top_basis`, `materiality_event_classes`를 보면 OpenDART 상세 공시에서 어떤
비율 근거가 판단에 들어왔는지 결과 파일만으로 확인할 수 있다.

ReviewQAAgent는 Agno runner에서 기본적으로 켜져 있지만, 모든 기업에 실행되지는 않는다. `agent_disagreement_level=high`이면서 치명 외부근거가 제한적인 `risk_hold`/`reject`는 우선 실행하고, `medium`은 `chair_risk_without_critical_evidence`, `chair_reject_without_critical_evidence`, `committee_label_memo_conflict`처럼 라벨과 근거의 충돌을 설명하는 reason이 있을 때만 실행한다. `low` disagreement 케이스는 ReviewQA를 건너뛰어 속도 비용을 줄인다. 이 disagreement score는 QuantCredit, EvidenceAudit, ChairReport/committee_view가 서로 다른 방향을 보는지 기록하는 진단 신호이며, batch 결과 CSV의 `agent_disagreement_score`, `agent_disagreement_level`, `agent_disagreement_reasons`, `agent_disagreement_summary`에서 확인한다. 운영 속도 테스트에서 순수 3-agent 지연시간만 보고 싶으면 `CAS_STAGE2_REVIEW_QA_ENABLED=0`을 추가한다.

ReviewQA는 최종 라벨을 직접 바꾸지 않는다. 다만 `risk_hold`가 과도하다고 권고하고 `veto_triggered=false`, `hidden_tail_risk_flag=false`이면 `committee_decision_type`만 `boundary_hold`로 낮출 수 있다. 또한 ReviewQA가 `risk_hold_without_critical_evidence` 조건에서 downgrade를 권고했고, 외부 공시가 모두 `caution/watch_context/procedural_or_one_off` 수준이며 veto·hidden-tail-risk가 없으면 같은 subtype 보정을 안정적으로 적용한다. 자금조달·채무보증 materiality는 10% 이상이어도 재무 스트레스나 hard distress 문맥이 없으면 단독으로 ReviewQA 보정을 막지 않는다. 이 subtype advisory 적용을 끄고 순수 관찰만 하려면 `CAS_STAGE2_REVIEW_QA_APPLY_ADVISORY=0`을 추가한다.

RiskRecallQAAgent는 ReviewQA의 반대편 안전망이다. 최종 라벨이 `적격`일 때도 확률이 기준선 근처라는 이유만으로는 실행하지 않고, 기준선 근처와 재무 취약 2축 이상이 함께 있거나, 재무 취약 3축 이상이거나, 실질 외부 위험 근거가 있을 때만 실행한다. watch 공시나 BBB-/BB+ 경계 맥락은 단독 trigger가 아니라 이 핵심 조건에 붙는 보조 맥락으로만 남긴다. 적격 판단을 유지하기 어렵다고 권고하면 `boundary_hold` 또는 아주 제한적으로 `risk_hold`로 올릴 수 있다. `eligible_with_substantive_evidence` trigger는 routine 감사보고서나 단순 공시가 아니라 `substantive_adverse`, `veto/critical context`, 또는 횡령·배임·상장폐지·감사의견 거절 같은 명시적 치명 제목에만 켜진다. 단, 자금조달·채무보증은 `materiality_ratio >= 10%`만으로는 충분하지 않고 재무 스트레스 또는 hard distress 문맥이 함께 있어야 실질 외부 위험으로 본다. 속도 테스트에서 끄려면 `CAS_STAGE2_RISK_RECALL_QA_ENABLED=0`, 권고 적용만 끄려면 `CAS_STAGE2_RISK_RECALL_QA_APPLY_ADVISORY=0`을 추가한다.

## Deterministic vs OpenAI Agno 설명 품질 비교

Codex 세션 안에서는 실제 기업-회계연도 평가 맥락을 OpenAI API로 보내는 실행이 차단될 수 있다. 팀/프로젝트 기준으로 외부 전송을 승인한 경우, 아래 명령은 Codex 밖의 로컬 터미널에서 직접 실행한다.

가장 편한 방법:

```bash
cd "/Users/inji/Documents/금융 데이터 분석/Project/Corporate-Analysis-System"

bash scripts/run_openai_agno_comparison_local.sh
```

이 스크립트는 다음을 순서대로 수행한다.

1. `scripts/check_agno_stage2.py`로 OpenAI Agno 설정을 확인한다.
2. 같은 4개 holdout 샘플을 deterministic Stage 2로 실행해 기준선을 만든다.
3. 같은 4개 샘플을 OpenAI Agno 단일 모델(`gpt-4.1-mini`)로 실행한다.
4. `scripts/export_stage2_agno_explanation_comparison.py`로 최종 라벨 변화, 성공 여부 변화, 설명 품질 점수를 비교한다.
5. `scripts/export_stage2_evaluation_report.py`로 통합 Stage 2 평가 리포트를 다시 생성한다.

생성되는 주요 파일:

- `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_openai_agno_comparison_deterministic/committee_review_batch_results.csv`
- `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_openai_agno_comparison_agno/committee_review_batch_results.csv`
- `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_openai_agno_explanation_comparison.md`
- `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_evaluation_report.md`

수동으로 나눠 실행하고 싶을 때:

```bash
cd "/Users/inji/Documents/금융 데이터 분석/Project/Corporate-Analysis-System"

/opt/anaconda3/envs/aura/bin/python scripts/run_committee_review_evaluation_batch.py \
  --samples data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_holdout_unseen_8_samples.csv \
  --output-dir data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_openai_agno_comparison_deterministic \
  --policy balanced_current_45_or_near_threshold_0_10 \
  --per-category 1 \
  --max-cases 4 \
  --stage2-runner deterministic \
  --workers 1

/usr/bin/env \
  CAS_STAGE2_FALLBACK_ON_ERROR=0 \
  /opt/anaconda3/envs/aura/bin/python scripts/run_committee_review_evaluation_batch.py \
  --samples data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_holdout_unseen_8_samples.csv \
  --output-dir data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_openai_agno_comparison_agno \
  --policy balanced_current_45_or_near_threshold_0_10 \
  --per-category 1 \
  --max-cases 4 \
  --stage2-runner agno \
  --stage2-agno-mode single \
  --stage2-model-provider openai \
  --stage2-model gpt-4.1-mini \
  --workers 1

/opt/anaconda3/envs/aura/bin/python scripts/export_stage2_agno_explanation_comparison.py
/opt/anaconda3/envs/aura/bin/python scripts/export_stage2_evaluation_report.py
```

해석 기준:

- `deterministic_success_rate`와 `agno_success_rate`는 같은 샘플에서 Stage 2 판단이 FN/FP/TP/TN 목적에 맞게 작동했는지 비교한다.
- `quality_delta_mean`이 양수면 Agno 설명이 deterministic 설명보다 메모 길이, 핵심 용어, 수치 포함 측면에서 더 풍부해졌다는 뜻이다.
- 최종 판단 라벨은 더 좋아졌는데 설명 품질이 낮아질 수도 있고, 반대로 라벨은 같지만 설명 품질만 좋아질 수도 있으므로 두 지표를 함께 본다.
- 이 비교는 4건 smoke test이므로 성능 일반화 지표가 아니라 “실제 LLM 설명 품질 확인용”으로 해석한다.
- 속도 비교가 필요하면 `committee_review_batch_results.csv`의 Stage 2 실행 진단 컬럼을 함께 확인한다. `single` 모드는 OpenAI 한 provider를 쓰는 3-agent 실행이므로, 역할별 지연은 QuantCredit/EvidenceAudit/ChairReport 컬럼에 나뉘어 기록된다.

## Multi-LLM Committee를 꼭 쓸 때

Claude/Gemini 역할까지 포함하려면 다음이 필요하다.

```bash
/opt/anaconda3/envs/aura/bin/python -m pip install -e ".[agent]"
```

그리고 `.env`에 `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` 또는 `GEMINI_API_KEY`를 추가한 뒤 `--stage2-agno-mode multi_llm_committee`로 실행한다.

## 속도 운영 원칙

Agno/OpenAI live batch는 deterministic guardrail보다 느리다. 실제 대시보드에서는 전체 기업을 보내지 않고 다음 조건 중 하나를 만족하는 기업만 live Agno로 보낸다.

- `stage2_review_trigger=True`
- `stage2_secondary_trigger=True`
- `stage2_review_priority`가 `medium` 또는 `high`
- 직접 관련 외부근거에서 치명 리스크 후보가 확인됨

대시보드에서는 이 원칙을 UI 실행 경로에도 적용한다. `위원회 검토` 탭은 먼저
deterministic Stage 2 결과를 즉시 표시하고, Agno/OpenAI와 외부 뉴스·공시 API는
`Agno 실행` 버튼을 눌렀을 때 백그라운드 작업으로 실행한다. 완료된 결과는
기업-회계연도-모델/runner 설정-외부근거 스냅샷 기준으로 캐시되어 같은 기업을 다시
열 때 API를 재호출하지 않는다. 기본값 `CAS_DASHBOARD_STAGE2_TRIGGER_ONLY=1`은 위
트리거가 있는 기업만 live Agno로 보내며, 운영 점검에서 모든 선택 기업을 강제로
돌리려면 `CAS_DASHBOARD_STAGE2_TRIGGER_ONLY=0`을 사용한다.
