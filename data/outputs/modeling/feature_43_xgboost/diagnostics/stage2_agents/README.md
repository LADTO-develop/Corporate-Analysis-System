# Stage 2 Agent Diagnostics

이 폴더는 2차 에이전트 위원회 판단을 평가하고 고도화하기 위한 산출물을 보관합니다.
1차 XGBoost 모델 자체의 성능 진단은 상위 `diagnostics/` 폴더에 둡니다.

## 자주 보는 파일

| 파일 | 용도 |
|---|---|
| `stage2_agent_improvement_summary.md` | 오늘 기준 에이전트 고도화 핵심 성과 요약 |
| `stage2_agent_performance_evidence.md` | 파일럿별 성능 변화와 해석 근거 |
| `stage2_evaluation_report.md` | Stage 2 평가 결과를 하나로 묶은 통합 리포트 |
| `stage2_validation_test_policy_report.md` | validation/test 기준 정책 성능과 decision trace 기여도 |
| `stage2_validation_test_trace_gate_contribution.csv` | veto, 경계등급, 과민경고 완화 등 게이트별 기여 수치 |
| `stage2_openai_agno_explanation_comparison.md` | deterministic 판단과 OpenAI Agno 설명 품질 비교 |
| `stage2_agent_agno_hold_subtype_metrics.csv` | Agno 파일럿의 보류 세부 유형별 성능 |
| `stage2_agent_error_risk_10_agno_metrics.csv` | 오류 위험 기업 10건 Agno 파일럿 성능 |

## 샘플/배치 파일

| 패턴 | 용도 |
|---|---|
| `committee_review_*_samples.csv` | 파일럿 실행 대상 기업-연도 샘플 |
| `committee_review_*_results.csv` | 위원회 배치 실행 결과 |
| `committee_review_openai_agno_comparison_*` | OpenAI Agno 설명 비교용 deterministic/Agno 실행 결과 |

## 재생성 스크립트

| 스크립트 | 생성 위치 |
|---|---|
| `scripts/export_committee_review_evaluation_plan.py` | 샘플링 계획과 historical/2026 후보 샘플 |
| `scripts/export_stage2_rolling_validation_samples.py` | rolling validation 기반 Stage 2 튜닝 샘플 |
| `scripts/run_committee_review_evaluation_batch.py` | deterministic 또는 Agno 위원회 배치 결과 |
| `scripts/export_stage2_validation_test_policy_evaluation.py` | validation/test 정책 성능과 decision trace 기여도 |
| `scripts/export_stage2_agno_explanation_comparison.py` | deterministic vs OpenAI Agno 설명 품질 비교 |
| `scripts/export_stage2_evaluation_report.py` | Stage 2 통합 평가 리포트 |
