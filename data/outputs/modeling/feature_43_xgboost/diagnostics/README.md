# Feature 43 XGBoost Diagnostics

이 폴더는 43개 변수 XGBoost 모델의 Stage 1 정량 모델 진단 산출물을 보관합니다.
Stage 2 에이전트/위원회 진단은 `stage2_agents/` 하위 폴더로 분리했습니다.

## 폴더 기준

| 위치 | 내용 |
|---|---|
| `diagnostics/` | Stage 1 모델 성능, threshold, calibration, SHAP 오류 분석, 변수셋 비교 |
| `diagnostics/stage2_agents/` | Stage 2 에이전트/위원회 평가, Agno 비교, 파일럿 배치 결과 |

## 자주 보는 Stage 1 파일

| 파일 | 용도 |
|---|---|
| `model_diagnostics_report.md` | 공식 43개 XGBoost 모델의 전체 성능 요약 |
| `feature_43_vs_44_performance_comparison.md` | 43개 변수셋과 44개 후보 변수셋 비교 근거 |
| `official_43_error_deep_dive_report.md` | FP/FN 오류 집중 구간과 경계등급 분석 |
| `error_case_review_report.md` | 오류 사례를 시장/산업/등급 경계 관점에서 정리 |
| `error_shap_analysis_report.md` | 오류 기업의 SHAP 기반 원인 분석 |
| `threshold_policy_experiment_report.md` | global/segment threshold 정책 비교 |
| `xgboost_hyperparameter_tuning_report.md` | XGBoost 하이퍼파라미터 튜닝 결과 |
| `external_validation_2026_report.md` | 2026 외부 신용평가 라벨 기반 검증 요약 |
| `missing_value_supplement_review.md` | 현재 기준 결측 보강 필요성 점검과 2026 추론 입력 OpenDART 보강 결과 |
| `missing_value_inference_2026_opendart_before_after.csv` | 2026 추론 입력 주요 변수 결측 보강 전후 비교 |

## Stage 2 에이전트 진단

에이전트 고도화, Agno 실행, 위원회 판단 성능 자료는 아래에서 확인합니다.

```text
data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/
```
