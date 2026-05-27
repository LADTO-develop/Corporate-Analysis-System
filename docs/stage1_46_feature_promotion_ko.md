# Stage 1 46-Feature Promotion Record

작성일: 2026-05-26

## 결정

공식 Stage 1 XGBoost 입력셋을 기존 43개 모델 입력에서 46개 모델 입력으로 승격했다.
새 공식 입력셋 이름은 `credit_46_features`, 모델 artifact 이름은
`feature_46_xgboost`로 둔다.

공식 재생성 명령은 `feature_46_*` 스크립트를 기본 entrypoint로 사용한다.
기존 `feature_43_*` 스크립트 파일명은 누적 실험/운영 히스토리와 CLI 호환성을
위한 wrapper로만 유지한다.

## 추가한 변수

아래 3개 변수는 기존 금액 변수를 대체하지 않고 추가한다.

| 추가 변수 | 원천 변수 | 계산 기준 |
|---|---|---|
| `assets_total_industry_year_pct` | `assets_total` | 같은 `fiscal_year`, `industry_macro_category` 그룹 내 percentile rank |
| `gross_profit_industry_year_pct` | `gross_profit` | 같은 `fiscal_year`, `industry_macro_category` 그룹 내 percentile rank |
| `depreciation_industry_year_pct` | `depreciation` | 같은 `fiscal_year`, `industry_macro_category` 그룹 내 percentile rank |

## 승격 근거

절대규모 변수에 대한 과민반응을 줄이기 위해 로그 변환, 산업-연도 percentile,
시장/규모별 z-score 후보를 비교했다. 2026 외부검증은 표본이 작아 공식 승격
판단에서 제외하고, rolling validation과 final test를 기준으로 판단했다.

`industry_year_amount_pct_add_native` 후보는 원래 43개 입력을 유지하면서 위 3개
산업-연도 금액 백분위를 추가한 46개 입력셋이다.

| 기준 | Baseline 43 | Promoted 46 | 변화 |
|---|---:|---:|---:|
| Rolling validation F1 | 0.7537 | 0.7589 | +0.0052 |
| Rolling validation PR-AUC | 0.8376 | 0.8374 | -0.0002 |
| Final test Precision | 0.7004 | 0.6941 | -0.0063 |
| Final test Recall | 0.8522 | 0.8719 | +0.0197 |
| Final test F1 | 0.7689 | 0.7729 | +0.0040 |
| Final test FP | 74 | 78 | +4 |
| Final test FN | 30 | 26 | -4 |

리스크 조기 탐지 목적에서는 Recall 상승과 FN 감소가 더 중요하므로, FP 4건 증가를
감수하고 46개 입력셋을 공식 모델로 승격한다.

## 변경 경로

| 구분 | 기존 | 신규 |
|---|---|---|
| 입력셋 | `data/input/credit_43_features/` | `data/input/credit_46_features/` |
| 모델 spec | `feature_43_list.json` | `feature_46_list.json` |
| 전체 입력 테이블 | `feature_43_master.csv` | `feature_46_master.csv` |
| 2026 추론 입력 | `feature_43_inference_2026.csv` | `feature_46_inference_2026.csv` |
| 모델 artifact | `data/outputs/modeling/feature_43_xgboost/` | `data/outputs/modeling/feature_46_xgboost/` |
| 대시보드 artifact | `data/outputs/dashboard/feature_43_mvp/` | `data/outputs/dashboard/feature_46_mvp/` |
| 2026 대시보드 artifact | `data/outputs/dashboard/feature_43_inference_2026/` | `data/outputs/dashboard/feature_46_inference_2026/` |

## 재생성 명령

```bash
/opt/anaconda3/envs/aura/bin/python scripts/rebuild_feature_46_dataset.py
/opt/anaconda3/envs/aura/bin/python scripts/build_feature_46_inference_2026.py
/opt/anaconda3/envs/aura/bin/python scripts/build_feature_46_inference_2026.py --check-only
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_dashboard_artifacts.py
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_inference_2026_dashboard_artifacts.py
```

## 관련 산출물

43-feature 기준 중간 실험 diagnostics는 46-feature 공식 승격 후 삭제했다. 이후
공식 artifact 트리에는 현재 운영 기준인 46-feature 모델의 핵심 진단 파일만
남긴다.

승격 판단에 필요한 전후 성능은 이 문서의 `승격 근거` 표에 본문으로 보존한다.
현재 운영 모델의 상세 진단은 아래 파일에서 확인한다.

- `data/outputs/modeling/feature_46_xgboost/diagnostics/model_diagnostics_report.md`
- `data/outputs/modeling/feature_46_xgboost/diagnostics/threshold_policy_experiment_report.md`
