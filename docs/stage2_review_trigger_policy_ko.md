# Stage 2 Review Trigger Policy

작성일: 2026-05-26

## 결정

Stage 2 에이전트 검토 큐를 넓히는 보조 트리거를 기존 `stage2_aux_48`
기준에서 `full_review_trigger_73` 기준으로 변경한다.

공식 Stage 1 모델은 계속 `feature_46_xgboost`를 사용한다. `full_review_trigger_73`은
최종 위험 판정을 덮어쓰는 모델이 아니라, Stage 1에서 놓칠 수 있는 기업을
Stage 2 에이전트 검토 대상으로 추가로 올리는 recall-oriented 보조 신호다.

대시보드의 최종 `stage2_review_trigger`는 아래 두 신호의 합집합이다.

- `full_review_trigger_73` 보조 XGBoost 검토 트리거
- KOSDAQ 제조업 저확률 FN rescue deterministic gate

## 변수 구성

`full_review_trigger_73`은 공식 Stage 1의 46개 입력 변수에 review-only 변수
27개를 추가한 73개 feature set이다.

추가 변수는 아래 그룹으로 구성한다.

- 기존 Stage2 aux: `delta_accruals_ratio`, `is_3y_consecutive_operating_loss`
- FN rescue composite score: `fn_rescue_working_capital_stress_score`,
  `fn_rescue_cashflow_turn_stress_score`, `fn_rescue_borrowing_pressure_score`,
  `fn_rescue_score`, `fn_rescue_group_count`
- FN rescue 원천 변수: 매출채권/재고/계약자산, 운전자본 회전일수 악화,
  OCF turn, 차입/유동성/자본잠식 악화 변수
- Macro regime: `market_spread_diff`, `spec_spread_diff`, `base_rate_diff`,
  `treasury_3y_diff`, `usd_krw_diff`

## 채택 근거

Rolling OOT 기준에서 `full_review_trigger_73`은 기존 `stage2_aux_48_baseline`
대비 Recall을 높이고 FN을 줄였다. Precision과 F1은 낮아지지만, 이 신호는
정식 라벨 변경용이 아니라 Stage 2 에이전트 검토 큐 확장용이므로 Recall 개선을
우선한다.

| 기준 | stage2_aux_48 | full_review_trigger_73 | 변화 |
|---|---:|---:|---:|
| Rolling Recall | 0.8709 | 0.8800 | +0.0091 |
| Rolling F1 | 0.7571 | 0.7392 | -0.0179 |
| Rolling FP | 274 | 318 | +44 |
| Rolling FN | 81 | 76 | -5 |
| Final Test Recall | 0.8818 | 0.8867 | +0.0049 |
| Final Test F1 | 0.7585 | 0.7453 | -0.0131 |
| Final Test FP | 90 | 100 | +10 |
| Final Test FN | 24 | 23 | -1 |

## 현재 대시보드 기준 성능

`scripts/export_feature_46_dashboard_artifacts.py`를 재실행한 현재 artifact는
`full_review_trigger_73` 보조 모델과 KOSDAQ 제조업 FN rescue gate를 함께
반영한다.

| Split | 기준 | Precision | Recall | F1 | FP | FN |
|---|---|---:|---:|---:|---:|---:|
| Validation | Stage1 46 | 0.7238 | 0.8636 | 0.7876 | 58 | 24 |
| Validation | Stage2 trigger | 0.6809 | 0.9091 | 0.7786 | 75 | 16 |
| Final Test | Stage1 46 | 0.6941 | 0.8719 | 0.7729 | 78 | 26 |
| Final Test | Stage2 trigger | 0.6364 | 0.8966 | 0.7444 | 104 | 21 |

Final Test 기준으로 Stage2 trigger는 Stage1 대비 FN을 26건에서 21건으로 줄이고,
Recall을 0.8719에서 0.8966으로 높인다. 대신 FP는 78건에서 104건으로 늘어난다.

## 재생성 명령

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_stage2_trigger_feature_experiments.py
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_dashboard_artifacts.py
```

## 관련 산출물

- `data/outputs/modeling/feature_46_xgboost/diagnostics/stage2_trigger_feature_set_report.md`
- `data/outputs/dashboard/feature_46_mvp/model_summary.json`
- `data/outputs/dashboard/feature_46_mvp/stage2_review_signals.csv`
