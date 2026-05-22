# Feature 43 Threshold Policy Experiments

이 리포트는 기존 XGBoost 모델의 확률값은 그대로 두고, decision threshold 정책만
바꿨을 때 test 성능이 어떻게 달라지는지 비교합니다. 모든 threshold는 test가 아닌
validation split에서 선택한 뒤 test에 적용했습니다.

## 1. 핵심 결과

- 현재 artifact threshold 정책: `0.320000`
- 현재 test 성능: Precision `0.7004`,
  Recall `0.8522`, F1 `0.7689`,
  FP `74`, FN `30`
- 이번 실험의 test F1 최상위 정책: `global_valid_precision_at_recall_0.75`
- 최상위 정책 test 성능: Precision `0.7547`,
  Recall `0.7882`, F1 `0.7711`,
  FP `52`, FN `43`
- 현재 대비 변화: F1 `+0.0022`, FP `-22`, FN `+13`

## 2. Test 정책 비교

| Policy | Threshold | Precision | Recall | F1 | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| global_valid_precision_at_recall_0.75 | 0.435000 | 0.7547 | 0.7882 | 0.7711 | 52 | 43 |
| current_artifact_threshold | 0.320000 | 0.7004 | 0.8522 | 0.7689 | 74 | 30 |
| global_valid_precision_at_recall_0.85 | 0.320000 | 0.7004 | 0.8522 | 0.7689 | 74 | 30 |
| kosdaq_conservative_recall_0.85 | market=KOSDAQ:0.335; fallback:0.320 | 0.7054 | 0.8374 | 0.7658 | 71 | 33 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | market=KOSDAQ:0.335; industry_macro_category=it_services:0.320; industry_macro_category=manufacturing:0.320; fallback:0.320 | 0.7054 | 0.8374 | 0.7658 | 71 | 33 |
| kosdaq_conservative_recall_0.80 | market=KOSDAQ:0.375; fallback:0.320 | 0.7155 | 0.8177 | 0.7632 | 66 | 37 |
| global_valid_precision_at_recall_0.80 | 0.375000 | 0.7225 | 0.8079 | 0.7628 | 63 | 39 |
| market_valid_best_f1_by_segment | KOSDAQ:0.280; KOSPI:0.310; fallback:0.280 | 0.6756 | 0.8719 | 0.7613 | 85 | 26 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | market=KOSDAQ:0.375; industry_macro_category=it_services:0.375; industry_macro_category=manufacturing:0.385; fallback:0.320 | 0.7193 | 0.8079 | 0.7610 | 64 | 39 |
| global_valid_best_f1_grid | 0.280000 | 0.6717 | 0.8768 | 0.7607 | 87 | 25 |
| industry_valid_best_f1_by_segment | it_services:0.345; manufacturing:0.285; wholesale_retail:0.435; fallback:0.280 | 0.6757 | 0.8621 | 0.7576 | 84 | 28 |
| global_valid_precision_at_recall_0.90 | 0.240000 | 0.6487 | 0.8916 | 0.7510 | 98 | 22 |
| default_0_5 | 0.500000 | 0.7737 | 0.7241 | 0.7481 | 43 | 56 |

## 3. 세그먼트 Threshold

시장별/산업별 threshold는 validation split에서 세그먼트별 F1이 가장 높은 값을
선택했습니다. KOSDAQ/IT서비스/제조업 targeted 정책은 현재 artifact threshold보다
낮아지지 않는 보수 후보만 사용했습니다. 단, validation 표본이 `30`개
미만이거나 양성 라벨이 `5`개 미만이면 전체 global threshold로
fallback했습니다.

| Policy | Segment | Threshold | Fallback | Valid Rows | Valid Positives | Valid F1 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| industry_valid_best_f1_by_segment | industry_macro_category=construction | 0.280 | yes | 34 | 4 | 0.4444 |
| industry_valid_best_f1_by_segment | industry_macro_category=it_services | 0.345 | no | 129 | 23 | 0.7917 |
| industry_valid_best_f1_by_segment | industry_macro_category=manufacturing | 0.285 | no | 450 | 136 | 0.8000 |
| industry_valid_best_f1_by_segment | industry_macro_category=other | 0.280 | yes | 11 | 0 | 0.0000 |
| industry_valid_best_f1_by_segment | industry_macro_category=transport_storage | 0.280 | yes | 11 | 1 | 1.0000 |
| industry_valid_best_f1_by_segment | industry_macro_category=wholesale_retail | 0.435 | no | 41 | 12 | 0.9565 |
| kosdaq_conservative_recall_0.80 | market=KOSDAQ | 0.375 | no | 403 | 143 | 0.7855 |
| kosdaq_conservative_recall_0.85 | market=KOSDAQ | 0.335 | no | 403 | 143 | 0.7910 |
| market_valid_best_f1_by_segment | market=KOSDAQ | 0.280 | no | 403 | 143 | 0.7975 |
| market_valid_best_f1_by_segment | market=KOSPI | 0.310 | no | 273 | 33 | 0.7714 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=it_services | 0.375 | no | 129 | 23 | 0.7917 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=manufacturing | 0.385 | no | 450 | 136 | 0.7703 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | market=KOSDAQ | 0.375 | no | 403 | 143 | 0.7855 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=it_services | 0.320 | yes | 129 | 23 | 0.7600 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=manufacturing | 0.320 | no | 450 | 136 | 0.7959 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | market=KOSDAQ | 0.335 | no | 403 | 143 | 0.7910 |

## 4. FP 집중 구간 변화

아래 표는 현재 artifact threshold 대비 KOSDAQ, IT서비스, 제조업의 FP/FN이
어떻게 바뀌는지 보여줍니다.

| Policy | Segment | Precision | Recall | F1 | FP | FP Δ | FN | FN Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_artifact_threshold | market=KOSDAQ | 0.7113 | 0.8466 | 0.7731 | 56 | +0 | 25 | +0 |
| current_artifact_threshold | industry_macro_category=it_services | 0.6400 | 0.7619 | 0.6957 | 9 | +0 | 5 | +0 |
| current_artifact_threshold | industry_macro_category=manufacturing | 0.7041 | 0.8519 | 0.7709 | 58 | +0 | 24 | +0 |
| market_valid_best_f1_by_segment | market=KOSDAQ | 0.6827 | 0.8712 | 0.7655 | 66 | +10 | 21 | -4 |
| market_valid_best_f1_by_segment | industry_macro_category=it_services | 0.6154 | 0.7619 | 0.6809 | 10 | +1 | 5 | +0 |
| market_valid_best_f1_by_segment | industry_macro_category=manufacturing | 0.6794 | 0.8765 | 0.7655 | 67 | +9 | 20 | -4 |
| industry_valid_best_f1_by_segment | market=KOSDAQ | 0.6847 | 0.8528 | 0.7596 | 64 | +8 | 24 | -1 |
| industry_valid_best_f1_by_segment | industry_macro_category=it_services | 0.6250 | 0.7143 | 0.6667 | 9 | +0 | 6 | +1 |
| industry_valid_best_f1_by_segment | industry_macro_category=manufacturing | 0.6762 | 0.8765 | 0.7634 | 68 | +10 | 20 | -4 |
| kosdaq_conservative_recall_0.85 | market=KOSDAQ | 0.7181 | 0.8282 | 0.7692 | 53 | -3 | 28 | +3 |
| kosdaq_conservative_recall_0.85 | industry_macro_category=it_services | 0.6250 | 0.7143 | 0.6667 | 9 | +0 | 6 | +1 |
| kosdaq_conservative_recall_0.85 | industry_macro_category=manufacturing | 0.7135 | 0.8457 | 0.7740 | 55 | -3 | 25 | +1 |
| kosdaq_conservative_recall_0.80 | market=KOSDAQ | 0.7318 | 0.8037 | 0.7661 | 48 | -8 | 32 | +7 |
| kosdaq_conservative_recall_0.80 | industry_macro_category=it_services | 0.6364 | 0.6667 | 0.6512 | 8 | -1 | 7 | +2 |
| kosdaq_conservative_recall_0.80 | industry_macro_category=manufacturing | 0.7258 | 0.8333 | 0.7759 | 51 | -7 | 27 | +3 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | market=KOSDAQ | 0.7181 | 0.8282 | 0.7692 | 53 | -3 | 28 | +3 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=it_services | 0.6250 | 0.7143 | 0.6667 | 9 | +0 | 6 | +1 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=manufacturing | 0.7135 | 0.8457 | 0.7740 | 55 | -3 | 25 | +1 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | market=KOSDAQ | 0.7318 | 0.8037 | 0.7661 | 48 | -8 | 32 | +7 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=it_services | 0.6364 | 0.6667 | 0.6512 | 8 | -1 | 7 | +2 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=manufacturing | 0.7308 | 0.8210 | 0.7733 | 49 | -9 | 29 | +5 |

## 5. 해석

- 현재 artifact threshold는 `global_valid_precision_at_recall_0.85` 정책과 동일하며,
  Recall 0.85 이상을 유지하면서 false positive를 줄이는 단순한 운영 기준입니다.
- 더 높은 Recall을 최우선으로 두면 `global_valid_precision_at_recall_0.90`을,
  false positive 축소를 더 중시하면 `global_valid_precision_at_recall_0.80`을
  보조 기준으로 비교할 수 있습니다.
- KOSDAQ 보수 threshold는 전체 F1을 거의 유지하면서 KOSDAQ FP를 줄이는 후보입니다.
- 성능 숫자만 보면 `industry_valid_best_f1_by_segment`가 가장 좋지만, validation이
  한 해뿐이라 산업별 threshold는 추가 기간 검증 후 production 반영을 권장합니다.
- 발표에서는 "모델 확률은 그대로 두고, 경고 기준선을 목적에 따라 조정할 수 있다"는
  메시지로 설명하면 좋습니다.

## 6. 산출물

- `threshold_policy_experiment_metrics.csv`: 정책별 valid/test 성능
- `threshold_policy_segment_thresholds.csv`: 시장/산업별 threshold와 fallback 여부
- `threshold_policy_focus_segment_metrics.csv`: KOSDAQ/IT서비스/제조업 집중 성능
- `threshold_policy_experiment_summary.json`: 주요 결과 요약
