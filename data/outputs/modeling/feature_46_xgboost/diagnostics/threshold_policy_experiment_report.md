# Feature 43 Threshold Policy Experiments

이 리포트는 기존 XGBoost 모델의 확률값은 그대로 두고, decision threshold 정책만
바꿨을 때 test 성능이 어떻게 달라지는지 비교합니다. 모든 threshold는 test가 아닌
validation split에서 선택한 뒤 test에 적용했습니다.

## 1. 핵심 결과

- 현재 artifact threshold 정책: `0.300000`
- 현재 test 성능: Precision `0.6941`,
  Recall `0.8719`, F1 `0.7729`,
  FP `78`, FN `26`
- 이번 실험의 test F1 최상위 정책: `kosdaq_conservative_recall_0.80`
- 최상위 정책 test 성능: Precision `0.7203`,
  Recall `0.8374`, F1 `0.7745`,
  FP `66`, FN `33`
- 현재 대비 변화: F1 `+0.0016`, FP `-12`, FN `+7`

## 2. Test 정책 비교

| Policy | Threshold | Precision | Recall | F1 | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| kosdaq_conservative_recall_0.80 | market=KOSDAQ:0.380; fallback:0.300 | 0.7203 | 0.8374 | 0.7745 | 66 | 33 |
| global_valid_precision_at_recall_0.80 | 0.380000 | 0.7222 | 0.8325 | 0.7735 | 65 | 34 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | market=KOSDAQ:0.380; industry_macro_category=it_services:0.320; industry_macro_category=manufacturing:0.380; fallback:0.300 | 0.7222 | 0.8325 | 0.7735 | 65 | 34 |
| current_artifact_threshold | 0.300000 | 0.6941 | 0.8719 | 0.7729 | 78 | 26 |
| global_valid_best_f1_grid | 0.300000 | 0.6941 | 0.8719 | 0.7729 | 78 | 26 |
| global_valid_precision_at_recall_0.85 | 0.300000 | 0.6941 | 0.8719 | 0.7729 | 78 | 26 |
| kosdaq_conservative_recall_0.85 | market=KOSDAQ:0.300; fallback:0.300 | 0.6941 | 0.8719 | 0.7729 | 78 | 26 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | market=KOSDAQ:0.300; industry_macro_category=it_services:0.300; industry_macro_category=manufacturing:0.300; fallback:0.300 | 0.6941 | 0.8719 | 0.7729 | 78 | 26 |
| industry_valid_best_f1_by_segment | it_services:0.290; manufacturing:0.300; wholesale_retail:0.520; fallback:0.300 | 0.6988 | 0.8571 | 0.7699 | 75 | 29 |
| market_valid_best_f1_by_segment | KOSDAQ:0.300; KOSPI:0.215; fallback:0.300 | 0.6756 | 0.8719 | 0.7613 | 85 | 26 |
| global_valid_precision_at_recall_0.75 | 0.465000 | 0.7537 | 0.7537 | 0.7537 | 50 | 50 |
| global_valid_precision_at_recall_0.90 | 0.225000 | 0.6411 | 0.9064 | 0.7510 | 103 | 19 |
| default_0_5 | 0.500000 | 0.7656 | 0.7241 | 0.7443 | 45 | 56 |

## 3. 세그먼트 Threshold

시장별/산업별 threshold는 validation split에서 세그먼트별 F1이 가장 높은 값을
선택했습니다. KOSDAQ/IT서비스/제조업 targeted 정책은 현재 artifact threshold보다
낮아지지 않는 보수 후보만 사용했습니다. 단, validation 표본이 `30`개
미만이거나 양성 라벨이 `5`개 미만이면 전체 global threshold로
fallback했습니다.

| Policy | Segment | Threshold | Fallback | Valid Rows | Valid Positives | Valid F1 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| industry_valid_best_f1_by_segment | industry_macro_category=construction | 0.300 | yes | 34 | 4 | 0.3636 |
| industry_valid_best_f1_by_segment | industry_macro_category=it_services | 0.290 | no | 129 | 23 | 0.7600 |
| industry_valid_best_f1_by_segment | industry_macro_category=manufacturing | 0.300 | no | 450 | 136 | 0.8013 |
| industry_valid_best_f1_by_segment | industry_macro_category=other | 0.300 | yes | 11 | 0 | 0.0000 |
| industry_valid_best_f1_by_segment | industry_macro_category=transport_storage | 0.300 | yes | 11 | 1 | 1.0000 |
| industry_valid_best_f1_by_segment | industry_macro_category=wholesale_retail | 0.520 | no | 41 | 12 | 0.9565 |
| kosdaq_conservative_recall_0.80 | market=KOSDAQ | 0.380 | no | 403 | 143 | 0.7733 |
| kosdaq_conservative_recall_0.85 | market=KOSDAQ | 0.300 | no | 403 | 143 | 0.7925 |
| market_valid_best_f1_by_segment | market=KOSDAQ | 0.300 | no | 403 | 143 | 0.7925 |
| market_valid_best_f1_by_segment | market=KOSPI | 0.215 | no | 273 | 33 | 0.7692 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=it_services | 0.320 | no | 129 | 23 | 0.7600 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=manufacturing | 0.380 | no | 450 | 136 | 0.7789 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | market=KOSDAQ | 0.380 | no | 403 | 143 | 0.7733 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=it_services | 0.300 | yes | 129 | 23 | 0.7600 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=manufacturing | 0.300 | no | 450 | 136 | 0.8013 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | market=KOSDAQ | 0.300 | no | 403 | 143 | 0.7925 |

## 4. FP 집중 구간 변화

아래 표는 현재 artifact threshold 대비 KOSDAQ, IT서비스, 제조업의 FP/FN이
어떻게 바뀌는지 보여줍니다.

| Policy | Segment | Precision | Recall | F1 | FP | FP Δ | FN | FN Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_artifact_threshold | market=KOSDAQ | 0.7015 | 0.8650 | 0.7747 | 60 | +0 | 22 | +0 |
| current_artifact_threshold | industry_macro_category=it_services | 0.6538 | 0.8095 | 0.7234 | 9 | +0 | 4 | +0 |
| current_artifact_threshold | industry_macro_category=manufacturing | 0.7035 | 0.8642 | 0.7756 | 59 | +0 | 22 | +0 |
| market_valid_best_f1_by_segment | market=KOSDAQ | 0.7015 | 0.8650 | 0.7747 | 60 | +0 | 22 | +0 |
| market_valid_best_f1_by_segment | industry_macro_category=it_services | 0.6538 | 0.8095 | 0.7234 | 9 | +0 | 4 | +0 |
| market_valid_best_f1_by_segment | industry_macro_category=manufacturing | 0.6796 | 0.8642 | 0.7609 | 66 | +7 | 22 | +0 |
| industry_valid_best_f1_by_segment | market=KOSDAQ | 0.7077 | 0.8466 | 0.7709 | 57 | -3 | 25 | +3 |
| industry_valid_best_f1_by_segment | industry_macro_category=it_services | 0.6538 | 0.8095 | 0.7234 | 9 | +0 | 4 | +0 |
| industry_valid_best_f1_by_segment | industry_macro_category=manufacturing | 0.7035 | 0.8642 | 0.7756 | 59 | +0 | 22 | +0 |
| kosdaq_conservative_recall_0.85 | market=KOSDAQ | 0.7015 | 0.8650 | 0.7747 | 60 | +0 | 22 | +0 |
| kosdaq_conservative_recall_0.85 | industry_macro_category=it_services | 0.6538 | 0.8095 | 0.7234 | 9 | +0 | 4 | +0 |
| kosdaq_conservative_recall_0.85 | industry_macro_category=manufacturing | 0.7035 | 0.8642 | 0.7756 | 59 | +0 | 22 | +0 |
| kosdaq_conservative_recall_0.80 | market=KOSDAQ | 0.7363 | 0.8221 | 0.7768 | 48 | -12 | 29 | +7 |
| kosdaq_conservative_recall_0.80 | industry_macro_category=it_services | 0.6818 | 0.7143 | 0.6977 | 7 | -2 | 6 | +2 |
| kosdaq_conservative_recall_0.80 | industry_macro_category=manufacturing | 0.7273 | 0.8395 | 0.7794 | 51 | -8 | 26 | +4 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | market=KOSDAQ | 0.7015 | 0.8650 | 0.7747 | 60 | +0 | 22 | +0 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=it_services | 0.6538 | 0.8095 | 0.7234 | 9 | +0 | 4 | +0 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=manufacturing | 0.7035 | 0.8642 | 0.7756 | 59 | +0 | 22 | +0 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | market=KOSDAQ | 0.7363 | 0.8221 | 0.7768 | 48 | -12 | 29 | +7 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=it_services | 0.6818 | 0.7143 | 0.6977 | 7 | -2 | 6 | +2 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=manufacturing | 0.7297 | 0.8333 | 0.7781 | 50 | -9 | 27 | +5 |

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
