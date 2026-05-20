# Feature 43 Threshold Policy Experiments

이 리포트는 기존 XGBoost 모델의 확률값은 그대로 두고, decision threshold 정책만
바꿨을 때 test 성능이 어떻게 달라지는지 비교합니다. 모든 threshold는 test가 아닌
validation split에서 선택한 뒤 test에 적용했습니다.

## 1. 핵심 결과

- 현재 artifact threshold 정책: `0.315000`
- 현재 test 성능: Precision `0.6603`,
  Recall `0.8522`, F1 `0.7441`,
  FP `89`, FN `30`
- 이번 실험의 test F1 최상위 정책: `industry_valid_best_f1_by_segment`
- 최상위 정책 test 성능: Precision `0.6787`,
  Recall `0.8325`, F1 `0.7478`,
  FP `80`, FN `34`
- 현재 대비 변화: F1 `+0.0037`, FP `-9`, FN `+4`

## 2. Test 정책 비교

| Policy | Threshold | Precision | Recall | F1 | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| industry_valid_best_f1_by_segment | it_services:0.310; manufacturing:0.335; wholesale_retail:0.500; fallback:0.285 | 0.6787 | 0.8325 | 0.7478 | 80 | 34 |
| kosdaq_conservative_recall_0.85 | market=KOSDAQ:0.330; fallback:0.315 | 0.6733 | 0.8325 | 0.7445 | 82 | 34 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | market=KOSDAQ:0.330; industry_macro_category=it_services:0.315; industry_macro_category=manufacturing:0.320; fallback:0.315 | 0.6733 | 0.8325 | 0.7445 | 82 | 34 |
| current_artifact_threshold | 0.315000 | 0.6603 | 0.8522 | 0.7441 | 89 | 30 |
| global_valid_precision_at_recall_0.85 | 0.315000 | 0.6603 | 0.8522 | 0.7441 | 89 | 30 |
| kosdaq_conservative_recall_0.80 | market=KOSDAQ:0.350; fallback:0.315 | 0.6789 | 0.8227 | 0.7439 | 79 | 36 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | market=KOSDAQ:0.350; industry_macro_category=it_services:0.315; industry_macro_category=manufacturing:0.345; fallback:0.315 | 0.6803 | 0.8177 | 0.7427 | 78 | 37 |
| global_valid_precision_at_recall_0.80 | 0.350000 | 0.6818 | 0.8128 | 0.7416 | 77 | 38 |
| market_valid_best_f1_by_segment | KOSDAQ:0.335; KOSPI:0.265; fallback:0.285 | 0.6667 | 0.8276 | 0.7385 | 84 | 35 |
| global_valid_best_f1_grid | 0.285000 | 0.6327 | 0.8571 | 0.7280 | 101 | 29 |
| global_valid_precision_at_recall_0.75 | 0.410000 | 0.6920 | 0.7635 | 0.7260 | 69 | 48 |
| global_valid_precision_at_recall_0.90 | 0.230000 | 0.5980 | 0.9015 | 0.7191 | 123 | 20 |
| default_0_5 | 0.500000 | 0.7302 | 0.6798 | 0.7041 | 51 | 65 |

## 3. 세그먼트 Threshold

시장별/산업별 threshold는 validation split에서 세그먼트별 F1이 가장 높은 값을
선택했습니다. KOSDAQ/IT서비스/제조업 targeted 정책은 현재 artifact threshold보다
낮아지지 않는 보수 후보만 사용했습니다. 단, validation 표본이 `30`개
미만이거나 양성 라벨이 `5`개 미만이면 전체 global threshold로
fallback했습니다.

| Policy | Segment | Threshold | Fallback | Valid Rows | Valid Positives | Valid F1 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| industry_valid_best_f1_by_segment | industry_macro_category=construction | 0.285 | yes | 34 | 4 | 0.7500 |
| industry_valid_best_f1_by_segment | industry_macro_category=it_services | 0.310 | no | 129 | 23 | 0.6667 |
| industry_valid_best_f1_by_segment | industry_macro_category=manufacturing | 0.335 | no | 450 | 136 | 0.7625 |
| industry_valid_best_f1_by_segment | industry_macro_category=other | 0.285 | yes | 11 | 0 | 0.0000 |
| industry_valid_best_f1_by_segment | industry_macro_category=transport_storage | 0.285 | yes | 11 | 1 | 1.0000 |
| industry_valid_best_f1_by_segment | industry_macro_category=wholesale_retail | 0.500 | no | 41 | 12 | 0.9565 |
| kosdaq_conservative_recall_0.80 | market=KOSDAQ | 0.350 | no | 403 | 143 | 0.7452 |
| kosdaq_conservative_recall_0.85 | market=KOSDAQ | 0.330 | no | 403 | 143 | 0.7508 |
| market_valid_best_f1_by_segment | market=KOSDAQ | 0.335 | no | 403 | 143 | 0.7516 |
| market_valid_best_f1_by_segment | market=KOSPI | 0.265 | no | 273 | 33 | 0.8000 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=it_services | 0.315 | no | 129 | 23 | 0.6667 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=manufacturing | 0.345 | no | 450 | 136 | 0.7619 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | market=KOSDAQ | 0.350 | no | 403 | 143 | 0.7452 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=it_services | 0.315 | yes | 129 | 23 | 0.6667 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=manufacturing | 0.320 | no | 450 | 136 | 0.7573 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | market=KOSDAQ | 0.330 | no | 403 | 143 | 0.7508 |

## 4. FP 집중 구간 변화

아래 표는 현재 artifact threshold 대비 KOSDAQ, IT서비스, 제조업의 FP/FN이
어떻게 바뀌는지 보여줍니다.

| Policy | Segment | Precision | Recall | F1 | FP | FP Δ | FN | FN Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_artifact_threshold | market=KOSDAQ | 0.6587 | 0.8405 | 0.7385 | 71 | +0 | 26 | +0 |
| current_artifact_threshold | industry_macro_category=it_services | 0.5294 | 0.8571 | 0.6545 | 16 | +0 | 3 | +0 |
| current_artifact_threshold | industry_macro_category=manufacturing | 0.6733 | 0.8395 | 0.7473 | 66 | +0 | 26 | +0 |
| market_valid_best_f1_by_segment | market=KOSDAQ | 0.6769 | 0.8098 | 0.7374 | 63 | -8 | 31 | +5 |
| market_valid_best_f1_by_segment | industry_macro_category=it_services | 0.5161 | 0.7619 | 0.6154 | 15 | -1 | 5 | +2 |
| market_valid_best_f1_by_segment | industry_macro_category=manufacturing | 0.6856 | 0.8210 | 0.7472 | 61 | -5 | 29 | +3 |
| industry_valid_best_f1_by_segment | market=KOSDAQ | 0.6802 | 0.8221 | 0.7444 | 63 | -8 | 29 | +3 |
| industry_valid_best_f1_by_segment | industry_macro_category=it_services | 0.5294 | 0.8571 | 0.6545 | 16 | +0 | 3 | +0 |
| industry_valid_best_f1_by_segment | industry_macro_category=manufacturing | 0.6947 | 0.8148 | 0.7500 | 58 | -8 | 30 | +4 |
| kosdaq_conservative_recall_0.85 | market=KOSDAQ | 0.6751 | 0.8160 | 0.7389 | 64 | -7 | 30 | +4 |
| kosdaq_conservative_recall_0.85 | industry_macro_category=it_services | 0.5312 | 0.8095 | 0.6415 | 15 | -1 | 4 | +1 |
| kosdaq_conservative_recall_0.85 | industry_macro_category=manufacturing | 0.6891 | 0.8210 | 0.7493 | 60 | -6 | 29 | +3 |
| kosdaq_conservative_recall_0.80 | market=KOSDAQ | 0.6823 | 0.8037 | 0.7380 | 61 | -10 | 32 | +6 |
| kosdaq_conservative_recall_0.80 | industry_macro_category=it_services | 0.5000 | 0.7143 | 0.5882 | 15 | -1 | 6 | +3 |
| kosdaq_conservative_recall_0.80 | industry_macro_category=manufacturing | 0.6963 | 0.8210 | 0.7535 | 58 | -8 | 29 | +3 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | market=KOSDAQ | 0.6751 | 0.8160 | 0.7389 | 64 | -7 | 30 | +4 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=it_services | 0.5312 | 0.8095 | 0.6415 | 15 | -1 | 4 | +1 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=manufacturing | 0.6891 | 0.8210 | 0.7493 | 60 | -6 | 29 | +3 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | market=KOSDAQ | 0.6823 | 0.8037 | 0.7380 | 61 | -10 | 32 | +6 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=it_services | 0.5000 | 0.7143 | 0.5882 | 15 | -1 | 6 | +3 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=manufacturing | 0.6984 | 0.8148 | 0.7521 | 57 | -9 | 30 | +4 |

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
