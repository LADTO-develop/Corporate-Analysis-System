# Feature 43 Threshold Policy Experiments

이 리포트는 기존 XGBoost 모델의 확률값은 그대로 두고, decision threshold 정책만
바꿨을 때 test 성능이 어떻게 달라지는지 비교합니다. 모든 threshold는 test가 아닌
validation split에서 선택한 뒤 test에 적용했습니다.

## 1. 핵심 결과

- 현재 artifact threshold 정책: `0.315000`
- 현재 test 성능: Precision `0.6542`,
  Recall `0.8383`, F1 `0.7349`,
  FP `74`, FN `27`
- 이번 실험의 test F1 최상위 정책: `current_artifact_threshold`
- 최상위 정책 test 성능: Precision `0.6542`,
  Recall `0.8383`, F1 `0.7349`,
  FP `74`, FN `27`
- 현재 대비 변화: F1 `+0.0000`, FP `+0`, FN `+0`

## 2. Test 정책 비교

| Policy | Threshold | Precision | Recall | F1 | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| current_artifact_threshold | 0.315000 | 0.6542 | 0.8383 | 0.7349 | 74 | 27 |
| global_valid_precision_at_recall_0.85 | 0.315000 | 0.6542 | 0.8383 | 0.7349 | 74 | 27 |
| global_valid_best_f1_grid | 0.340000 | 0.6667 | 0.8144 | 0.7332 | 68 | 31 |
| industry_valid_best_f1_by_segment | it_services:0.350; manufacturing:0.340; wholesale_retail:0.455; fallback:0.340 | 0.6683 | 0.8084 | 0.7317 | 67 | 32 |
| kosdaq_conservative_recall_0.85 | market=KOSDAQ:0.340; fallback:0.315 | 0.6602 | 0.8144 | 0.7292 | 70 | 31 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | market=KOSDAQ:0.340; industry_macro_category=it_services:0.350; industry_macro_category=manufacturing:0.315; fallback:0.315 | 0.6602 | 0.8144 | 0.7292 | 70 | 31 |
| kosdaq_conservative_recall_0.80 | market=KOSDAQ:0.360; fallback:0.315 | 0.6617 | 0.7964 | 0.7228 | 68 | 34 |
| global_valid_precision_at_recall_0.80 | 0.360000 | 0.6684 | 0.7844 | 0.7218 | 65 | 36 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | market=KOSDAQ:0.360; industry_macro_category=it_services:0.350; industry_macro_category=manufacturing:0.360; fallback:0.315 | 0.6633 | 0.7904 | 0.7213 | 67 | 35 |
| market_valid_best_f1_by_segment | KOSDAQ:0.340; KOSPI:0.225; fallback:0.340 | 0.6415 | 0.8144 | 0.7177 | 76 | 31 |
| global_valid_precision_at_recall_0.90 | 0.255000 | 0.5976 | 0.8802 | 0.7119 | 99 | 20 |
| global_valid_precision_at_recall_0.75 | 0.425000 | 0.6684 | 0.7485 | 0.7062 | 62 | 42 |
| default_0_5 | 0.500000 | 0.7019 | 0.6766 | 0.6890 | 48 | 54 |

## 3. 세그먼트 Threshold

시장별/산업별 threshold는 validation split에서 세그먼트별 F1이 가장 높은 값을
선택했습니다. KOSDAQ/IT서비스/제조업 targeted 정책은 현재 artifact threshold보다
낮아지지 않는 보수 후보만 사용했습니다. 단, validation 표본이 `30`개
미만이거나 양성 라벨이 `5`개 미만이면 전체 global threshold로
fallback했습니다.

| Policy | Segment | Threshold | Fallback | Valid Rows | Valid Positives | Valid F1 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| industry_valid_best_f1_by_segment | industry_macro_category=construction | 0.340 | yes | 34 | 4 | 0.5714 |
| industry_valid_best_f1_by_segment | industry_macro_category=it_services | 0.350 | no | 129 | 23 | 0.7273 |
| industry_valid_best_f1_by_segment | industry_macro_category=manufacturing | 0.340 | no | 450 | 136 | 0.7729 |
| industry_valid_best_f1_by_segment | industry_macro_category=other | 0.340 | yes | 11 | 0 | 0.0000 |
| industry_valid_best_f1_by_segment | industry_macro_category=transport_storage | 0.340 | yes | 11 | 1 | 1.0000 |
| industry_valid_best_f1_by_segment | industry_macro_category=wholesale_retail | 0.455 | no | 41 | 12 | 0.9565 |
| kosdaq_conservative_recall_0.80 | market=KOSDAQ | 0.360 | no | 403 | 143 | 0.7557 |
| kosdaq_conservative_recall_0.85 | market=KOSDAQ | 0.340 | no | 403 | 143 | 0.7601 |
| market_valid_best_f1_by_segment | market=KOSDAQ | 0.340 | no | 403 | 143 | 0.7601 |
| market_valid_best_f1_by_segment | market=KOSPI | 0.225 | no | 273 | 33 | 0.7949 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=it_services | 0.350 | no | 129 | 23 | 0.7273 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=manufacturing | 0.360 | no | 450 | 136 | 0.7676 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | market=KOSDAQ | 0.360 | no | 403 | 143 | 0.7557 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=it_services | 0.350 | no | 129 | 23 | 0.7273 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=manufacturing | 0.315 | no | 450 | 136 | 0.7557 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | market=KOSDAQ | 0.340 | no | 403 | 143 | 0.7601 |

## 4. FP 집중 구간 변화

아래 표는 현재 artifact threshold 대비 KOSDAQ, IT서비스, 제조업의 FP/FN이
어떻게 바뀌는지 보여줍니다.

| Policy | Segment | Precision | Recall | F1 | FP | FP Δ | FN | FN Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_artifact_threshold | market=KOSDAQ | 0.6425 | 0.8333 | 0.7256 | 64 | +0 | 23 | +0 |
| current_artifact_threshold | industry_macro_category=it_services | 0.4839 | 0.7143 | 0.5769 | 16 | +0 | 6 | +0 |
| current_artifact_threshold | industry_macro_category=manufacturing | 0.6914 | 0.8485 | 0.7619 | 50 | +0 | 20 | +0 |
| market_valid_best_f1_by_segment | market=KOSDAQ | 0.6491 | 0.8043 | 0.7184 | 60 | -4 | 27 | +4 |
| market_valid_best_f1_by_segment | industry_macro_category=it_services | 0.4839 | 0.7143 | 0.5769 | 16 | +0 | 6 | +0 |
| market_valid_best_f1_by_segment | industry_macro_category=manufacturing | 0.6750 | 0.8182 | 0.7397 | 52 | +2 | 24 | +4 |
| industry_valid_best_f1_by_segment | market=KOSDAQ | 0.6509 | 0.7971 | 0.7166 | 59 | -5 | 28 | +5 |
| industry_valid_best_f1_by_segment | industry_macro_category=it_services | 0.4839 | 0.7143 | 0.5769 | 16 | +0 | 6 | +0 |
| industry_valid_best_f1_by_segment | industry_macro_category=manufacturing | 0.7059 | 0.8182 | 0.7579 | 45 | -5 | 24 | +4 |
| kosdaq_conservative_recall_0.85 | market=KOSDAQ | 0.6491 | 0.8043 | 0.7184 | 60 | -4 | 27 | +4 |
| kosdaq_conservative_recall_0.85 | industry_macro_category=it_services | 0.4839 | 0.7143 | 0.5769 | 16 | +0 | 6 | +0 |
| kosdaq_conservative_recall_0.85 | industry_macro_category=manufacturing | 0.7013 | 0.8182 | 0.7552 | 46 | -4 | 24 | +4 |
| kosdaq_conservative_recall_0.80 | market=KOSDAQ | 0.6506 | 0.7826 | 0.7105 | 58 | -6 | 30 | +7 |
| kosdaq_conservative_recall_0.80 | industry_macro_category=it_services | 0.4667 | 0.6667 | 0.5490 | 16 | +0 | 7 | +1 |
| kosdaq_conservative_recall_0.80 | industry_macro_category=manufacturing | 0.7067 | 0.8030 | 0.7518 | 44 | -6 | 26 | +6 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | market=KOSDAQ | 0.6491 | 0.8043 | 0.7184 | 60 | -4 | 27 | +4 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=it_services | 0.4839 | 0.7143 | 0.5769 | 16 | +0 | 6 | +0 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=manufacturing | 0.7013 | 0.8182 | 0.7552 | 46 | -4 | 24 | +4 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | market=KOSDAQ | 0.6506 | 0.7826 | 0.7105 | 58 | -6 | 30 | +7 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=it_services | 0.4667 | 0.6667 | 0.5490 | 16 | +0 | 7 | +1 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=manufacturing | 0.7095 | 0.7955 | 0.7500 | 43 | -7 | 27 | +7 |

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
