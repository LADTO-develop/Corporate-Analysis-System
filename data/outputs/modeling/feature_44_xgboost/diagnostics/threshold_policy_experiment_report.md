# Feature 43 Threshold Policy Experiments

이 리포트는 기존 XGBoost 모델의 확률값은 그대로 두고, decision threshold 정책만
바꿨을 때 test 성능이 어떻게 달라지는지 비교합니다. 모든 threshold는 test가 아닌
validation split에서 선택한 뒤 test에 적용했습니다.

## 1. 핵심 결과

- 현재 artifact threshold 정책: `0.305000`
- 현재 test 성능: Precision `0.6196`,
  Recall `0.8424`, F1 `0.7140`,
  FP `105`, FN `32`
- 이번 실험의 test F1 최상위 정책: `kosdaq_conservative_recall_0.85`
- 최상위 정책 test 성능: Precision `0.6409`,
  Recall `0.8177`, F1 `0.7186`,
  FP `93`, FN `37`
- 현재 대비 변화: F1 `+0.0046`, FP `-12`, FN `+5`

## 2. Test 정책 비교

| Policy | Threshold | Precision | Recall | F1 | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| kosdaq_conservative_recall_0.85 | market=KOSDAQ:0.315; fallback:0.305 | 0.6409 | 0.8177 | 0.7186 | 93 | 37 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | market=KOSDAQ:0.315; industry_macro_category=it_services:0.305; industry_macro_category=manufacturing:0.305; fallback:0.305 | 0.6409 | 0.8177 | 0.7186 | 93 | 37 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | market=KOSDAQ:0.375; industry_macro_category=it_services:0.380; industry_macro_category=manufacturing:0.360; fallback:0.305 | 0.6625 | 0.7833 | 0.7178 | 81 | 44 |
| kosdaq_conservative_recall_0.80 | market=KOSDAQ:0.375; fallback:0.305 | 0.6584 | 0.7882 | 0.7175 | 83 | 43 |
| global_valid_precision_at_recall_0.80 | 0.375000 | 0.6639 | 0.7783 | 0.7166 | 80 | 45 |
| market_valid_best_f1_by_segment | KOSDAQ:0.360; KOSPI:0.370; fallback:0.360 | 0.6598 | 0.7833 | 0.7162 | 82 | 44 |
| global_valid_best_f1_grid | 0.360000 | 0.6570 | 0.7833 | 0.7146 | 83 | 44 |
| industry_valid_best_f1_by_segment | it_services:0.370; manufacturing:0.360; wholesale_retail:0.380; fallback:0.360 | 0.6570 | 0.7833 | 0.7146 | 83 | 44 |
| current_artifact_threshold | 0.305000 | 0.6196 | 0.8424 | 0.7140 | 105 | 32 |
| global_valid_precision_at_recall_0.85 | 0.305000 | 0.6196 | 0.8424 | 0.7140 | 105 | 32 |
| global_valid_precision_at_recall_0.75 | 0.380000 | 0.6624 | 0.7734 | 0.7136 | 80 | 46 |
| global_valid_precision_at_recall_0.90 | 0.245000 | 0.5967 | 0.8818 | 0.7117 | 121 | 24 |
| default_0_5 | 0.500000 | 0.7725 | 0.6355 | 0.6973 | 38 | 74 |

## 3. 세그먼트 Threshold

시장별/산업별 threshold는 validation split에서 세그먼트별 F1이 가장 높은 값을
선택했습니다. KOSDAQ/IT서비스/제조업 targeted 정책은 현재 artifact threshold보다
낮아지지 않는 보수 후보만 사용했습니다. 단, validation 표본이 `30`개
미만이거나 양성 라벨이 `5`개 미만이면 전체 global threshold로
fallback했습니다.

| Policy | Segment | Threshold | Fallback | Valid Rows | Valid Positives | Valid F1 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| industry_valid_best_f1_by_segment | industry_macro_category=construction | 0.360 | yes | 34 | 4 | 0.5714 |
| industry_valid_best_f1_by_segment | industry_macro_category=it_services | 0.370 | no | 129 | 23 | 0.7037 |
| industry_valid_best_f1_by_segment | industry_macro_category=manufacturing | 0.360 | no | 450 | 136 | 0.7703 |
| industry_valid_best_f1_by_segment | industry_macro_category=other | 0.360 | yes | 11 | 0 | 0.0000 |
| industry_valid_best_f1_by_segment | industry_macro_category=transport_storage | 0.360 | yes | 11 | 1 | 1.0000 |
| industry_valid_best_f1_by_segment | industry_macro_category=wholesale_retail | 0.380 | no | 41 | 12 | 0.9600 |
| kosdaq_conservative_recall_0.80 | market=KOSDAQ | 0.375 | no | 403 | 143 | 0.7632 |
| kosdaq_conservative_recall_0.85 | market=KOSDAQ | 0.315 | no | 403 | 143 | 0.7381 |
| market_valid_best_f1_by_segment | market=KOSDAQ | 0.360 | no | 403 | 143 | 0.7687 |
| market_valid_best_f1_by_segment | market=KOSPI | 0.370 | no | 273 | 33 | 0.7812 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=it_services | 0.380 | no | 129 | 23 | 0.7037 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=manufacturing | 0.360 | no | 450 | 136 | 0.7703 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | market=KOSDAQ | 0.375 | no | 403 | 143 | 0.7632 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=it_services | 0.305 | no | 129 | 23 | 0.6667 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=manufacturing | 0.305 | yes | 450 | 136 | 0.7308 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | market=KOSDAQ | 0.315 | no | 403 | 143 | 0.7381 |

## 4. FP 집중 구간 변화

아래 표는 현재 artifact threshold 대비 KOSDAQ, IT서비스, 제조업의 FP/FN이
어떻게 바뀌는지 보여줍니다.

| Policy | Segment | Precision | Recall | F1 | FP | FP Δ | FN | FN Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_artifact_threshold | market=KOSDAQ | 0.6044 | 0.8344 | 0.7010 | 89 | +0 | 27 | +0 |
| current_artifact_threshold | industry_macro_category=it_services | 0.4706 | 0.7619 | 0.5818 | 18 | +0 | 5 | +0 |
| current_artifact_threshold | industry_macro_category=manufacturing | 0.6326 | 0.8395 | 0.7215 | 79 | +0 | 26 | +0 |
| market_valid_best_f1_by_segment | market=KOSDAQ | 0.6443 | 0.7669 | 0.7003 | 69 | -20 | 38 | +11 |
| market_valid_best_f1_by_segment | industry_macro_category=it_services | 0.5000 | 0.7619 | 0.6038 | 16 | -2 | 5 | +0 |
| market_valid_best_f1_by_segment | industry_macro_category=manufacturing | 0.6739 | 0.7654 | 0.7168 | 60 | -19 | 38 | +12 |
| industry_valid_best_f1_by_segment | market=KOSDAQ | 0.6443 | 0.7669 | 0.7003 | 69 | -20 | 38 | +11 |
| industry_valid_best_f1_by_segment | industry_macro_category=it_services | 0.5000 | 0.7619 | 0.6038 | 16 | -2 | 5 | +0 |
| industry_valid_best_f1_by_segment | industry_macro_category=manufacturing | 0.6703 | 0.7654 | 0.7147 | 61 | -18 | 38 | +12 |
| kosdaq_conservative_recall_0.85 | market=KOSDAQ | 0.6298 | 0.8037 | 0.7062 | 77 | -12 | 32 | +5 |
| kosdaq_conservative_recall_0.85 | industry_macro_category=it_services | 0.5000 | 0.7619 | 0.6038 | 16 | -2 | 5 | +0 |
| kosdaq_conservative_recall_0.85 | industry_macro_category=manufacturing | 0.6550 | 0.8086 | 0.7238 | 69 | -10 | 31 | +5 |
| kosdaq_conservative_recall_0.80 | market=KOSDAQ | 0.6510 | 0.7669 | 0.7042 | 67 | -22 | 38 | +11 |
| kosdaq_conservative_recall_0.80 | industry_macro_category=it_services | 0.5000 | 0.7619 | 0.6038 | 16 | -2 | 5 | +0 |
| kosdaq_conservative_recall_0.80 | industry_macro_category=manufacturing | 0.6720 | 0.7716 | 0.7184 | 61 | -18 | 37 | +11 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | market=KOSDAQ | 0.6298 | 0.8037 | 0.7062 | 77 | -12 | 32 | +5 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=it_services | 0.5000 | 0.7619 | 0.6038 | 16 | -2 | 5 | +0 |
| targeted_kosdaq_it_mfg_conservative_recall_0.85 | industry_macro_category=manufacturing | 0.6550 | 0.8086 | 0.7238 | 69 | -10 | 31 | +5 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | market=KOSDAQ | 0.6510 | 0.7669 | 0.7042 | 67 | -22 | 38 | +11 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=it_services | 0.5000 | 0.7619 | 0.6038 | 16 | -2 | 5 | +0 |
| targeted_kosdaq_it_mfg_conservative_recall_0.80 | industry_macro_category=manufacturing | 0.6776 | 0.7654 | 0.7188 | 59 | -20 | 38 | +12 |

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
