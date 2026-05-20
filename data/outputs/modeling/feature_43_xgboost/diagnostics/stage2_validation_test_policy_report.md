# Stage 2 Validation/Test Policy Evaluation

## 원칙

- Stage 2 보류/검토 정책은 validation 기준으로만 비교합니다.
- test는 validation에서 고른 후보가 유지되는지 확인하는 holdout 용도입니다.
- 2026 신용평가 공시 라벨은 외부검증셋이므로 이 리포트의 선택 과정에는 사용하지 않습니다.
- `보류`는 최종 부적격 확정이 아니라 추가 검토 대상으로 해석합니다.

## Validation 기준 선택 결과

- 공식 1차 모델 기준: `stage1_model` (valid precision `0.6652`, recall `0.8580`, F1 `0.7494`, FP `76`, FN `25`)
- Validation F1 최대 후보: `stage1_minus_overwarning_candidate` (valid precision `0.7000`, recall `0.8352`, F1 `0.7617`, FP `63`, FN `29`)
- Recall 0.88 이상 중 precision 최대 후보: `stage1_or_45_no_it_low_threshold` (valid precision `0.6568`, recall `0.8807`, F1 `0.7524`, FP `81`, FN `21`)
- 현재 보조 review trigger: `stage1_or_45_or_it_low_threshold` (valid precision `0.6434`, recall `0.8920`, F1 `0.7476`, FP `87`, FN `19`)

## Policy Metrics

| Split | Policy | Precision | Recall | F1 | TP | FP | FN | TN | Count | ΔFP | ΔFN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | stage1_minus_overwarning_candidate | 0.7000 | 0.8352 | 0.7617 | 147 | 63 | 29 | 437 | 210 | -13 | 4 |
| valid | stage1_or_45_high_margin | 0.6667 | 0.8636 | 0.7525 | 152 | 76 | 24 | 424 | 228 | 0 | -1 |
| valid | stage1_or_45_no_it_low_threshold | 0.6568 | 0.8807 | 0.7524 | 155 | 81 | 21 | 419 | 236 | 5 | -4 |
| valid | stage1_or_45 | 0.6540 | 0.8807 | 0.7506 | 155 | 82 | 21 | 418 | 237 | 6 | -4 |
| valid | stage1_model | 0.6652 | 0.8580 | 0.7494 | 151 | 76 | 25 | 424 | 227 | 0 | 0 |
| valid | stage1_or_45_or_it_low_threshold | 0.6434 | 0.8920 | 0.7476 | 157 | 87 | 19 | 413 | 244 | 11 | -6 |
| valid | stage1_or_it_low_threshold | 0.6511 | 0.8693 | 0.7445 | 153 | 82 | 23 | 418 | 235 | 6 | -2 |
| valid | current_committee_reject_only | 0.8491 | 0.5114 | 0.6383 | 90 | 16 | 86 | 484 | 106 | -60 | 61 |
| valid | current_committee_hold_or_reject | 0.3413 | 0.9773 | 0.5059 | 172 | 332 | 4 | 168 | 504 | 256 | -21 |
| test | stage1_model | 0.6603 | 0.8522 | 0.7441 | 173 | 89 | 30 | 632 | 262 | 0 | 0 |
| test | stage1_or_45_high_margin | 0.6578 | 0.8522 | 0.7425 | 173 | 90 | 30 | 631 | 263 | 1 | 0 |
| test | stage1_or_45 | 0.6460 | 0.8719 | 0.7421 | 177 | 97 | 26 | 624 | 274 | 8 | -4 |
| test | stage1_or_45_no_it_low_threshold | 0.6471 | 0.8670 | 0.7411 | 176 | 96 | 27 | 625 | 272 | 7 | -3 |
| test | stage1_or_it_low_threshold | 0.6434 | 0.8621 | 0.7368 | 175 | 97 | 28 | 624 | 272 | 8 | -2 |
| test | stage1_or_45_or_it_low_threshold | 0.6312 | 0.8768 | 0.7340 | 178 | 104 | 25 | 617 | 282 | 15 | -5 |
| test | stage1_minus_overwarning_candidate | 0.6793 | 0.7931 | 0.7318 | 161 | 76 | 42 | 645 | 237 | -13 | 12 |
| test | current_committee_reject_only | 0.8276 | 0.4729 | 0.6019 | 96 | 20 | 107 | 701 | 116 | -69 | 77 |
| test | current_committee_hold_or_reject | 0.2891 | 0.9754 | 0.4459 | 198 | 487 | 5 | 234 | 685 | 398 | -25 |

## 해석

- 현재 deterministic committee의 `보류/부적격` 전체를 위험 판단처럼 사용하면 recall은 높지만 FP와 검토량이 과도합니다.
- 따라서 모델 성능표에는 1차 모델과 validation-selected review trigger를 분리해서 보여주는 편이 안전합니다.
- 2차 위원회는 `부적격 확정기`가 아니라, 1차 모델 경고와 보조 trigger가 잡은 기업을 추가 검토하는 설명/검증 단계로 두는 것이 적절합니다.
- test 결과는 후보 선택에 쓰지 않고, validation에서 고른 정책의 일반화 확인용으로만 기록합니다.

## Segment Diagnostics

아래는 validation-selected 후보와 stage1 기준의 주요 취약 구간을 보기 위한 상세 CSV입니다.

- `stage2_validation_test_segment_metrics.csv`

### Recall-floor 후보의 test FP 상위 세그먼트

| Dimension | Segment | Rows | Positives | Precision | Recall | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| fiscal_year | 2023 | 637 | 161 | 0.6228 | 0.8820 | 86 | 19 |
| market | KOSDAQ | 427 | 163 | 0.6452 | 0.8589 | 77 | 23 |
| industry_macro_category | manufacturing | 598 | 162 | 0.6588 | 0.8580 | 72 | 23 |
| firm_size_group | small_and_medium | 232 | 107 | 0.6303 | 0.9720 | 61 | 3 |
| firm_size_group | mid_sized | 434 | 95 | 0.6961 | 0.7474 | 31 | 24 |
| market | KOSPI | 497 | 40 | 0.6545 | 0.9000 | 19 | 4 |
| industry_macro_category | it_services | 176 | 21 | 0.5294 | 0.8571 | 16 | 3 |
| fiscal_year | 2024 | 287 | 42 | 0.7727 | 0.8095 | 10 | 8 |
| firm_size_group | large | 256 | 1 | 0.2000 | 1.0000 | 4 | 0 |
| industry_macro_category | wholesale_retail | 63 | 14 | 0.7647 | 0.9286 | 4 | 1 |