# Stage 2 Validation/Test Policy Evaluation

## 원칙

- Stage 2 보류/검토 정책은 validation 기준으로만 비교합니다.
- test는 validation에서 고른 후보가 유지되는지 확인하는 holdout 용도입니다.
- 2026 신용평가 공시 라벨은 외부검증셋이므로 이 리포트의 선택 과정에는 사용하지 않습니다.
- `보류`는 최종 부적격 확정이 아니라 추가 검토 대상으로 해석합니다.

## Validation 기준 선택 결과

- 공식 1차 모델 기준: `stage1_model` (valid precision `0.7295`, recall `0.8580`, F1 `0.7885`, FP `56`, FN `25`)
- Validation F1 최대 후보: `stage1_or_45_or_it_low_threshold` (valid precision `0.7264`, recall `0.8750`, F1 `0.7938`, FP `58`, FN `22`)
- 현재 보조 review trigger: `stage1_or_45_or_it_low_threshold` (valid precision `0.7264`, recall `0.8750`, F1 `0.7938`, FP `58`, FN `22`)

## Policy Metrics

| Split | Policy | Precision | Recall | F1 | TP | FP | FN | TN | Count | ΔFP | ΔFN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | stage1_or_45_or_it_low_threshold | 0.7264 | 0.8750 | 0.7938 | 154 | 58 | 22 | 442 | 212 | 2 | -3 |
| valid | stage1_or_45_high_margin | 0.7308 | 0.8636 | 0.7917 | 152 | 56 | 24 | 444 | 208 | 0 | -1 |
| valid | stage1_or_45_no_it_low_threshold | 0.7308 | 0.8636 | 0.7917 | 152 | 56 | 24 | 444 | 208 | 0 | -1 |
| valid | stage1_or_it_low_threshold | 0.7251 | 0.8693 | 0.7907 | 153 | 58 | 23 | 442 | 211 | 2 | -2 |
| valid | stage1_or_45 | 0.7273 | 0.8636 | 0.7896 | 152 | 57 | 24 | 443 | 209 | 1 | -1 |
| valid | stage1_model | 0.7295 | 0.8580 | 0.7885 | 151 | 56 | 25 | 444 | 207 | 0 | 0 |
| valid | stage1_minus_overwarning_candidate | 0.7363 | 0.8409 | 0.7851 | 148 | 53 | 28 | 447 | 201 | -3 | 3 |
| valid | current_committee_hold_or_reject | 0.5880 | 0.8920 | 0.7088 | 157 | 110 | 19 | 390 | 267 | 54 | -6 |
| valid | current_committee_reject_only | 0.9583 | 0.2614 | 0.4107 | 46 | 2 | 130 | 498 | 48 | -54 | 105 |
| test | stage1_model | 0.7004 | 0.8522 | 0.7689 | 173 | 74 | 30 | 647 | 247 | 0 | 0 |
| test | stage1_or_45_high_margin | 0.6948 | 0.8522 | 0.7655 | 173 | 76 | 30 | 645 | 249 | 2 | 0 |
| test | stage1_or_it_low_threshold | 0.6920 | 0.8522 | 0.7638 | 173 | 77 | 30 | 644 | 250 | 3 | 0 |
| test | stage1_minus_overwarning_candidate | 0.7042 | 0.8325 | 0.7630 | 169 | 71 | 34 | 650 | 240 | -3 | 4 |
| test | stage1_or_45 | 0.6797 | 0.8571 | 0.7582 | 174 | 82 | 29 | 639 | 256 | 8 | -1 |
| test | stage1_or_45_no_it_low_threshold | 0.6797 | 0.8571 | 0.7582 | 174 | 82 | 29 | 639 | 256 | 8 | -1 |
| test | stage1_or_45_or_it_low_threshold | 0.6718 | 0.8571 | 0.7532 | 174 | 85 | 29 | 636 | 259 | 11 | -1 |
| test | current_committee_hold_or_reject | 0.4879 | 0.8916 | 0.6307 | 181 | 190 | 22 | 531 | 371 | 116 | -8 |
| test | current_committee_reject_only | 0.9800 | 0.2414 | 0.3874 | 49 | 1 | 154 | 720 | 50 | -73 | 124 |

## 해석

- 현재 deterministic committee의 `보류/부적격` 전체를 위험 판단처럼 사용하면 recall은 높지만 FP와 검토량이 과도합니다.
- 따라서 모델 성능표에는 1차 모델과 validation-selected review trigger를 분리해서 보여주는 편이 안전합니다.
- 2차 위원회는 `부적격 확정기`가 아니라, 1차 모델 경고와 보조 trigger가 잡은 기업을 추가 검토하는 설명/검증 단계로 두는 것이 적절합니다.
- test 결과는 후보 선택에 쓰지 않고, validation에서 고른 정책의 일반화 확인용으로만 기록합니다.

## Segment Diagnostics

아래는 validation-selected 후보와 stage1 기준의 주요 취약 구간을 보기 위한 상세 CSV입니다.

- `stage2_validation_test_segment_metrics.csv`

## Decision Trace Gate Contribution

아래 표는 `decision_trace` 게이트가 1차 모델의 FN/FP를 보완한 사례 수를 집계한 결과입니다.
한 기업에서 여러 게이트가 동시에 켜질 수 있으므로, 게이트별 건수는 서로 배타적이지 않습니다.

| Split | Gate | Triggered | FN escalated | FN share | FP softened | FP share | Effect |
|---|---|---:|---:|---:|---:|---:|---|
| test | 과민경고 완화 점검 | 131 | 0 | 0.0000 | 57 | 0.7703 | fp_softening |
| test | 경계등급 점검 | 17 | 1 | 0.0333 | 9 | 0.1216 | fn_and_fp |
| test | 부적격 확정 게이트 | 727 | 8 | 0.2667 | 0 | 0.0000 | fn_escalation |
| test | 2차 보조 레이더 | 11 | 1 | 0.0333 | 0 | 0.0000 | fn_escalation |
| test | 강제 경고 게이트 | 0 | 0 | 0.0000 | 0 | 0.0000 | none |
| test | 숨은 꼬리위험 점검 | 0 | 0 | 0.0000 | 0 | 0.0000 | none |
| valid | 과민경고 완화 점검 | 106 | 0 | 0.0000 | 42 | 0.7500 | fp_softening |
| valid | 부적격 확정 게이트 | 518 | 6 | 0.2400 | 1 | 0.0179 | fn_and_fp |
| valid | 경계등급 점검 | 12 | 1 | 0.0400 | 4 | 0.0714 | fn_and_fp |
| valid | 2차 보조 레이더 | 4 | 3 | 0.1200 | 0 | 0.0000 | fn_escalation |
| valid | 강제 경고 게이트 | 0 | 0 | 0.0000 | 0 | 0.0000 | none |
| valid | 숨은 꼬리위험 점검 | 0 | 0 | 0.0000 | 0 | 0.0000 | none |
