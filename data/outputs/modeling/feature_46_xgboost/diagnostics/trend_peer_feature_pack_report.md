# 46-Feature Trend + Peer-Relative Feature Pack Experiments

공식 `credit_46_features` 입력에 trend diff와 peer-relative percentile 후보를 추가해
walk-forward rolling OOT 기준으로 비교한 실험입니다.

Rolling 평가연도는 `2019, 2020, 2021, 2022`이고, Final Test는 공식 test split인 2023~2024 구간입니다.
각 fold는 `과거 연도 학습 -> 직전 1년 Platt calibration/threshold 선택 -> 다음 1년 평가` 구조입니다.

## 1. 결론

- Baseline rolling F1/PR-AUC: `0.7589` / `0.8363`
- Rolling 기준 최상위 후보: `baseline_46_native` (rolling F1 `0.7589`, PR-AUC `0.8363`)
- Rolling F1 변화: `+0.0000`
- Final Test F1 변화: `+0.0000`
- Rolling OOT 기준으로도 현재 46-feature baseline이 가장 안정적입니다. 이번 feature pack은 공식 모델에 반영하지 않는 편이 좋습니다.

## 2. 후보별 Rolling + Final Test 비교

| Variant | Added | Roll PR | Roll P | Roll R | Roll F1 | Roll dF1 | Roll FP | Roll FN | Final PR | Final P | Final R | Final F1 | Final dF1 | Final FP | Final FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_46_native | 0 | 0.8363 | 0.6917 | 0.8432 | 0.7589 | 0.0000 | 240 | 99 | 0.8321 | 0.6941 | 0.8719 | 0.7729 | 0.0000 | 78 | 26 |
| trend_diff_pack_add_native | 6 | 0.8331 | 0.6892 | 0.8405 | 0.7561 | -0.0028 | 243 | 101 | 0.8329 | 0.6923 | 0.8424 | 0.7600 | -0.0129 | 76 | 32 |
| peer_ratio_pct_pack_add_native | 4 | 0.8374 | 0.6845 | 0.8466 | 0.7551 | -0.0038 | 251 | 97 | 0.8200 | 0.6862 | 0.8079 | 0.7421 | -0.0308 | 75 | 39 |
| trend_peer_combined_pack_add_native | 10 | 0.8361 | 0.6864 | 0.8332 | 0.7523 | -0.0066 | 243 | 105 | 0.8404 | 0.7049 | 0.8473 | 0.7696 | -0.0034 | 72 | 31 |

## 3. Baseline 연도별 Rolling 성능

| Eval Year | Threshold | PR-AUC | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | 0.2600 | 0.8339 | 0.6776 | 0.8857 | 0.7678 | 59 | 16 |
| 2020 | 0.3550 | 0.8540 | 0.7407 | 0.7947 | 0.7668 | 42 | 31 |
| 2021 | 0.3050 | 0.8217 | 0.6456 | 0.8061 | 0.7170 | 73 | 32 |
| 2022 | 0.2400 | 0.8355 | 0.7027 | 0.8864 | 0.7839 | 66 | 20 |

## 4. 최상위 후보 연도별 Rolling 성능

| Eval Year | Threshold | PR-AUC | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | 0.2600 | 0.8339 | 0.6776 | 0.8857 | 0.7678 | 59 | 16 |
| 2020 | 0.3550 | 0.8540 | 0.7407 | 0.7947 | 0.7668 | 42 | 31 |
| 2021 | 0.3050 | 0.8217 | 0.6456 | 0.8061 | 0.7170 | 73 | 32 |
| 2022 | 0.2400 | 0.8355 | 0.7027 | 0.8864 | 0.7839 | 66 | 20 |

## 5. 참고용 Final Test 순위

| Variant | Threshold | PR-AUC | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_46_native | 0.3000 | 0.8321 | 0.6941 | 0.8719 | 0.7729 | 78 | 26 |
| trend_peer_combined_pack_add_native | 0.3150 | 0.8404 | 0.7049 | 0.8473 | 0.7696 | 72 | 31 |
| trend_diff_pack_add_native | 0.3100 | 0.8329 | 0.6923 | 0.8424 | 0.7600 | 76 | 32 |
| peer_ratio_pct_pack_add_native | 0.3450 | 0.8200 | 0.6862 | 0.8079 | 0.7421 | 75 | 39 |

## 6. Final Test 세그먼트

| Variant | Segment | Rows | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_46_native | it_services | 176 | 0.6538 | 0.8095 | 0.7234 | 9 | 4 |
| baseline_46_native | manufacturing | 598 | 0.7035 | 0.8642 | 0.7756 | 59 | 22 |
| baseline_46_native | KOSDAQ | 427 | 0.7015 | 0.8650 | 0.7747 | 60 | 22 |
| baseline_46_native | KOSPI | 497 | 0.6667 | 0.9000 | 0.7660 | 18 | 4 |
| baseline_46_native | all | 924 | 0.6941 | 0.8719 | 0.7729 | 78 | 26 |
| peer_ratio_pct_pack_add_native | it_services | 176 | 0.6000 | 0.7143 | 0.6522 | 10 | 6 |
| peer_ratio_pct_pack_add_native | manufacturing | 598 | 0.6984 | 0.8148 | 0.7521 | 57 | 30 |
| peer_ratio_pct_pack_add_native | KOSDAQ | 427 | 0.6952 | 0.7975 | 0.7429 | 57 | 33 |
| peer_ratio_pct_pack_add_native | KOSPI | 497 | 0.6538 | 0.8500 | 0.7391 | 18 | 6 |
| peer_ratio_pct_pack_add_native | all | 924 | 0.6862 | 0.8079 | 0.7421 | 75 | 39 |
| trend_diff_pack_add_native | it_services | 176 | 0.6087 | 0.6667 | 0.6364 | 9 | 7 |
| trend_diff_pack_add_native | manufacturing | 598 | 0.7041 | 0.8519 | 0.7709 | 58 | 24 |
| trend_diff_pack_add_native | KOSDAQ | 427 | 0.6939 | 0.8344 | 0.7577 | 60 | 27 |
| trend_diff_pack_add_native | KOSPI | 497 | 0.6863 | 0.8750 | 0.7692 | 16 | 5 |
| trend_diff_pack_add_native | all | 924 | 0.6923 | 0.8424 | 0.7600 | 76 | 32 |
| trend_peer_combined_pack_add_native | it_services | 176 | 0.5926 | 0.7619 | 0.6667 | 11 | 5 |
| trend_peer_combined_pack_add_native | manufacturing | 598 | 0.7225 | 0.8519 | 0.7819 | 53 | 24 |
| trend_peer_combined_pack_add_native | KOSDAQ | 427 | 0.7188 | 0.8466 | 0.7775 | 54 | 25 |
| trend_peer_combined_pack_add_native | KOSPI | 497 | 0.6538 | 0.8500 | 0.7391 | 18 | 6 |
| trend_peer_combined_pack_add_native | all | 924 | 0.7049 | 0.8473 | 0.7696 | 72 | 31 |

## 7. 추가 변수

- Trend diff pack: `interest_coverage_ratio_diff`, `cash_ratio_diff`, `ocf_to_sales_diff`, `operating_roa_diff`, `short_term_borrowings_share_diff`, `total_borrowings_ratio_diff`
- Peer ratio percentile pack: `net_margin_industry_year_pct`, `interest_coverage_ratio_industry_year_pct`, `short_term_borrowings_share_industry_year_pct`, `cashflow_coverage_ratio_industry_year_pct`

## 8. 해석 주의

- 후보 선택은 rolling OOT 평균 기준으로 판단했습니다.
- Final Test는 공식 test split 사후 확인용입니다.
- F1이 좋아도 FN이 늘면 조기경보 모델로는 보수적으로 해석합니다.
- 이 실험은 official artifact를 덮어쓰지 않는 feature-pack screening입니다.