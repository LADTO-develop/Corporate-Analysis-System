# Feature 43 Variable Improvement Experiments

이 리포트는 43-feature XGBoost 기준에서 시장 더미 축소, 절대금액 변수 변환, 산업 내 백분위 변수, 결측 대체 전략을 비교한 실험입니다.
모든 실험은 동일한 train/valid/test split과 동일한 XGBoost 레시피를 사용하고, validation 기준 Platt scaling과 F1 threshold tuning을 적용했습니다.

## 1. 변수 개선 실험 요약

- 가장 높은 F1 변형: `drop_market_kospi` (F1 `0.7097`, PR-AUC `0.7685`)
- F1 기준 최상위는 `drop_market_kospi`이고 baseline_43 대비 `+0.0018` 차이입니다.
- 변수 변경은 성능 차이가 작아 production 반영 전 별도 모델 선택 합의가 필요합니다.

| Variant | Features | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier | Logloss | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drop_market_kospi | 42 | 0.7685 | 0.9052 | 0.6439 | 0.7904 | 0.7097 | 0.1047 | 0.3293 | market_KOSDAQ만 유지 |
| drop_market_kosdaq | 42 | 0.7685 | 0.9052 | 0.6439 | 0.7904 | 0.7097 | 0.1047 | 0.3293 | market_KOSPI만 유지 |
| industry_pct_add_amounts | 46 | 0.7648 | 0.9031 | 0.6267 | 0.8144 | 0.7083 | 0.1054 | 0.3311 | 절대금액 raw 유지 + fiscal_year+industry 내부 백분위 추가 |
| baseline_43 | 43 | 0.7666 | 0.9031 | 0.6667 | 0.7545 | 0.7079 | 0.1052 | 0.3323 |  |
| log_amounts_replace | 43 | 0.7664 | 0.9039 | 0.6377 | 0.7904 | 0.7059 | 0.1053 | 0.3312 | 절대금액 3개를 signed log1p로 대체 |
| drop_market_log_add | 45 | 0.7756 | 0.9079 | 0.6562 | 0.7545 | 0.7019 | 0.1031 | 0.3246 | market_KOSDAQ만 유지 + log 변수 추가 |
| industry_pct_replace_amounts | 43 | 0.7645 | 0.9042 | 0.6286 | 0.7904 | 0.7003 | 0.1069 | 0.3334 | 절대금액 3개를 fiscal_year+industry 내부 백분위로 대체 |
| log_amounts_add | 46 | 0.7689 | 0.9045 | 0.6561 | 0.7425 | 0.6966 | 0.1047 | 0.3279 | 절대금액 raw 유지 + log 변수 추가 |
| drop_market_pct_add | 45 | 0.7738 | 0.9071 | 0.5660 | 0.8982 | 0.6944 | 0.1039 | 0.3253 | market_KOSDAQ만 유지 + 산업 백분위 추가 |

## 2. 결측값 대체 실험 요약

- 가장 높은 F1 결측 전략: `xgboost_native_missing` (F1 `0.7160`)
- F1 기준 최상위 결측 전략은 `xgboost_native_missing`이고 중앙값 대체 대비 `+0.0082` 차이입니다.
- missing indicator 추가와 시장+산업별 중앙값 대체는 현재 split에서 신중하게 봐야 합니다.

| Variant | Features | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier | Logloss | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost_native_missing | 43 | 0.7744 | 0.9110 | 0.6092 | 0.8683 | 0.7160 | 0.1018 | 0.3211 | XGBoost가 NaN 방향을 직접 학습 |
| median_imputation | 43 | 0.7666 | 0.9031 | 0.6667 | 0.7545 | 0.7079 | 0.1052 | 0.3323 | 비교 기준: train 중앙값으로 결측 대체 |
| median_plus_missing_indicators | 63 | 0.7783 | 0.9090 | 0.6546 | 0.7605 | 0.7036 | 0.1026 | 0.3237 | 중앙값 대체 + 결측 여부 indicator 20개 추가 |
| market_industry_median_imputation | 43 | 0.7780 | 0.9066 | 0.6079 | 0.8263 | 0.7005 | 0.1028 | 0.3243 | train 기준 시장+산업별 중앙값 대체 |

## 3. 결측률 점검

결측률이 높은 변수는 차입금·현금흐름 관련 변수와 전년 대비 변화 변수입니다. 결측 여부 자체의 양성 비율 차이는 크지 않아, 단순 indicator 추가 효과는 제한적이었습니다.

| Feature | Train Missing | Valid Missing | Test Missing | Missing Rows | Label Rate Missing | Label Rate Observed |
| --- | --- | --- | --- | --- | --- | --- |
| ocf_to_total_borrowings | 21.1% | 21.4% | 20.5% | 812 | 21.4% | 23.2% |
| short_term_borrowings_share | 21.1% | 21.4% | 20.5% | 812 | 21.4% | 23.2% |
| net_margin_diff | 19.3% | 16.4% | 17.0% | 744 | 24.5% | 22.4% |
| total_assets_growth | 18.4% | 15.8% | 15.3% | 709 | 23.4% | 22.7% |
| market_to_book | 14.4% | 12.9% | 12.8% | 556 | 23.7% | 22.6% |
| operating_roa | 14.4% | 12.9% | 12.8% | 555 | 23.6% | 22.7% |
| current_ratio | 14.4% | 12.9% | 12.8% | 555 | 23.6% | 22.7% |
| intangible_assets_ratio | 14.4% | 12.9% | 12.8% | 555 | 23.6% | 22.7% |
| accruals_ratio | 14.4% | 12.9% | 12.8% | 555 | 23.6% | 22.7% |
| ocf_to_sales | 14.4% | 12.9% | 12.8% | 555 | 23.6% | 22.7% |
| ocf_to_total_liabilities | 14.4% | 12.9% | 12.8% | 555 | 23.6% | 22.7% |
| pretax_roe | 14.4% | 12.9% | 12.8% | 555 | 23.6% | 22.7% |

## 4. 판단

- `market_to_book` 원본 값 복구 후에는 성능 순위가 이전 all-zero 기준과 달라질 수 있습니다.
- 현재 production artifact는 43개 변수와 XGBoost native missing 기준입니다.
- native missing은 중앙값 대체보다 Recall과 F1이 높아 조기경보 목적에 더 잘 맞습니다.
- 변수셋 축소나 결측 전략 변경은 성능 차이가 작으므로 발표/운영 기준을 먼저 합의하는 편이 안전합니다.
