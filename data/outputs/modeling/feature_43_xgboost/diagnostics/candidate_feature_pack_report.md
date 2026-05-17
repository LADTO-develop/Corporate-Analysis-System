# Candidate Feature-Pack Experiments

원본 Model V1에는 존재하지만 현재 43-feature 입력에는 빠져 있는 후보 변수를 묶음별로 추가해 비교한 실험입니다.
모든 실험은 XGBoost native missing, Platt scaling, validation 기준 `recall >= 0.85` 조건에서 precision 최대 threshold를 사용했습니다.

## 1. 결론

- Baseline valid/test F1: `0.7494` / `0.7347`
- Validation 기준 선택 후보: `combined_interpretable_add_native` (valid F1 `0.7612`, test F1 `0.7125`)
- Validation 선택 후보의 baseline 대비 변화: valid F1 `+0.0118`, test F1 `-0.0222`
- 참고용 test F1 최상위 후보: `baseline_43_native` (test F1 `0.7347`, baseline 대비 `+0.0000`)
- `combined_interpretable_add_native`는 validation에서는 좋아졌지만 test에서는 악화되었습니다. 과적합 가능성이 있어 production 반영은 보류하는 편이 좋습니다.

## 2. Validation 기준 성능 비교

| Variant | Added | Threshold | Valid PR | Valid P | Valid R | Valid F1 | Test PR | Test P | Test R | Test F1 | Test FP | Test FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combined_interpretable_add_native | 31 | 0.3250 | 0.8222 | 0.6770 | 0.8693 | 0.7612 | 0.7665 | 0.6042 | 0.8683 | 0.7125 | 95 | 22 |
| top_univariate_add_native | 11 | 0.3150 | 0.8292 | 0.6711 | 0.8580 | 0.7531 | 0.7646 | 0.6205 | 0.8323 | 0.7110 | 85 | 28 |
| profitability_quality_add_native | 7 | 0.3250 | 0.8216 | 0.6696 | 0.8523 | 0.7500 | 0.7775 | 0.6313 | 0.8204 | 0.7135 | 80 | 30 |
| baseline_43_native | 0 | 0.3150 | 0.8156 | 0.6652 | 0.8580 | 0.7494 | 0.7744 | 0.6400 | 0.8623 | 0.7347 | 81 | 23 |
| working_capital_quality_add_native | 10 | 0.2800 | 0.8239 | 0.6567 | 0.8693 | 0.7482 | 0.7741 | 0.5890 | 0.8323 | 0.6898 | 97 | 28 |
| cashflow_quality_add_native | 6 | 0.3200 | 0.8065 | 0.6623 | 0.8580 | 0.7475 | 0.7662 | 0.6140 | 0.8383 | 0.7089 | 88 | 27 |
| macro_delta_add_native | 5 | 0.3050 | 0.8192 | 0.6537 | 0.8580 | 0.7420 | 0.7639 | 0.6157 | 0.8443 | 0.7121 | 88 | 26 |
| audit_flag_add_native | 1 | 0.3100 | 0.8126 | 0.6550 | 0.8523 | 0.7407 | 0.7742 | 0.6043 | 0.8503 | 0.7065 | 93 | 25 |
| distress_flags_add_native | 8 | 0.3150 | 0.8079 | 0.6550 | 0.8523 | 0.7407 | 0.7702 | 0.6096 | 0.8323 | 0.7038 | 89 | 28 |
| combined_all_candidate_add_native | 44 | 0.3150 | 0.8272 | 0.6468 | 0.8636 | 0.7397 | 0.7680 | 0.5940 | 0.8323 | 0.6933 | 95 | 28 |
| macro_context_add_native | 12 | 0.2950 | 0.8150 | 0.6441 | 0.8636 | 0.7379 | 0.7682 | 0.6008 | 0.8563 | 0.7062 | 95 | 24 |

## 3. 참고용 Test 기준 상위 후보

아래 표는 사후 점검용이며, 모델 선택 기준으로는 사용하지 않습니다.

| Variant | Added | Valid F1 | Test PR | Test P | Test R | Test F1 | Test FP | Test FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_43_native | 0 | 0.7494 | 0.7744 | 0.6400 | 0.8623 | 0.7347 | 81 | 23 |
| profitability_quality_add_native | 7 | 0.7500 | 0.7775 | 0.6313 | 0.8204 | 0.7135 | 80 | 30 |
| combined_interpretable_add_native | 31 | 0.7612 | 0.7665 | 0.6042 | 0.8683 | 0.7125 | 95 | 22 |
| macro_delta_add_native | 5 | 0.7420 | 0.7639 | 0.6157 | 0.8443 | 0.7121 | 88 | 26 |
| top_univariate_add_native | 11 | 0.7531 | 0.7646 | 0.6205 | 0.8323 | 0.7110 | 85 | 28 |
| cashflow_quality_add_native | 6 | 0.7475 | 0.7662 | 0.6140 | 0.8383 | 0.7089 | 88 | 27 |
| audit_flag_add_native | 1 | 0.7407 | 0.7742 | 0.6043 | 0.8503 | 0.7065 | 93 | 25 |
| macro_context_add_native | 12 | 0.7379 | 0.7682 | 0.6008 | 0.8563 | 0.7062 | 95 | 24 |

## 4. KOSDAQ 오류 관점

- Baseline KOSDAQ FP/FN: `70` / `19`
- Validation 선택 후보 KOSDAQ FP/FN: `85` / `17`

## 5. 후보 변수 묶음

| Variant | Note | Features |
| --- | --- | --- |
| audit_flag_add_native | 감사의견 관련 플래그 후보 추가 | audit_qualified_flag |
| cashflow_quality_add_native | 영업현금흐름의 질과 변화량 후보 추가 | ocf_to_total_assets<br>ocf_deficit_flag<br>delta_accruals_ratio<br>ocf_to_total_liabilities_diff<br>ocf_to_total_borrowings_diff<br>rolling_3y_cv_ocf_to_total_borrowings |
| combined_all_candidate_add_native | 전체 후보 변수팩 통합 | roe<br>operating_roe<br>ebitda_margin<br>interest_burden_ratio<br>gross_margin<br>operating_margin_diff<br>ebitda_margin_diff<br>ocf_to_total_assets<br>ocf_deficit_flag<br>delta_accruals_ratio<br>ocf_to_total_liabilities_diff<br>ocf_to_total_borrowings_diff<br>rolling_3y_cv_ocf_to_total_borrowings<br>is_zombie_3y<br>is_3y_consecutive_operating_loss<br>is_3y_consecutive_ocf_deficit<br>is_operating_income_turn_negative<br>is_ocf_turn_negative<br>negative_equity_flag<br>is_negative_equity_entry<br>is_current_ratio_below_1<br>ar_days<br>inventory_days<br>ap_days<br>ar_days_diff<br>inventory_days_diff<br>ap_days_diff<br>accounts_receivable_ratio<br>inventory_ratio<br>contract_assets_ratio<br>advances_from_customers_ratio<br>base_rate<br>treasury_3y<br>corp_aa_3y<br>corp_bbb_3y<br>market_spread<br>usd_krw<br>ppi<br>base_rate_diff<br>treasury_3y_diff<br>usd_krw_diff<br>market_spread_diff<br>spec_spread_diff<br>audit_qualified_flag |
| combined_interpretable_add_native | 수익성, 현금흐름, 부실 플래그, 운전자본 후보 통합 | roe<br>operating_roe<br>ebitda_margin<br>interest_burden_ratio<br>gross_margin<br>operating_margin_diff<br>ebitda_margin_diff<br>ocf_to_total_assets<br>ocf_deficit_flag<br>delta_accruals_ratio<br>ocf_to_total_liabilities_diff<br>ocf_to_total_borrowings_diff<br>rolling_3y_cv_ocf_to_total_borrowings<br>is_zombie_3y<br>is_3y_consecutive_operating_loss<br>is_3y_consecutive_ocf_deficit<br>is_operating_income_turn_negative<br>is_ocf_turn_negative<br>negative_equity_flag<br>is_negative_equity_entry<br>is_current_ratio_below_1<br>ar_days<br>inventory_days<br>ap_days<br>ar_days_diff<br>inventory_days_diff<br>ap_days_diff<br>accounts_receivable_ratio<br>inventory_ratio<br>contract_assets_ratio<br>advances_from_customers_ratio |
| distress_flags_add_native | 좀비/연속 적자/현금흐름 악화 등 해석 가능한 부실 징후 플래그 추가 | is_zombie_3y<br>is_3y_consecutive_operating_loss<br>is_3y_consecutive_ocf_deficit<br>is_operating_income_turn_negative<br>is_ocf_turn_negative<br>negative_equity_flag<br>is_negative_equity_entry<br>is_current_ratio_below_1 |
| macro_context_add_native | 거시 레벨 지표와 변화량 후보 추가 | base_rate<br>treasury_3y<br>corp_aa_3y<br>corp_bbb_3y<br>market_spread<br>usd_krw<br>ppi<br>base_rate_diff<br>treasury_3y_diff<br>usd_krw_diff<br>market_spread_diff<br>spec_spread_diff |
| macro_delta_add_native | 금리/환율/시장 스프레드 변화량 후보 추가 | base_rate_diff<br>treasury_3y_diff<br>usd_krw_diff<br>market_spread_diff<br>spec_spread_diff |
| profitability_quality_add_native | ROE, EBITDA margin, 이자부담률 등 수익성/상환능력 후보 추가 | roe<br>operating_roe<br>ebitda_margin<br>interest_burden_ratio<br>gross_margin<br>operating_margin_diff<br>ebitda_margin_diff |
| top_univariate_add_native | 단변량 선별에서 상대적으로 강했던 후보 묶음 추가 | roe<br>ebitda_margin<br>interest_burden_ratio<br>operating_roe<br>ocf_to_total_assets<br>is_zombie_3y<br>rolling_3y_cv_operating_margin<br>ar_days<br>capital_impairment_diff<br>equity_growth<br>non_paid_in_equity_ratio |
| working_capital_quality_add_native | 매출채권/재고/매입채무 회전일수와 운전자본 비율 후보 추가 | ar_days<br>inventory_days<br>ap_days<br>ar_days_diff<br>inventory_days_diff<br>ap_days_diff<br>accounts_receivable_ratio<br>inventory_ratio<br>contract_assets_ratio<br>advances_from_customers_ratio |

## 6. 해석 원칙

- 실제 모델 선택은 validation 성능만 기준으로 합니다.
- test 기준 최상위 후보는 사후 참고용이며, 운영 반영 전 추가 OOT 검증이 필요합니다.
- 절대금액 원값은 기존 오류 분석에서 FP를 키울 수 있어 이번 실험에서는 해석 가능한 비율/플래그/변화량 후보를 우선했습니다.