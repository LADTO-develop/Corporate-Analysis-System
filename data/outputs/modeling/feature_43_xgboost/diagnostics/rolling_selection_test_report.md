# Rolling-Selected Candidate Test Experiments

전체 단일 후보 변수를 rolling OOT validation으로 평가하고, rolling 상위 단일 후보의 2개 조합까지 비교한 뒤 final test 성능을 확인했습니다.
final test는 후보 선택에 사용하지 않고, rolling 기준으로 고른 후보가 마지막 구간에서 어떤지 확인하는 용도로만 사용합니다.

## 1. 결론

- Baseline rolling mean F1/PR-AUC: `0.7022` / `0.7955`
- Baseline final test F1/PR-AUC: `0.7347` / `0.7744`
- Rolling F1 기준 최상위: `pair_rolling_pool__interest_burden_ratio__is_operating_income_turn_negative` (rolling F1 `0.7164`, final test F1 `0.6952`)
- Rolling PR-AUC 기준 최상위: `pair_rolling_pool__interest_burden_ratio__capital_impairment_diff` (rolling PR-AUC `0.8050`, final test PR-AUC `0.7754`)
- Rolling 기준으로 좋아지는 후보는 있지만 final test F1까지 동시에 좋아지는 후보는 없습니다. feature 반영은 보류하고 threshold 정책/추가 OOT 검증을 먼저 보는 편이 안전합니다.

## 2. Rolling F1 기준 상위 후보와 Final Test 확인

| Variant | Stage | Features | Roll PR | Roll P | Roll R | Roll F1 | Roll ΔF1 | Test PR | Test P | Test R | Test F1 | Test ΔF1 | Test ΔFN | Test ΔFP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pair_rolling_pool__interest_burden_ratio__is_operating_income_turn_negative | pair_rolling_pool | interest_burden_ratio, is_operating_income_turn_negative | 0.7994 | 0.6275 | 0.8435 | 0.7164 | 0.0142 | 0.7650 | 0.6000 | 0.8263 | 0.6952 | -0.0395 | 6 | 11 |
| pair_rolling_pool__interest_burden_ratio__ap_days_diff | pair_rolling_pool | interest_burden_ratio, ap_days_diff | 0.8038 | 0.6206 | 0.8558 | 0.7161 | 0.0139 | 0.7720 | 0.6425 | 0.7964 | 0.7112 | -0.0235 | 11 | -7 |
| pair_rolling_pool__interest_burden_ratio__is_3y_consecutive_operating_loss | pair_rolling_pool | interest_burden_ratio, is_3y_consecutive_operating_loss | 0.8006 | 0.6173 | 0.8542 | 0.7146 | 0.0124 | 0.7684 | 0.6261 | 0.8323 | 0.7147 | -0.0200 | 5 | 2 |
| pair_rolling_pool__interest_burden_ratio__equity_growth | pair_rolling_pool | interest_burden_ratio, equity_growth | 0.8011 | 0.6230 | 0.8469 | 0.7145 | 0.0123 | 0.7761 | 0.6228 | 0.8503 | 0.7190 | -0.0157 | 2 | 5 |
| single__rolling_3y_cv_operating_margin | single | rolling_3y_cv_operating_margin | 0.7917 | 0.6156 | 0.8583 | 0.7139 | 0.0118 | 0.7676 | 0.6061 | 0.8383 | 0.7035 | -0.0312 | 4 | 10 |
| pair_rolling_pool__ar_days__capital_impairment_diff | pair_rolling_pool | ar_days, capital_impairment_diff | 0.7963 | 0.6232 | 0.8415 | 0.7124 | 0.0102 | 0.7646 | 0.5887 | 0.8743 | 0.7036 | -0.0311 | -2 | 21 |
| pair_rolling_pool__interest_burden_ratio__ocf_deficit_flag | pair_rolling_pool | interest_burden_ratio, ocf_deficit_flag | 0.7983 | 0.6162 | 0.8500 | 0.7124 | 0.0102 | 0.7650 | 0.6000 | 0.8263 | 0.6952 | -0.0395 | 6 | 11 |
| pair_rolling_pool__interest_burden_ratio__is_3y_consecutive_ocf_deficit | pair_rolling_pool | interest_burden_ratio, is_3y_consecutive_ocf_deficit | 0.7983 | 0.6162 | 0.8500 | 0.7124 | 0.0102 | 0.7650 | 0.6000 | 0.8263 | 0.6952 | -0.0395 | 6 | 11 |
| pair_rolling_pool__interest_burden_ratio__negative_equity_flag | pair_rolling_pool | interest_burden_ratio, negative_equity_flag | 0.7983 | 0.6162 | 0.8500 | 0.7124 | 0.0102 | 0.7650 | 0.6000 | 0.8263 | 0.6952 | -0.0395 | 6 | 11 |
| pair_rolling_pool__interest_burden_ratio__is_negative_equity_entry | pair_rolling_pool | interest_burden_ratio, is_negative_equity_entry | 0.7983 | 0.6162 | 0.8500 | 0.7124 | 0.0102 | 0.7650 | 0.6000 | 0.8263 | 0.6952 | -0.0395 | 6 | 11 |
| pair_rolling_pool__interest_burden_ratio__ocf_to_total_borrowings_diff | pair_rolling_pool | interest_burden_ratio, ocf_to_total_borrowings_diff | 0.8006 | 0.6214 | 0.8418 | 0.7121 | 0.0100 | 0.7603 | 0.5847 | 0.8683 | 0.6988 | -0.0359 | -1 | 22 |
| pair_rolling_pool__is_operating_income_turn_negative__ar_days | pair_rolling_pool | is_operating_income_turn_negative, ar_days | 0.7945 | 0.6215 | 0.8407 | 0.7120 | 0.0099 | 0.7616 | 0.6157 | 0.8443 | 0.7121 | -0.0226 | 3 | 7 |
| pair_rolling_pool__interest_burden_ratio__ebitda_margin | pair_rolling_pool | interest_burden_ratio, ebitda_margin | 0.8010 | 0.6142 | 0.8579 | 0.7120 | 0.0099 | 0.7766 | 0.6261 | 0.8623 | 0.7254 | -0.0093 | 0 | 5 |
| pair_rolling_pool__interest_burden_ratio__delta_accruals_ratio | pair_rolling_pool | interest_burden_ratio, delta_accruals_ratio | 0.8032 | 0.6160 | 0.8528 | 0.7118 | 0.0096 | 0.7676 | 0.6188 | 0.8263 | 0.7077 | -0.0270 | 6 | 4 |
| pair_rolling_pool__rolling_3y_cv_operating_margin__market_spread | pair_rolling_pool | rolling_3y_cv_operating_margin, market_spread | 0.7909 | 0.6123 | 0.8534 | 0.7111 | 0.0090 | 0.7710 | 0.6144 | 0.8683 | 0.7196 | -0.0151 | -1 | 10 |
| pair_rolling_pool__interest_burden_ratio__market_spread | pair_rolling_pool | interest_burden_ratio, market_spread | 0.7990 | 0.6164 | 0.8477 | 0.7108 | 0.0086 | 0.7660 | 0.6167 | 0.8383 | 0.7107 | -0.0240 | 4 | 6 |
| pair_rolling_pool__interest_burden_ratio__treasury_3y_diff | pair_rolling_pool | interest_burden_ratio, treasury_3y_diff | 0.7967 | 0.6131 | 0.8515 | 0.7105 | 0.0084 | 0.7728 | 0.6227 | 0.8204 | 0.7080 | -0.0267 | 7 | 2 |
| pair_rolling_pool__gross_margin__treasury_3y_diff | pair_rolling_pool | gross_margin, treasury_3y_diff | 0.7958 | 0.6139 | 0.8462 | 0.7098 | 0.0077 | 0.7719 | 0.6356 | 0.8563 | 0.7296 | -0.0051 | 1 | 1 |
| pair_rolling_pool__ebitda_margin__treasury_3y_diff | pair_rolling_pool | ebitda_margin, treasury_3y_diff | 0.7938 | 0.6092 | 0.8555 | 0.7096 | 0.0074 | 0.7777 | 0.6109 | 0.8743 | 0.7192 | -0.0155 | -2 | 12 |
| pair_rolling_pool__interest_burden_ratio__gross_margin | pair_rolling_pool | interest_burden_ratio, gross_margin | 0.8002 | 0.6162 | 0.8437 | 0.7093 | 0.0071 | 0.7666 | 0.6321 | 0.8024 | 0.7071 | -0.0276 | 10 | -3 |

## 3. Rolling PR-AUC 기준 상위 후보

| Variant | Stage | Features | Roll PR | Roll F1 | Test PR | Test ROC | Test P | Test R | Test F1 | Test ΔPR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pair_rolling_pool__interest_burden_ratio__capital_impairment_diff | pair_rolling_pool | interest_burden_ratio, capital_impairment_diff | 0.8050 | 0.7055 | 0.7754 | 0.9098 | 0.6393 | 0.8383 | 0.7254 | 0.0011 |
| pair_rolling_pool__interest_burden_ratio__ap_days_diff | pair_rolling_pool | interest_burden_ratio, ap_days_diff | 0.8038 | 0.7161 | 0.7720 | 0.9102 | 0.6425 | 0.7964 | 0.7112 | -0.0023 |
| pair_rolling_pool__interest_burden_ratio__delta_accruals_ratio | pair_rolling_pool | interest_burden_ratio, delta_accruals_ratio | 0.8032 | 0.7118 | 0.7676 | 0.9065 | 0.6188 | 0.8263 | 0.7077 | -0.0068 |
| single__interest_burden_ratio | single | interest_burden_ratio | 0.8015 | 0.7081 | 0.7653 | 0.9083 | 0.6026 | 0.8443 | 0.7032 | -0.0091 |
| pair_rolling_pool__interest_burden_ratio__ar_days | pair_rolling_pool | interest_burden_ratio, ar_days | 0.8015 | 0.7080 | 0.7597 | 0.9070 | 0.6130 | 0.8443 | 0.7103 | -0.0147 |
| pair_rolling_pool__interest_burden_ratio__equity_growth | pair_rolling_pool | interest_burden_ratio, equity_growth | 0.8011 | 0.7145 | 0.7761 | 0.9120 | 0.6228 | 0.8503 | 0.7190 | 0.0017 |
| pair_rolling_pool__interest_burden_ratio__ebitda_margin | pair_rolling_pool | interest_burden_ratio, ebitda_margin | 0.8010 | 0.7120 | 0.7766 | 0.9133 | 0.6261 | 0.8623 | 0.7254 | 0.0022 |
| pair_rolling_pool__gross_margin__non_paid_in_equity_ratio | pair_rolling_pool | gross_margin, non_paid_in_equity_ratio | 0.8007 | 0.7000 | 0.7730 | 0.9067 | 0.6043 | 0.8503 | 0.7065 | -0.0014 |
| pair_rolling_pool__interest_burden_ratio__is_3y_consecutive_operating_loss | pair_rolling_pool | interest_burden_ratio, is_3y_consecutive_operating_loss | 0.8006 | 0.7146 | 0.7684 | 0.9071 | 0.6261 | 0.8323 | 0.7147 | -0.0060 |
| pair_rolling_pool__equity_growth__non_paid_in_equity_ratio | pair_rolling_pool | equity_growth, non_paid_in_equity_ratio | 0.8006 | 0.6996 | 0.7672 | 0.9068 | 0.6170 | 0.8683 | 0.7214 | -0.0072 |
| pair_rolling_pool__interest_burden_ratio__ocf_to_total_borrowings_diff | pair_rolling_pool | interest_burden_ratio, ocf_to_total_borrowings_diff | 0.8006 | 0.7121 | 0.7603 | 0.9050 | 0.5847 | 0.8683 | 0.6988 | -0.0141 |
| pair_rolling_pool__interest_burden_ratio__gross_margin | pair_rolling_pool | interest_burden_ratio, gross_margin | 0.8002 | 0.7093 | 0.7666 | 0.9093 | 0.6321 | 0.8024 | 0.7071 | -0.0078 |
| pair_rolling_pool__equity_growth__ebitda_margin | pair_rolling_pool | equity_growth, ebitda_margin | 0.8002 | 0.6959 | 0.7719 | 0.9086 | 0.5885 | 0.8563 | 0.6976 | -0.0025 |
| single__delta_accruals_ratio | single | delta_accruals_ratio | 0.7999 | 0.7011 | 0.7702 | 0.9044 | 0.5855 | 0.8204 | 0.6833 | -0.0042 |
| pair_rolling_pool__interest_burden_ratio__is_operating_income_turn_negative | pair_rolling_pool | interest_burden_ratio, is_operating_income_turn_negative | 0.7994 | 0.7164 | 0.7650 | 0.9059 | 0.6000 | 0.8263 | 0.6952 | -0.0094 |
| pair_rolling_pool__interest_burden_ratio__non_paid_in_equity_ratio | pair_rolling_pool | interest_burden_ratio, non_paid_in_equity_ratio | 0.7994 | 0.7037 | 0.7678 | 0.9069 | 0.6025 | 0.8623 | 0.7094 | -0.0066 |
| single__gross_margin | single | gross_margin | 0.7994 | 0.7074 | 0.7724 | 0.9085 | 0.5917 | 0.8503 | 0.6978 | -0.0020 |
| pair_rolling_pool__delta_accruals_ratio__capital_impairment_diff | pair_rolling_pool | delta_accruals_ratio, capital_impairment_diff | 0.7992 | 0.7078 | 0.7782 | 0.9098 | 0.5934 | 0.8563 | 0.7010 | 0.0038 |
| pair_rolling_pool__ap_days_diff__ocf_to_total_borrowings_diff | pair_rolling_pool | ap_days_diff, ocf_to_total_borrowings_diff | 0.7992 | 0.6953 | 0.7681 | 0.9057 | 0.6025 | 0.8623 | 0.7094 | -0.0063 |
| pair_rolling_pool__ar_days__is_3y_consecutive_operating_loss | pair_rolling_pool | ar_days, is_3y_consecutive_operating_loss | 0.7990 | 0.7010 | 0.7650 | 0.9075 | 0.6026 | 0.8443 | 0.7032 | -0.0094 |

## 4. Rolling과 Final Test가 같이 좋아진 후보

해당 후보가 없습니다.

## 5. 해석 기준

- 변수 선택은 rolling validation 기준으로만 판단합니다.
- threshold는 별도 정책으로 조정 가능하므로 PR-AUC/ROC-AUC 같은 ranking 지표도 함께 봅니다.
- 다만 최종 서비스에 표시되는 Precision/Recall/F1은 threshold 정책의 영향을 받으므로, 후보 모델 확정 전 threshold 재탐색이 필요합니다.
- rolling fold rows: `1036`