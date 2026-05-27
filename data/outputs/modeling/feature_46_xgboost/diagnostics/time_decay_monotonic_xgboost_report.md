# Feature 46 Time-Decay & Monotonic XGBoost Experiment

## 결론

rolling 기준에서 F1, Recall, FN 조건을 동시에 만족하는 승격 후보는 없습니다. 공식 모델은 현재 46-feature baseline을 유지하는 편이 안전합니다.

## Rolling OOT 요약

| Candidate | Features | Constraints | PR-AUC | Precision | Recall | F1 | F1 Δ | FP | FN | FN Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_current | 46 | 0 | 0.8363 | 0.6917 | 0.8432 | 0.7589 | +0.0000 | 240 | 99 | 0 |
| time_decay_half_life_1y | 46 | 0 | 0.8311 | 0.6774 | 0.8463 | 0.7515 | -0.0074 | 257 | 97 | -2 |
| risk_proxy_monotonic_time_decay_3y | 58 | 12 | 0.8330 | 0.6878 | 0.8322 | 0.7514 | -0.0074 | 241 | 107 | 8 |
| monotonic_guardrail_time_decay_3y | 46 | 9 | 0.8289 | 0.6862 | 0.8324 | 0.7511 | -0.0078 | 242 | 107 | 8 |
| time_decay_half_life_2y | 46 | 0 | 0.8353 | 0.6778 | 0.8444 | 0.7503 | -0.0086 | 256 | 99 | 0 |
| monotonic_directional | 46 | 24 | 0.8153 | 0.6745 | 0.8461 | 0.7489 | -0.0100 | 263 | 97 | -2 |
| time_decay_half_life_3y | 46 | 0 | 0.8329 | 0.6773 | 0.8389 | 0.7485 | -0.0103 | 253 | 102 | 3 |
| monotonic_time_decay_3y | 46 | 24 | 0.8151 | 0.6672 | 0.8577 | 0.7471 | -0.0118 | 279 | 90 | -9 |
| risk_proxy_time_decay_3y | 58 | 0 | 0.8333 | 0.6950 | 0.8198 | 0.7469 | -0.0120 | 237 | 115 | 16 |
| monotonic_leverage_liquidity_time_decay_3y | 46 | 10 | 0.8298 | 0.6667 | 0.8447 | 0.7435 | -0.0154 | 269 | 99 | 0 |
| monotonic_time_decay_2y | 46 | 24 | 0.8156 | 0.6605 | 0.8508 | 0.7413 | -0.0176 | 285 | 94 | -5 |
| monotonic_core_time_decay_3y | 46 | 15 | 0.8284 | 0.6655 | 0.8380 | 0.7407 | -0.0182 | 267 | 103 | 4 |
| risk_proxy_core_monotonic_time_decay_3y | 58 | 27 | 0.8282 | 0.6687 | 0.8263 | 0.7380 | -0.0208 | 261 | 110 | 11 |

## Final Test 참고

Final test는 rolling 선택 이후의 참고 확인용입니다.

| Candidate | Features | Constraints | PR-AUC | Precision | Recall | F1 | F1 Δ | FP | FN | FN Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| time_decay_half_life_3y | 46 | 0 | 0.8435 | 0.7190 | 0.8571 | 0.7820 | +0.0091 | 68 | 29 | 3 |
| time_decay_half_life_2y | 46 | 0 | 0.8397 | 0.7190 | 0.8571 | 0.7820 | +0.0091 | 68 | 29 | 3 |
| monotonic_guardrail_time_decay_3y | 46 | 9 | 0.8339 | 0.7208 | 0.8522 | 0.7810 | +0.0081 | 67 | 30 | 4 |
| monotonic_time_decay_3y | 46 | 24 | 0.8244 | 0.6969 | 0.8719 | 0.7746 | +0.0017 | 77 | 26 | 0 |
| baseline_current | 46 | 0 | 0.8321 | 0.6941 | 0.8719 | 0.7729 | +0.0000 | 78 | 26 | 0 |
| monotonic_core_time_decay_3y | 46 | 15 | 0.8302 | 0.6932 | 0.8571 | 0.7665 | -0.0064 | 77 | 29 | 3 |
| risk_proxy_time_decay_3y | 58 | 0 | 0.8364 | 0.6948 | 0.8522 | 0.7655 | -0.0074 | 76 | 30 | 4 |
| risk_proxy_monotonic_time_decay_3y | 58 | 12 | 0.8330 | 0.6905 | 0.8571 | 0.7648 | -0.0081 | 78 | 29 | 3 |
| monotonic_time_decay_2y | 46 | 24 | 0.8112 | 0.6951 | 0.8424 | 0.7617 | -0.0112 | 75 | 32 | 6 |
| monotonic_directional | 46 | 24 | 0.8176 | 0.6967 | 0.8374 | 0.7606 | -0.0123 | 74 | 33 | 7 |
| time_decay_half_life_1y | 46 | 0 | 0.8304 | 0.6783 | 0.8621 | 0.7592 | -0.0137 | 83 | 28 | 2 |
| monotonic_leverage_liquidity_time_decay_3y | 46 | 10 | 0.8341 | 0.6593 | 0.8768 | 0.7526 | -0.0203 | 92 | 25 | -1 |
| risk_proxy_core_monotonic_time_decay_3y | 58 | 27 | 0.8305 | 0.6827 | 0.8374 | 0.7522 | -0.0207 | 79 | 33 | 7 |

## Monotonic Profiles

| Profile | Directions | Features |
| --- | ---: | --- |
| broad | 24 | current_ratio, cash_ratio, equity_ratio, debt_ratio, total_borrowings_ratio, capital_impairment_ratio, net_margin, interest_coverage_ratio, pretax_roa, operating_roa, pretax_roe, ocf_to_total_liabilities, ocf_to_total_borrowings, ocf_to_sales, cashflow_coverage_ratio, accruals_ratio, intangible_assets_ratio, spec_spread, short_term_borrowings_share, net_margin_diff, is_2y_consecutive_ocf_deficit, icr_under_1, is_2y_consecutive_operating_loss, gross_profit_industry_year_pct |
| core | 15 | current_ratio, cash_ratio, equity_ratio, debt_ratio, total_borrowings_ratio, capital_impairment_ratio, interest_coverage_ratio, ocf_to_total_liabilities, ocf_to_total_borrowings, cashflow_coverage_ratio, spec_spread, short_term_borrowings_share, is_2y_consecutive_ocf_deficit, icr_under_1, is_2y_consecutive_operating_loss |
| leverage_liquidity | 10 | current_ratio, cash_ratio, equity_ratio, debt_ratio, total_borrowings_ratio, capital_impairment_ratio, interest_coverage_ratio, ocf_to_total_borrowings, cashflow_coverage_ratio, short_term_borrowings_share |
| distress_guardrail | 9 | debt_ratio, total_borrowings_ratio, capital_impairment_ratio, interest_coverage_ratio, cashflow_coverage_ratio, spec_spread, is_2y_consecutive_ocf_deficit, icr_under_1, is_2y_consecutive_operating_loss |
| risk_proxy | 12 | risk_proxy_debt_ratio_industry_year_pct, risk_proxy_total_borrowings_ratio_industry_year_pct, risk_proxy_short_term_borrowings_share_industry_year_pct, risk_proxy_capital_impairment_ratio_industry_year_pct, risk_proxy_spec_spread_industry_year_pct, risk_proxy_accruals_ratio_industry_year_pct, risk_proxy_current_ratio_inverse_industry_year_pct, risk_proxy_cash_ratio_inverse_industry_year_pct, risk_proxy_interest_coverage_inverse_industry_year_pct, risk_proxy_cashflow_coverage_inverse_industry_year_pct, risk_proxy_ocf_to_borrowings_inverse_industry_year_pct, risk_proxy_operating_roa_inverse_industry_year_pct |
| risk_proxy_core | 27 | current_ratio, cash_ratio, equity_ratio, debt_ratio, total_borrowings_ratio, capital_impairment_ratio, interest_coverage_ratio, ocf_to_total_liabilities, ocf_to_total_borrowings, cashflow_coverage_ratio, spec_spread, short_term_borrowings_share, is_2y_consecutive_ocf_deficit, icr_under_1, is_2y_consecutive_operating_loss, risk_proxy_debt_ratio_industry_year_pct, risk_proxy_total_borrowings_ratio_industry_year_pct, risk_proxy_short_term_borrowings_share_industry_year_pct, risk_proxy_capital_impairment_ratio_industry_year_pct, risk_proxy_spec_spread_industry_year_pct, risk_proxy_accruals_ratio_industry_year_pct, risk_proxy_current_ratio_inverse_industry_year_pct, risk_proxy_cash_ratio_inverse_industry_year_pct, risk_proxy_interest_coverage_inverse_industry_year_pct, risk_proxy_cashflow_coverage_inverse_industry_year_pct, risk_proxy_ocf_to_borrowings_inverse_industry_year_pct, risk_proxy_operating_roa_inverse_industry_year_pct |

## Risk Proxy Features

Risk proxy는 모두 값이 클수록 위험이 커지는 방향으로 맞춘 산업-연도 percentile입니다.

| Source | Proxy | Orientation |
| --- | --- | --- |
| debt_ratio | risk_proxy_debt_ratio_industry_year_pct | direct |
| total_borrowings_ratio | risk_proxy_total_borrowings_ratio_industry_year_pct | direct |
| short_term_borrowings_share | risk_proxy_short_term_borrowings_share_industry_year_pct | direct |
| capital_impairment_ratio | risk_proxy_capital_impairment_ratio_industry_year_pct | direct |
| spec_spread | risk_proxy_spec_spread_industry_year_pct | direct |
| accruals_ratio | risk_proxy_accruals_ratio_industry_year_pct | direct |
| current_ratio | risk_proxy_current_ratio_inverse_industry_year_pct | inverse |
| cash_ratio | risk_proxy_cash_ratio_inverse_industry_year_pct | inverse |
| interest_coverage_ratio | risk_proxy_interest_coverage_inverse_industry_year_pct | inverse |
| cashflow_coverage_ratio | risk_proxy_cashflow_coverage_inverse_industry_year_pct | inverse |
| ocf_to_total_borrowings | risk_proxy_ocf_to_borrowings_inverse_industry_year_pct | inverse |
| operating_roa | risk_proxy_operating_roa_inverse_industry_year_pct | inverse |

## Broad Monotonic Constraint Reference

| Feature | Constraint |
| --- | --- |
| current_ratio | -1 위험 감소 |
| cash_ratio | -1 위험 감소 |
| equity_ratio | -1 위험 감소 |
| debt_ratio | +1 위험 증가 |
| total_borrowings_ratio | +1 위험 증가 |
| capital_impairment_ratio | +1 위험 증가 |
| net_margin | -1 위험 감소 |
| interest_coverage_ratio | -1 위험 감소 |
| pretax_roa | -1 위험 감소 |
| operating_roa | -1 위험 감소 |
| pretax_roe | -1 위험 감소 |
| ocf_to_total_liabilities | -1 위험 감소 |
| ocf_to_total_borrowings | -1 위험 감소 |
| ocf_to_sales | -1 위험 감소 |
| cashflow_coverage_ratio | -1 위험 감소 |
| accruals_ratio | +1 위험 증가 |
| intangible_assets_ratio | +1 위험 증가 |
| spec_spread | +1 위험 증가 |
| short_term_borrowings_share | +1 위험 증가 |
| net_margin_diff | -1 위험 감소 |
| is_2y_consecutive_ocf_deficit | +1 위험 증가 |
| icr_under_1 | +1 위험 증가 |
| is_2y_consecutive_operating_loss | +1 위험 증가 |
| gross_profit_industry_year_pct | -1 위험 감소 |

## 재생성 명령

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_time_decay_monotonic_experiments.py
```