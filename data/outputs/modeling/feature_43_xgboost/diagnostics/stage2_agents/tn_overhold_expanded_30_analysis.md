# TN Overhold Expanded Analysis

rolling validation TN 후보 30건을 추가로 뽑아 deterministic Stage 2 committee를 재실행하고, 재무 방어축/차단 신호 기준으로 보류 유지와 적격 하향 후보를 분리했다.

## Summary

- Rows: 30
- Before liquidity-buffer guardrail: eligible 21/30, hold 9/30
- After liquidity-buffer guardrail: eligible 22/30, hold 8/30
- Review-safe success: 30/30 = 100.0% before and after
- Changed rows: 1
- Remaining hold rows with financial blockers: 8/8

## Changed Rows

| corp_name | fiscal_year | credit_rating | before_final_committee_label | after_final_committee_label | financial_supports | financial_blockers | current_ratio | cash_ratio | interest_coverage_ratio | ocf_to_sales | ocf_to_total_liabilities | equity_ratio | debt_ratio | total_borrowings_ratio | short_term_borrowings_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (주)레몬 | 2020.0000 | BBB | 보류 | 적격 | liquidity,cashflow,capital |  | 0.7443 | 0.2969 | 18.7971 | 0.2612 | 0.5441 | 0.6099 | 0.6396 | 0.2635 | 1.0000 |

## Remaining Hold Rows

| corp_name | fiscal_year | credit_rating | after_committee_decision_type | financial_blockers | financial_supports | interest_coverage_ratio | ocf_to_sales | ocf_to_total_liabilities | net_margin | equity_ratio | debt_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 현대무벡스(주) | 2020.0000 | AA- | risk_hold | icr_under_1 | capital | -0.5055 |  | 0.0758 |  | 0.8399 | 0.1907 |
| (주)한울반도체 | 2020.0000 | BBB | boundary_hold | ocf_both_negative,net_margin_below_minus_10pct | liquidity,capital | 3.1357 | -0.0697 | -0.4005 | -0.1438 | 0.9126 | 0.0957 |
| 신원종합개발(주) | 2020.0000 | A | risk_hold | icr_under_1,ocf_both_negative,net_margin_below_minus_10pct | liquidity,capital | 0.5981 | -0.0862 | -0.1078 | -0.1156 | 0.4306 | 1.3224 |
| (주)화승알앤에이 | 2021.0000 | BBB- | boundary_hold | weak_interest_and_capital,short_term_borrowings_pressure | cashflow | 1.4174 | 0.0561 | 0.0830 | 0.0113 | 0.1500 | 5.6683 |
| (주)아시아경제 | 2022.0000 | BBB | boundary_hold | icr_under_1,weak_interest_and_capital | liquidity | -0.7575 | 0.9244 | 0.1487 | 0.2049 | 0.2639 | 2.7891 |
| (주)하나투어 | 2022.0000 | BBB- | boundary_hold | icr_under_1,ocf_both_negative,net_margin_below_minus_10pct,two_year_op_loss_and_ocf_deficit,weak_interest_and_capital | liquidity | -35.9097 | -0.0891 | -0.0288 | -0.5616 | 0.2189 | 3.5679 |
| (주)예림당 | 2019.0000 | BBB | boundary_hold | icr_under_1,weak_interest_and_capital | liquidity,cashflow | -0.8258 | 0.1467 | 0.1945 | -0.0586 | 0.3453 | 1.8963 |
| (주)일지테크 | 2019.0000 | BBB | boundary_hold | icr_under_1,weak_interest_and_capital |  | -2.4588 | 0.1654 | 0.1371 | -0.0224 | 0.3403 | 1.9386 |

## Current Outcome Counts

| after_final_committee_label | after_committee_decision_type | after_recommended_guardrail_action | rows |
| --- | --- | --- | --- |
| 보류 | boundary_hold | keep_hold | 6.0000 |
| 보류 | risk_hold | keep_hold | 2.0000 |
| 적격 | eligible | already_eligible | 22.0000 |

## Interpretation

- 기존에는 보류 9건 중 `(주)레몬` 1건이 유동성/현금흐름/자본 방어축이 모두 강한 적격 하향 후보였다.
- cash ratio와 OCF가 강한 current-ratio watch 예외를 좁게 추가한 뒤 `(주)레몬`만 보류에서 적격으로 내려갔다.
- 남은 보류 8건은 ICR 1 미만, OCF 동시 음수, 순이익률 -10% 미만, 약한 자본/이자보상 조합, 단기차입 압력 중 하나 이상이 있어 휴맥스형 보류 유지 케이스로 분류한다.
- 따라서 다음 guardrail은 보류를 더 공격적으로 줄이기보다, 남은 8건을 `위험 보류`가 아닌 `경계/확인필요 보류`로 명확히 표시하는 UX/라벨 세분화 쪽이 더 안전하다.
