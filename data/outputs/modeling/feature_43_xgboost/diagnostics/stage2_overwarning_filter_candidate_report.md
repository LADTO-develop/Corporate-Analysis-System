# Stage 2 Over-Warning Filter Candidate Experiments

- Generated at: `2026-05-19T14:09:37.973440+00:00`
- Test rows: `924`
- Stage 1 risk rows: `262`
- Stage 1 FP / TP among risk rows: `89` / `173`

## Top Candidate Policies

| policy | candidate_count | fp_mitigated | tp_softened | fp_mitigation_rate | tp_softening_rate | candidate_precision_for_fp | weighted_net_fp_minus_2tp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dividend_balance_sheet_buffer | 4 | 3 | 1 | 0.034 | 0.006 | 0.750 | 1 |
| liquidity_capital_profit_core_plus_ocf_not_deep_negative | 9 | 6 | 3 | 0.067 | 0.017 | 0.667 | 0 |
| liquidity_capital_profit_core | 11 | 7 | 4 | 0.079 | 0.023 | 0.636 | -1 |
| liquidity_capital_profit_core_prob_lt_0_85 | 11 | 7 | 4 | 0.079 | 0.023 | 0.636 | -1 |
| liquidity_capital_profit_core_plus_support8 | 11 | 7 | 4 | 0.079 | 0.023 | 0.636 | -1 |
| liquidity_capital_profit_core_plus_support8_prob_lt_0_85 | 11 | 7 | 4 | 0.079 | 0.023 | 0.636 | -1 |
| liquidity_capital_profit_core_prob_lt_0_80 | 10 | 6 | 4 | 0.067 | 0.023 | 0.600 | -2 |
| low_borrowing_buffer | 10 | 6 | 4 | 0.067 | 0.023 | 0.600 | -2 |

## How To Read

- `fp_mitigated`: 실제 투자적격인데 1차 모델이 부적격으로 본 FP를 위원회 `보류` 후보로 낮춘 수입니다.
- `tp_softened`: 실제 투기등급인데 위원회가 `보류`로 낮출 위험이 있는 수입니다. 낮을수록 좋습니다.
- `candidate_precision_for_fp`: 보류 완화 후보 중 실제 FP 비율입니다.
- `hold_or_reject_recall_after`: `보류`도 2차 검토 대상으로 보면 Recall은 유지된다는 가정의 값입니다.

이 실험은 외부근거가 없거나 강한 악재가 없다는 전제의 오프라인 후보 비교입니다.
실제 위원회 판단에서는 `veto`, 직접 관련 외부 악재, hidden-tail-risk 조건이 우선합니다.
