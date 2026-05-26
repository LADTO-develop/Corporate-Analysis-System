# Committee Review Batch Results

- Rows: 8
- Strict committee success rate: 37.5%
- Review-safe success rate: 100.0%
- Speed: wall `1.7129` sec, mean case `0.8556` sec, throughput `280.2265` cases/min
- Stage 2 LLM speed: not measured

## Result Preview

| sample_category | corp_name | prior_credit_rating | actual_label_name | model_predicted_label_name | final_committee_label | committee_effect | evidence_status | evidence_items |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| true_negative_overescalation_guardrail | (주)엔에프씨 |  | 투자적격 | 투자적격 | 적격 | tn_kept_eligible | ready | 8 |
| true_negative_overescalation_guardrail | (주)휴니드테크놀러지스 |  | 투자적격 | 투자적격 | 적격 | tn_kept_eligible | ready | 12 |
| true_negative_overescalation_guardrail | (주)머큐리 |  | 투자적격 | 투자적격 | 적격 | tn_kept_eligible | ready | 6 |
| true_negative_overescalation_guardrail | (주)레몬 |  | 투자적격 | 투자적격 | 보류 | tn_escalated | ready | 6 |
| true_negative_overescalation_guardrail | 현대무벡스(주) |  | 투자적격 | 투자적격 | 보류 | tn_escalated | ready | 6 |
| true_negative_overescalation_guardrail | (주)한울반도체 |  | 투자적격 | 투자적격 | 보류 | tn_escalated | ready | 6 |
| true_negative_overescalation_guardrail | (주)화승알앤에이 |  | 투자적격 | 투자적격 | 보류 | tn_escalated | ready | 6 |
| true_negative_overescalation_guardrail | (주)하나투어 |  | 투자적격 | 투자적격 | 보류 | tn_escalated | ready | 6 |
