# Committee Review Batch Results

- Rows: 75
- Strict committee success rate: 85.3%
- Review-safe success rate: 94.7%
- Retry: attempts `1`, initial failed rows `1`, recovered `1`, final failed rows `0`
- Speed: wall `710.7249` sec, mean case `17.9812` sec, throughput `6.3316` cases/min
- Stage 2 LLM speed: mean `17.9127` sec, max `38.8178` sec, cache hits `0`

## Result Preview

| sample_category | corp_name | prior_credit_rating | actual_label_name | model_predicted_label_name | final_committee_label | committee_effect | evidence_status | evidence_items | materiality_event_count | materiality_max_ratio | materiality_top_basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fn_caught_by_stage2_review | (주)토박스코리아 |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 | 0 |  |  |
| fn_caught_by_stage2_review | 대한방직(주) |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 | 0 |  |  |
| fn_caught_by_stage2_review | (주)아모텍 |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 | 0 |  |  |
| fn_caught_by_stage2_review | (주)인스코비 |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 | 0 |  |  |
| fn_caught_by_stage2_review | (주)신성이엔지 |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 | 0 |  |  |
| fn_caught_by_stage2_review | 일성건설(주) |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 | 0 |  |  |
| fn_caught_by_stage2_review | (주)센코 |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 | 0 |  |  |
| fn_caught_by_stage2_review | (주)쏠리드 |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 | 0 |  |  |
| fn_caught_by_stage2_review | 씨아이에스(주) |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 | 0 |  |  |
| fn_caught_by_stage2_review | (주)정다운 |  | 투기등급 | 투자적격 | 적격 | fn_not_escalated | disabled | 0 | 0 |  |  |
| fn_caught_by_stage2_review | (주)시너지이노베이션 |  | 투기등급 | 투자적격 | 적격 | fn_not_escalated | disabled | 0 | 0 |  |  |
| fn_caught_by_stage2_review | (주)미래아이앤지 |  | 투기등급 | 투자적격 | 적격 | fn_not_escalated | disabled | 0 | 0 |  |  |
| fn_caught_by_stage2_review | (주)서산 |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 | 0 |  |  |
| fn_caught_by_stage2_review | (주)금호에이치티 |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 | 0 |  |  |
| fn_caught_by_stage2_review | (주)지앤비에스에코 |  | 투기등급 | 투자적격 | 적격 | fn_not_escalated | disabled | 0 | 0 |  |  |
| fp_needing_committee_mitigation | 에스지씨에너지(주) |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 | 0 |  |  |
| fp_needing_committee_mitigation | (주)디지캡 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 | 0 |  |  |
| fp_needing_committee_mitigation | (주)신원 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 | 0 |  |  |
| fp_needing_committee_mitigation | 에코플라스틱(주) |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 | 0 |  |  |
| fp_needing_committee_mitigation | (주)힘스 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 | 0 |  |  |
| fp_needing_committee_mitigation | 다스코(주) |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 | 0 |  |  |
| fp_needing_committee_mitigation | (주)이상네트웍스 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 | 0 |  |  |
| fp_needing_committee_mitigation | 신원종합개발(주) |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 | 0 |  |  |
| fp_needing_committee_mitigation | 디와이피(주) |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 | 0 |  |  |
| fp_needing_committee_mitigation | (주)동방 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 | 0 |  |  |
| fp_needing_committee_mitigation | (주)티웨이항공 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 | 0 |  |  |
| fp_needing_committee_mitigation | (주)휴니드테크놀러지스 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 | 0 |  |  |
| fp_needing_committee_mitigation | (주)디케이앤디 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 | 0 |  |  |
| fp_needing_committee_mitigation | (주)대성미생물연구소 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 | 0 |  |  |
| fp_needing_committee_mitigation | (주)혜인 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 | 0 |  |  |
