# Committee Review Batch Results

- Rows: 25
- Strict committee success rate: 76.0%
- Review-safe success rate: 80.0%
- Speed: wall `2.255` sec, mean case `0.2673` sec, throughput `665.1885` cases/min

## Result Preview

| sample_category | corp_name | prior_credit_rating | actual_label_name | model_predicted_label_name | final_committee_label | committee_effect | evidence_status | evidence_items |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fn_caught_by_stage2_review | 참좋은여행(주) |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 |
| fn_caught_by_stage2_review | (주)예선테크 |  | 투기등급 | 투자적격 | 적격 | fn_not_escalated | disabled | 0 |
| fn_caught_by_stage2_review | (주)누보 |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 |
| fn_caught_by_stage2_review | (주)휴럼 |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 |
| fn_caught_by_stage2_review | 명신산업(주) |  | 투기등급 | 투자적격 | 적격 | fn_not_escalated | disabled | 0 |
| fp_needing_committee_mitigation | (주)지란지교시큐리티 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 |
| fp_needing_committee_mitigation | (주)에이프로 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 |
| fp_needing_committee_mitigation | (주)예림당 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 |
| fp_needing_committee_mitigation | (주)디지캡 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 |
| fp_needing_committee_mitigation | (주)대성미생물연구소 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 |
| bbb_minus_bb_plus_boundary | (주)바른손 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 |
| bbb_minus_bb_plus_boundary | (주)에스디생명공학 |  | 투자적격 | 투기등급 | 부적격 | fp_not_mitigated | disabled | 0 |
| bbb_minus_bb_plus_boundary | (주)제닉 |  | 투자적격 | 투기등급 | 부적격 | fp_not_mitigated | disabled | 0 |
| bbb_minus_bb_plus_boundary | (주)라닉스 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 |
| bbb_minus_bb_plus_boundary | 대한광통신(주) |  | 투자적격 | 투기등급 | 부적격 | fp_not_mitigated | disabled | 0 |
| true_positive_risk_explanation | 금호전기(주) |  | 투기등급 | 투기등급 | 부적격 | tp_risk_supported | disabled | 0 |
| true_positive_risk_explanation | (주)대창솔루션 |  | 투기등급 | 투기등급 | 부적격 | tp_risk_supported | disabled | 0 |
| true_positive_risk_explanation | 차에이아이헬스케어(주) |  | 투기등급 | 투기등급 | 부적격 | tp_risk_supported | disabled | 0 |
| true_positive_risk_explanation | (주)노블엠앤비 |  | 투기등급 | 투기등급 | 부적격 | tp_risk_supported | disabled | 0 |
| true_positive_risk_explanation | (주)대창솔루션 |  | 투기등급 | 투기등급 | 부적격 | tp_risk_supported | disabled | 0 |
| true_negative_overescalation_guardrail | (주)데이타솔루션 |  | 투자적격 | 투자적격 | 보류 | tn_escalated | disabled | 0 |
| true_negative_overescalation_guardrail | (주)아진엑스텍 |  | 투자적격 | 투자적격 | 적격 | tn_kept_eligible | disabled | 0 |
| true_negative_overescalation_guardrail | 제룡산업(주) |  | 투자적격 | 투자적격 | 적격 | tn_kept_eligible | disabled | 0 |
| true_negative_overescalation_guardrail | (주)서전기전 |  | 투자적격 | 투자적격 | 적격 | tn_kept_eligible | disabled | 0 |
| true_negative_overescalation_guardrail | (주)강동씨앤엘 |  | 투자적격 | 투자적격 | 적격 | tn_kept_eligible | disabled | 0 |
