# OpenAI Agno Explanation Comparison

- 생성시각(UTC): `2026-05-22T04:33:41Z`
- deterministic rows: `4`
- Agno rows: `4`
- matched rows: `4`

## 요약

- deterministic success rate: `1.0000`
- Agno success rate: `1.0000`
- 평균 설명 품질 점수 변화: `0.1191`
- 최종 라벨 변경 건수: `0`
- 성공 여부 변경 건수: `0`

## Case Preview

| corp_name | model_error_type | stage1_label | deterministic_label | agno_label | deterministic_quality_score | agno_quality_score | quality_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (주)이수앱지스 | false_negative | 투자적격 | 보류 | 보류 | 0.775 | 0.775 | 0.0 |
| (주)타이거일렉 | false_positive | 투기등급 | 보류 | 보류 | 0.7729 | 1.0 | 0.2271 |
| (주)엠젠솔루션 | true_positive | 투기등급 | 보류 | 보류 | 0.6236 | 0.7981 | 0.1745 |
| (주)플라즈맵 | true_positive | 투기등급 | 부적격 | 부적격 | 0.925 | 1.0 | 0.075 |
