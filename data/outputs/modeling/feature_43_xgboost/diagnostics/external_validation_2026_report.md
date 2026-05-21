# 2026 External Rating Validation

## Scope

- Rows: `141`
- Speculative-grade labels: `15`
- Investment-grade labels: `126`
- Positive rate: `10.6%`
- Committee mode: `offline`
- Note: Stage 2 was run with deterministic/offline evidence only. Live Naver/Tavily/OpenDART evidence is intentionally excluded so the benchmark is reproducible.

## Overall Metrics

| View | PR-AUC | ROC-AUC | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage 1 model reject | 0.9528 | 0.9931 | 0.7647 | 0.8667 | 0.8125 | 13 | 4 | 2 | 122 |
| Stage 2 review route | - | - | 0.1471 | 1.0000 | 0.2564 | 15 | 87 | 0 | 39 |
| Stage 2 hold/reject as review | - | - | 0.1400 | 0.9333 | 0.2435 | 14 | 86 | 1 | 40 |
| Stage 2 reject only | - | - | 1.0000 | 0.6667 | 0.8000 | 10 | 0 | 5 | 126 |

## Review Load

- Stage 1 reject count: `17`
- Stage 2 review route count: `102`
- Stage 2 hold/reject count: `100`
- Stage 2 reject-only count: `10`

## Stage 2 Effect Counts

- `tn_escalated_to_hold_or_reject`: `83`
- `tn_kept_eligible`: `39`
- `tp_risk_preserved`: `12`
- `fp_softened_to_eligible_or_hold`: `4`
- `fn_caught_as_review_or_reject`: `2`
- `tp_softened_too_much`: `1`

## Interpretation

- `Stage 1 model reject`는 XGBoost 43개 공식 모델만 사용한 이진 판단입니다.
- `Stage 2 review route`는 1차 모델 경고 기업을 검토 대상으로 유지하면서, 위원회가 보류/부적격으로 올린 기업도 추가합니다.
- `Stage 2 hold/reject as review`는 조기경보 관점에서 보류와 부적격을 모두 추가 검토 대상으로 봅니다.
- `Stage 2 reject only`는 위원회가 최종 부적격까지 올린 경우만 위험 판단으로 봅니다.
- 현재 평가는 live 외부 API를 사용하지 않은 재현 가능한 offline 평가입니다. 뉴스/웹/OpenDART를 실제로 켠 평가는 별도 실험으로 분리해야 합니다.
