# 결측값 보강 점검 리포트

## 요약

- 1차 모델 학습/검증/테스트 데이터는 XGBoost native missing 기준을 유지하는 편이 적절합니다. 결측 자체가 의미 있는 신호일 수 있고, 과거 성능 비교에서도 native missing이 Recall/F1 기준으로 유리했습니다.

- 2026 추론 입력은 학습 데이터와 달리 핵심 재무비율 결측이 과도하게 많아 OpenDART 보강 대상으로 판단했습니다. 기존 완전 결측 기준을 핵심 변수 부분 결측 기준으로 넓혀 2025 재무제표 보강을 수행했고, 1,997개 기업에 보강값을 반영했습니다.

- 추가로 2026 추론 대상 중 2024 회계연도 행이 Model_V1에 없는 기업만 따로 추려 OpenDART 2024 재무제표를 수집했습니다. 2,143개 후보 중 2,079개 기업의 2024 lag row를 구성했고, 이를 2025 행과 붙여 `total_assets_growth`, `net_margin_diff` 등 전년 대비 변화량을 다시 계산했습니다.

- 보강 스크립트는 기존 비결측값을 NaN으로 덮지 않도록 수정했습니다. 또한 총차입금은 `short_term_borrowings + long_term_borrowings + bonds_payable`로 보되, 세 항목 중 일부만 수집된 경우에도 수집된 항목을 합산하도록 계산식을 보정했습니다.

## 2026 추론 입력 주요 결측 전후

| feature                     |   before_missing_count |   after_missing_count |   reduced_count |   after_missing_rate |
|:----------------------------|-----------------------:|----------------------:|----------------:|---------------------:|
| short_term_borrowings_share |                   2378 |                   743 |            1635 |               0.3061 |
| ocf_to_total_borrowings     |                   2378 |                   686 |            1692 |               0.2827 |
| total_borrowings_ratio      |                   2377 |                   624 |            1753 |               0.2571 |
| market_to_book              |                    424 |                   424 |               0 |               0.1747 |
| net_margin_diff             |                   2335 |                   393 |            1942 |               0.1619 |
| intangible_assets_ratio     |                   2002 |                   162 |            1840 |               0.0667 |
| total_assets_growth         |                   2001 |                    63 |            1938 |               0.0260 |
| net_margin                  |                   2004 |                    27 |            1977 |               0.0111 |
| ocf_to_sales                |                   2008 |                    24 |            1984 |               0.0099 |
| accruals_ratio              |                   2002 |                    22 |            1980 |               0.0091 |
| cashflow_coverage_ratio     |                   2008 |                    20 |            1988 |               0.0082 |
| interest_coverage_ratio     |                   2003 |                    17 |            1986 |               0.0070 |
| total_debt_turnover         |                   2003 |                    16 |            1987 |               0.0066 |
| ocf_to_total_liabilities    |                   2007 |                    15 |            1992 |               0.0062 |
| cash_ratio                  |                   2006 |                    14 |            1992 |               0.0058 |
| pretax_roe                  |                   2004 |                    14 |            1990 |               0.0058 |
| current_ratio               |                   2005 |                    13 |            1992 |               0.0054 |
| pretax_roa                  |                   2004 |                    13 |            1991 |               0.0054 |
| debt_ratio                  |                   2004 |                    11 |            1993 |               0.0045 |
| equity_ratio                |                   2004 |                    11 |            1993 |               0.0045 |
| operating_roa               |                   2004 |                    11 |            1993 |               0.0045 |

## 현재 feature_43_master 상위 결측 변수

| feature                     |   missing_count |   missing_rate |
|:----------------------------|----------------:|---------------:|
| net_margin_diff             |            1106 |         0.2029 |
| total_assets_growth         |            1086 |         0.1992 |
| market_to_book              |             742 |         0.1361 |
| short_term_borrowings_share |             708 |         0.1299 |
| ocf_to_total_borrowings     |             708 |         0.1299 |
| net_margin                  |              93 |         0.0171 |
| ocf_to_sales                |              93 |         0.0171 |
| capital_impairment_ratio    |              87 |         0.0160 |
| cash_ratio                  |              82 |         0.0150 |
| current_ratio               |              82 |         0.0150 |
| pretax_roe                  |              76 |         0.0139 |
| debt_ratio                  |              76 |         0.0139 |

## 현재 2026 inference 상위 결측 변수

| feature                     |   missing_count |   missing_rate |
|:----------------------------|----------------:|---------------:|
| short_term_borrowings_share |             743 |         0.3061 |
| ocf_to_total_borrowings     |             686 |         0.2827 |
| total_borrowings_ratio      |             624 |         0.2571 |
| market_to_book              |             424 |         0.1747 |
| net_margin_diff             |             393 |         0.1619 |
| intangible_assets_ratio     |             162 |         0.0667 |
| total_assets_growth         |              63 |         0.0260 |
| capital_impairment_ratio    |              54 |         0.0222 |
| net_margin                  |              27 |         0.0111 |
| ocf_to_sales                |              24 |         0.0099 |
| accruals_ratio              |              22 |         0.0091 |
| cashflow_coverage_ratio     |              20 |         0.0082 |
| interest_coverage_ratio     |              17 |         0.0070 |
| total_debt_turnover         |              16 |         0.0066 |
| ocf_to_total_liabilities    |              15 |         0.0062 |

## 판단

- `feature_43_master`의 결측은 대부분 과거 초기 연도, 신규 상장/전년 비교 불가, 원천 값 부재에서 발생하므로 학습용으로는 임의 보강보다 native missing 유지가 안전합니다.

- `feature_43_inference_2026`은 대시보드와 2026 예측 설명에 직접 쓰이므로 OpenDART 기반 보강을 적용했습니다. 이 보강은 모델 입력뿐 아니라 에이전트/대시보드 설명에서 0원 또는 빈 비율이 보이는 문제도 줄여줍니다.

- `total_assets_growth`는 2024 lag 보강 후 2,001개 결측에서 63개까지 감소했습니다. 전년 자산총계가 잡힌 기업은 대부분 정상적으로 변화율이 계산됩니다.

- `net_margin_diff`는 2024 lag 보강과 2025 현재 원천값 재적용 후 2,335개 결측에서 393개까지 감소했습니다. 남은 결측은 2024/2025 중 한쪽의 매출액 또는 순이익이 공시에서 안정적으로 잡히지 않은 경우가 대부분입니다.

- 차입금 관련 변수는 계정 공시 방식 차이와 무차입 또는 미매칭 가능성이 섞여 있어 일부 결측이 남습니다. 현재 총차입금은 단기차입금, 장기차입금, 사채성 차입금을 합산하며, 리스부채는 모델 정의 변경 위험이 있어 포함하지 않았습니다.

- `market_to_book`은 OpenDART 재무제표 항목이 아니라 시장가치/장부가치 기반 변수이므로 이번 OpenDART 보강으로 줄어들지 않습니다. 추가 보강이 필요하면 시가총액 또는 주가 데이터 계열을 별도로 붙여야 합니다.

## 산출물

- `missing_value_current_summary.csv`: 현재 기준 dataset/feature별 결측률

- `missing_value_heavy_rows_model_v1.csv`: 학습 마스터에서 결측이 많은 기업-연도 행

- `missing_value_inference_2026_supplement_candidates.csv`: 보강 후에도 핵심 변수 결측이 많이 남은 2026 추론 후보

- `missing_value_inference_2026_opendart_before_after.csv`: 2026 OpenDART 보강 전후 주요 변수 결측 비교

- `data/raw/opendart/inference_2026_opendart_supplement_audit.csv`: 2025 현재 재무제표 보강 audit

- `data/raw/opendart/inference_2026_opendart_lag_2024_audit.csv`: 2024 lag 재무제표 보강 audit
