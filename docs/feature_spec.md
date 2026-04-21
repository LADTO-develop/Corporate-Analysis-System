# Feature Specification

파생변수 카탈로그는 `bfd.features.registry.REGISTRY`가 런타임에 자동으로
채우며, 이 문서는 그 등록부의 사람용 요약입니다. 새 피처를 추가할 때는
반드시 `@feature(...)` 데코레이터를 사용하세요 — 그래야 CLI와 툴 카탈로그가
자동으로 인식합니다.

## 재무상태표 기반

| 이름 | 종류 | 설명 |
|------|------|------|
| `current_ratio` | numeric | 유동비율 = 유동자산 / 유동부채 |
| `quick_ratio` | numeric | 당좌비율 근사 = (유동자산 − 재고) / 유동부채 |
| `debt_to_equity` | numeric | 부채비율 = 총부채 / 자기자본 |
| `debt_to_assets` | numeric | 총부채 / 총자산 |
| `equity_ratio` | numeric | 자기자본비율 = 자기자본 / 총자산 |
| `capital_impairment` | boolean | 자본잠식 플래그 (자기자본 < 0) |
| `partial_capital_impairment` | boolean | 부분잠식 (자기자본 < 자본금) |
| `capital_impairment_ratio` | numeric | (자본금 − 자기자본) / 자본금 |

## 손익계산서 기반

| 이름 | 종류 | 설명 |
|------|------|------|
| `gross_margin` | numeric | 매출총이익률 |
| `operating_margin` | numeric | 영업이익률 |
| `net_margin` | numeric | 순이익률 |
| `interest_coverage` | numeric | 이자보상배율 = 영업이익 / 이자비용 |
| `interest_coverage_below_one` | boolean | 이자보상배율 < 1 플래그 |
| `operating_loss_flag` | boolean | 영업손실 발생 |
| `net_loss_flag` | boolean | 당기순손실 발생 |

## 현금흐름표 기반

| 이름 | 종류 | 설명 |
|------|------|------|
| `cfo_to_assets` | numeric | 영업현금흐름 / 총자산 |
| `cfo_to_total_debt` | numeric | 영업현금흐름 / 총부채 |
| `free_cash_flow` | numeric | 영업현금흐름 − CAPEX |
| `profit_but_negative_cfo` | boolean | 흑자도산 의심 (영업이익 > 0 ∧ CFO < 0) |
| `triple_negative_cashflow` | boolean | CFO < 0 ∧ CFI < 0 ∧ CFF < 0 |

## 자본변동표 기반

| 이름 | 종류 | 설명 |
|------|------|------|
| `equity_growth_rate` | numeric | 자기자본 증가율 |
| `had_capital_increase` | boolean | 유상증자 이벤트 발생 |
| `had_capital_decrease` | boolean | 감자 이벤트 발생 |
| `dividend_suspension` | boolean | 배당 정지 (전년 > 0, 당년 = 0) |

## 주석 기반 (키워드)

| 이름 | 종류 | 설명 |
|------|------|------|
| `going_concern_mention` | boolean | 계속기업 의심 문구 |
| `litigation_mention` | boolean | 중대 소송/가압류 언급 |
| `contingent_liability_mention` | boolean | 우발부채/지급보증 언급 |
| `footnote_risk_score` | numeric | 위 범주 + 규제 키워드 합 |

더 정교한 의미 기반 추출은 `bfd.rag.llm_features`(Claude 구조화 출력)가 담당합니다.

## 거시 (ECOS 스냅샷)

| 이름 | 종류 | 설명 |
|------|------|------|
| `credit_spread` | numeric | BBB−/AA− 회사채 금리차 (3년물) |
| `real_rate` | numeric | 기준금리 − CPI 전년동월비 |
| `fx_volatility_30d` | numeric | 30일 USD/KRW 로그수익률 표준편차 |

거시 피처는 기업-연도 단위가 아니라 연도 단위 스칼라입니다. `bfd.data.alignment`에서
`fiscal_year_end(t)` 스냅샷으로 각 기업 행에 broadcast됩니다.

## 서브셋

- `kospi_v1` — 전체 피처 세트.
- `kosdaq_v1` — 규모 정규화와 변동성 피처가 더 중요하게 설정된 세트.

`@feature(..., subsets=("kospi_v1",))` 과 같이 특정 시장 서브셋에만 포함되도록
제한할 수 있습니다 (기본값은 두 시장 모두).

## 새 피처 추가 절차

1. 해당 소스 파일(예: `bfd/features/balance_sheet.py`)에 함수를 추가하고
   `@feature(...)` 데코레이터를 붙입니다.
2. 함수명이 곧 피처 이름이 됩니다. `snake_case`.
3. Series 반환. 입력 DataFrame의 인덱스와 정렬되어야 합니다.
4. `tests/unit/test_features_<source>.py` (없으면 추가) 에 최소 1개 테스트 케이스 작성.
5. 본 문서의 해당 표에 행을 추가.

카탈로그는 자동으로 갱신되므로 `configs/market/*.yaml`에서 따로 이름을 나열할
필요가 없습니다.
