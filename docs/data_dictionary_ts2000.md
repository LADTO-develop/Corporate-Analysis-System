# TS2000 Data Dictionary

TS2000 (FnGuide Pro / NICE Pro 계열)은 한국 상장사의 재무제표를 다섯 개의 파일로
내보냅니다. 이 문서는 각 파일의 필수 컬럼과 의미를 정리합니다. 컬럼명은 로더가
소문자로 정규화한 이후 기준입니다.

## 공통 키

모든 파일은 아래 네 개 키로 조인됩니다.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `corp_code` | str | KRX 6자리 종목코드 (e.g. `005930`) |
| `fiscal_year` | int | 회계연도, YYYY |
| `fiscal_quarter` | int ∈ {1,2,3,4} | 4 = 연간, 2 = 반기, 기타 분기 |
| `report_type` | str | `consolidated` \| `separate` |

또한 `market` 컬럼(`KOSPI`/`KOSDAQ`/`KONEX`)은 5본 중 적어도 하나에 포함되어야 합니다.

## 1. 재무상태표 (`bs_{year}.csv`)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `total_assets` | float | 자산총계 |
| `current_assets` | float | 유동자산 |
| `non_current_assets` | float | 비유동자산 |
| `total_liabilities` | float | 부채총계 |
| `current_liabilities` | float | 유동부채 |
| `non_current_liabilities` | float | 비유동부채 |
| `total_equity` | float | 자본총계 (음수 가능 — 자본잠식) |
| `paid_in_capital` | float | 자본금 |
| `retained_earnings` | float | 이익잉여금 |

항등식 `total_assets = total_liabilities + total_equity`가 1% 오차 내에서 성립해야
스키마를 통과합니다 (`bfd.data.schemas.balance_sheet_schema`).

## 2. 손익계산서 (`is_{year}.csv`)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `revenue` | float | 매출액 |
| `cost_of_sales` | float | 매출원가 |
| `gross_profit` | float | 매출총이익 |
| `operating_income` | float | 영업이익 (음수 가능) |
| `interest_expense` | float ≥ 0 | 이자비용 |
| `pretax_income` | float | 법인세차감전순이익 |
| `net_income` | float | 당기순이익 (음수 가능) |

## 3. 현금흐름표 (`cf_{year}.csv`)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `cfo` | float | 영업활동 현금흐름 |
| `cfi` | float | 투자활동 현금흐름 |
| `cff` | float | 재무활동 현금흐름 |
| `capex` | float | 유형자산 취득 (부호 협약: 지출이 음수) |
| `cash_end_of_period` | float ≥ 0 | 기말 현금및현금성자산 |

## 4. 자본변동표 (`sce_{year}.csv`)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `equity_beginning` | float | 기초자본 |
| `equity_ending` | float | 기말자본 |
| `capital_increase` | float ≥ 0 | 유상증자 총액 |
| `capital_decrease` | float ≥ 0 | 감자 총액 |
| `dividends_paid` | float ≥ 0 | 배당금 지급액 |
| `treasury_stock_change` | float | 자기주식 변동 (매입 음수, 처분 양수) |

## 5. 주석 (`notes_{year}.csv`)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `contingent_liabilities_text` | str | 우발부채·지급보증 관련 주석 원문 |
| `litigation_text` | str | 소송·가압류 관련 주석 원문 |
| `going_concern_text` | str | 계속기업 의심 관련 주석 원문 |
| `related_party_text` | str | 특수관계자 거래 주석 원문 |

주석 컬럼은 한국어 평문이며, `bfd.features.footnotes`는 키워드 기반으로 이들에서
리스크 플래그를 추출합니다. 의미 기반 추출은 `bfd.rag.llm_features`가 담당합니다.

## 로딩 예시

```python
from bfd.data.loaders.ts2000 import TS2000Loader

loader = TS2000Loader()
wide = loader.load_wide(year=2024, markets=["KOSPI"])
print(wide.shape, wide.columns[:10].tolist())
```

## 단위 협약

- 통화 단위: **원 (KRW)**. TS2000이 백만원/억원 단위로 내보내는 열이 섞여 있을 경우
  `preprocessing.unit_normalization: KRW` 설정이 로더에서 일괄 원 단위로 정규화합니다.
- 날짜: ISO (YYYY-MM-DD).
- 인코딩: CP949 (TS2000 export 기본). UTF-8 export 시 `configs/data/ts2000.yaml`에서
  `encoding: utf-8`로 오버라이드.
