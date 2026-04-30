# TS2000 Model_V1 Build Process One-Pager

팀원 공유용 요약본입니다.  
핵심은 **raw 데이터에서 바로 모델링한 게 아니라**, `타겟 정제 -> 거시 전처리 -> company-year 패널 정리 -> 파생변수 생성 -> 마스터셋 생성 -> Model_V1/Core29 export` 순서로 만들었다는 점입니다.

## 1. 사용한 raw 파일

- 재무/프로필:
  - `KOSPI_Profile.csv`, `KOSDAQ_Profile.csv`
  - `KOSPI_BS.csv`, `KOSDAQ_BS.csv`
  - `KOSPI_IS.csv`, `KOSDAQ_IS.csv`
  - `KOSPI_CF.csv`, `KOSDAQ_CF.csv`
- 타겟:
  - `KOSPI_Target.csv`, `KOSDAQ_Target.csv`
- 거시:
  - `ECOS_Macro.csv`
- 시장/배당:
  - `KOSPI_market_and_div.csv`, `KOSDAQ_market_and_div.csv`

## 2. 생성 순서

1. `preprocess_target_ratings.py`
   - raw 타겟을 정제해서 `Target_Processed.csv`, `Target_Processed_audit.csv` 생성
2. `process_ecos_macro.py`
   - `ECOS_Macro.csv`를 `ECOS_Macro_Processed.csv`로 변환
3. `build_ts2000_credit_model_dataset.py`
   - 재무패널 + 타겟 + 거시 + 시장/배당을 결합해서 `TS2000_Credit_Model_Dataset.csv` 생성
4. `export_ts2000_model_outputs.py`
   - 마스터셋에서 `TS2000_Credit_Model_Dataset_Model_V1.csv`, `Core29 manifest`, `Model_V1 manifest` 생성

## 3. 타겟 처리 규칙

- 장기등급만 사용
- 단기물/CP/전자단기사채/유동화 계열 제외
- 해외 평가사 제외
- 평가일은 결측 보정 후 `eval_year` 산출
- `matched fiscal_year = eval_year - 1`
- 같은 `stock_code + eval_year` 안에서는:
  - BIG3 평가가 있으면 BIG3 중 가장 낮은 등급 선택
  - 없으면 기타 국내 평가사 중 가장 낮은 등급 선택
- 최종 이진 타겟:
  - `BBB-` 이상 = `0`
  - `BB+` 이하 = `1`

## 4. company-year 단위 정리 방식

- raw 재무데이터의 `회계년도`는 `YYYY/MM` 형태
- 예: `2020/03`, `2020/06`, `2020/12`
- 우리는 `fiscal_year`, `fiscal_month`를 분리한 뒤,
- 같은 `market + stock_code + fiscal_year` 안에서 **가장 늦은 `fiscal_month`만 남김**
- 그래서 최종 패널은 `company-year 1행` 구조임

## 5. 결합 방식

공통 key:
- `market`
- `stock_code`
- `fiscal_year`

결합 순서:
- `profile INNER JOIN bs`
- `INNER JOIN income`
- `INNER JOIN cashflow`
- `LEFT JOIN market_div`
- `INNER JOIN target`
- `INNER JOIN macro` on `fiscal_year`

즉 **타겟, 재무패널, 거시가 모두 있어야 최종 데이터에 남습니다.**

## 6. 시장/배당 데이터 처리

같은 `market + stock_code + fiscal_year` 안에서:
- `shares_outstanding`: 최대값
- `close_price`: 최대값
- `dividend_payer`: 배당금이 0이 아닌 값이 하나라도 있으면 `1`
- `market_cap = shares_outstanding * close_price`
- `market_to_book = market_cap / (equity_total * 1000)`

주의:
- `equity_total`은 `천원` 단위라 `1000`을 곱함

## 7. 대표 파생변수

- 안정성/레버리지:
  - `current_ratio`, `cash_ratio`, `equity_ratio`, `debt_ratio`, `total_borrowings_ratio`
- 수익성/상환능력:
  - `net_margin`, `gross_profit`, `interest_coverage_ratio`, `pretax_roa`, `operating_roa`, `pretax_roe`
- 현금흐름/구조:
  - `ocf_to_total_liabilities`, `accruals_ratio`, `depreciation`
- 시장/주주정책:
  - `dividend_payer`, `market_to_book`
- 추세/조기경보:
  - `total_assets_growth`, `net_margin_diff`, `is_2y_consecutive_ocf_deficit`

## 8. 비율/결측 계산 규칙

- 일반 비율은 분모가 `0`이면 `NaN`
- 성장률도 전기값이 `0`이면 `NaN`
- `interest_coverage_ratio`는 분모(`interest_expense`)가 `0`이면 `1,000,000`으로 cap 처리

## 9. Model_V1와 Core29

- `TS2000_Credit_Model_Dataset.csv`
  - 실험/보류 변수까지 포함한 최종 마스터셋
- `TS2000_Credit_Model_Dataset_Model_V1.csv`
  - 마스터셋에서 `feature_x + id + time + target`만 남긴 공식 모델 입력셋
- `Core29`
  - Model_V1의 137개 feature 중 문헌, 성능, SHAP, 공선성, 설명 가능성을 기준으로 고른 공식 29개 핵심 변수셋

## 10. 공식 모델링 기준

- OOT split:
  - `train 2014~2020`
  - `valid 2021~2022`
  - `test 2023~2024`
- 결측 대치:
  - 범주형은 `market`별 train 최빈값
  - 수치형은 `market`별 train 중앙값
  - 필요 시 전체 train 통계 fallback

## 11. 한 줄 요약

이 데이터셋은 **raw를 바로 붙인 annual table이 아니라**,  
`company-year 단위 정리 + 보수적 inner join + 파생변수 생성 + Model_V1/Core29 export`를 거친 공식 모델링용 데이터셋입니다.
