# TS2000 Model_V1 Build Process

이 문서는 **raw TS2000 데이터에서 최종 `TS2000_Credit_Model_Dataset_Model_V1.csv`가 어떻게 만들어졌는지**를 설명하는 문서입니다.

목적:
- 팀원이 같은 규칙으로 데이터를 재현할 수 있게 하기
- AI/LLM이 데이터 생성 흐름을 이해할 수 있게 하기
- 모델 성능 비교 시 전처리 기준 차이를 명확히 하기

## 1. 최종 산출물

최종적으로 만들어지는 핵심 파일은 4개입니다.

1. `TS2000_Credit_Model_Dataset.csv`
   - 모든 파생변수를 포함한 최종 마스터 데이터셋
2. `TS2000_Credit_Model_Dataset_Model_V1.csv`
   - 1차 공식 모델 입력 데이터셋
3. `TS2000_Model_V1_Manifest.json`
   - Model_V1 사용 규칙
4. `TS2000_Model_Core29_Manifest.json`
   - 공식 Core29 feature set 정의

## 2. 사용한 원천 파일

### 2-1. 재무/프로필 원천

`01_Raw_Data/`

- `KOSPI_Profile.csv`
- `KOSDAQ_Profile.csv`
- `KOSPI_BS.csv`
- `KOSDAQ_BS.csv`
- `KOSPI_IS.csv`
- `KOSDAQ_IS.csv`
- `KOSPI_CF.csv`
- `KOSDAQ_CF.csv`

### 2-2. 타겟 원천

- `KOSPI_Target.csv`
- `KOSDAQ_Target.csv`

### 2-3. 거시 원천

- `ECOS_Macro.csv`

### 2-4. 시장/배당 원천

- `KOSPI_market_and_div.csv`
- `KOSDAQ_market_and_div.csv`

## 3. 사용한 주요 스크립트

`01_Raw_Data/`

1. `preprocess_target_ratings.py`
   - raw 신용등급 데이터를 `Target_Processed.csv` / `Target_Processed_audit.csv`로 변환
2. `process_ecos_macro.py`
   - `ECOS_Macro.csv`를 `ECOS_Macro_Processed.csv`로 변환
3. `build_ts2000_credit_model_dataset.py`
   - 재무패널 + 타겟 + 거시 + 시장/배당 정보를 결합해 최종 마스터 데이터셋 생성
4. `export_ts2000_model_outputs.py`
   - 마스터 데이터셋에서 `Model_V1`, `Core29`, manifest, dictionary metadata를 생성

## 4. 전체 생성 흐름

순서는 아래와 같습니다.

1. raw target 전처리
   - `KOSPI_Target.csv`, `KOSDAQ_Target.csv`
   - output:
     - `02_Processed_Data/Target_Processed.csv`
     - `02_Processed_Data/Target_Processed_audit.csv`
2. raw ECOS 전처리
   - `ECOS_Macro.csv`
   - output:
     - `02_Processed_Data/ECOS_Macro_Processed.csv`
3. 최종 마스터 데이터셋 생성
   - 재무패널 + target + macro + market/div
   - output:
     - `03_Outputs/ts2000/TS2000_Credit_Model_Dataset.csv`
4. 모델용 산출물 export
   - output:
     - `03_Outputs/ts2000/TS2000_Credit_Model_Dataset_Model_V1.csv`
     - `03_Outputs/ts2000/TS2000_Model_V1_Manifest.json`
     - `03_Outputs/ts2000/TS2000_Model_Core29_Manifest.json`

## 5. 타겟 전처리 규칙

스크립트:
- `01_Raw_Data/preprocess_target_ratings.py`

핵심 규칙은 다음과 같습니다.

### 5-1. 남기는 등급 범위

- 장기등급만 사용
- 최종 장기등급 순서는 다음과 같이 표준화
  - `AAA`
  - `AA+`, `AA`, `AA-`
  - `A+`, `A`, `A-`
  - `BBB+`, `BBB`, `BBB-`
  - `BB+`, `BB`, `BB-`
  - `B+`, `B`, `B-`
  - `CCC`, `CC`, `C`, `D`

별칭 예:
- `A0 -> A`
- `AA0 -> AA`
- `BBB0 -> BBB`
- `BB0 -> BB`
- `B0 -> B`

### 5-2. 남기는 증권 유형

- 장기 회사채성 증권만 유지
- 허용:
  - 회사채
  - 무보증사채
  - 공모사채
  - security type code `40`(기업신용등급)
- 제외:
  - 기업어음
  - 단기사채
  - 전자단기사채
  - CP
  - ABS/유동화 계열
  - 전환사채/BW 등

### 5-3. 평가일 정규화

- 평가일에 월이 없으면 `12월`로 보정
- 일이 없으면 해당 월 말일로 보정
- `eval_year = normalized evaluation date year`
- `matched fiscal_year = eval_year - 1`

즉 **평가연도 2022의 등급은 회계연도 2021 데이터와 매칭**됩니다.

### 5-4. 평가사 처리

- 국내 평가사를 우선 사용
  - BIG3: `NICE신용평가`, `한국신용평가`, `한국기업평가`
  - 기타 국내 평가사도 국내 후보군으로 유지
- 해외 평가사
  - 예: `S&P`, `Fitch`, `JCR`, `Moody's` 계열
  - 국내 평가사가 같은 회사-평가연도에 하나도 없을 때만 backfill로 사용

### 5-5. 같은 회사 + 평가연도 안에서 대표 등급 선택

같은 `stock_code + eval_year` 안에서:

1. BIG3 평가(`NICE신용평가`, `한국신용평가`, `한국기업평가`)가 하나라도 있으면
   - BIG3 안에서 가장 낮은 등급을 선택
2. BIG3가 없고 기타 국내 평가사가 있으면
   - 기타 국내 평가사 중 가장 낮은 등급을 선택
3. 국내 평가사가 하나도 없으면
   - foreign agency를 backfill로 사용
   - 이때 `기업신용등급/ICR/회사` 성격의 foreign rating을 우선하고, 없으면 plain 회사채 계열 foreign rating 중 가장 낮은 등급을 선택

즉 타겟은 **같은 평가연도 안에서 가장 보수적인 대표 장기등급**으로 정리됩니다.

### 5-6. 상장연도 기준 target 필터

- `Profile` raw에서 `listed_year`를 읽어옵니다.
- `listed_year`가 존재하는 기업은 `fiscal_year >= listed_year`인 target row만 남깁니다.
- 즉 **상장 전 회계연도에 대응하는 target은 공식 타겟셋에서 제외**합니다.
- 이 규칙은 상장기업 패널 기준의 모델 유니버스를 유지하기 위한 보수적 필터입니다.

### 5-7. 이진 타겟

- `BBB-` 이상: 투자적격 `0`
- `BB+` 이하: 투기등급 `1`

## 6. 거시변수 전처리 규칙

스크립트:
- `01_Raw_Data/process_ecos_macro.py`

입력:
- `ECOS_Macro.csv`

출력:
- `ECOS_Macro_Processed.csv`

생성 변수:
- `base_rate`
- `treasury_3y`
- `corp_aa_3y`
- `corp_bbb_3y`
- `ppi`
- `usd_krw`
- `market_spread = corp_aa_3y - treasury_3y`
- `spec_spread = corp_bbb_3y - corp_aa_3y`
- `ppi_yoy`

주의:
- 현재 공식 방식은 **연도별 단일값 기반**
- 팀원 방식처럼 월별 하반기 평균을 쓰지 않음

## 7. 재무패널 정리 규칙

스크립트:
- `01_Raw_Data/build_ts2000_credit_model_dataset.py`

### 7-1. 핵심 key

공통 key:
- `market`
- `stock_code`
- `fiscal_year`

### 7-2. fiscal period 처리

raw 재무데이터의 `회계년도`는 `YYYY/MM` 형태입니다.

예:
- `2020/03`
- `2020/06`
- `2020/12`

우리 파이프라인은:

1. `fiscal_year` 추출
2. `fiscal_month` 추출
3. 같은 `market + stock_code + fiscal_year` 안에서
   - 가장 늦은 `fiscal_month`를 남기고
   - 나머지는 제거

즉 최종 패널은 **company-year 1행**입니다.

### 7-3. 패널 구성 방식

1. `Profile`, `BS`, `IS`, `CF`를 각각 정리
2. 각 패널에서 duplicate key가 있으면 latest fiscal_month만 유지
3. 아래 순서로 `inner join`
   - `profile`
   - `bs`
   - `income`
   - `cashflow`
4. `market_div`는 `left join`
5. 최종적으로 `target`와 `panel`을 `inner join`
6. 마지막으로 `macro`를 `fiscal_year` 기준 `inner join`

즉 **타겟, 재무패널, macro가 모두 갖춰진 행만 최종 모델 데이터에 남습니다.**

## 8. 시장/배당 데이터 처리 규칙

입력:
- `KOSPI_market_and_div.csv`
- `KOSDAQ_market_and_div.csv`

처리 방식:

같은 `market + stock_code + fiscal_year` 내에서 집계:
- `shares_outstanding`: 최대값
- `close_price`: 최대값
- `cash_dividend_thousand`: 절대값이 가장 큰 관측
- `dividend_payer`: 배당금이 0이 아닌 값이 하나라도 있으면 `1`, 아니면 `0`

추가 파생:
- `market_cap = shares_outstanding * close_price`
- `market_to_book = market_cap / (equity_total * 1000)`

주의:
- `equity_total`은 `천원` 단위라 `1000`을 곱합니다.

## 9. 파생변수 생성 규칙

대표적인 파생 축은 다음과 같습니다.

### 9-1. 안정성 / 레버리지

- `current_ratio`
- `cash_ratio`
- `equity_ratio`
- `debt_ratio`
- `total_borrowings_ratio`
- `short_term_borrowings_share`
- `capital_impairment_ratio`

### 9-2. 수익성 / 상환능력

- `net_margin`
- `gross_profit`
- `interest_coverage_ratio`
- `pretax_roa`
- `operating_roa`
- `pretax_roe`
- `operating_roe`

### 9-3. 현금흐름 / 구조

- `ocf_to_total_liabilities`
- `accruals_ratio`
- `depreciation`
- `intangible_assets_ratio`
- `total_debt_turnover`

### 9-4. 시장 / 주주정책

- `dividend_payer`
- `market_to_book`

### 9-5. 추세 / 조기경보

- `total_assets_growth`
- `net_margin_diff`
- `is_2y_consecutive_ocf_deficit`
- `inventory_days_diff`
- `ap_days_diff`
- `icr_under_1`
- `is_zombie_3y`

## 10. 결측과 비율 계산 규칙

### 10-1. 원천 숫자 처리

raw 숫자 컬럼은 문자열 정리 후 숫자로 변환합니다.

### 10-2. 일반 비율

일반 비율은 `safe_ratio` 규칙을 사용합니다.

- 분자/분모가 모두 존재하고
- 분모가 `0`이 아닐 때만 계산
- 아니면 `NaN`

### 10-3. capped ratio

일부 변수는 `capped_ratio`를 사용합니다.

대표:
- `interest_coverage_ratio = operating_income / interest_expense`

규칙:
- 분모가 0이 아니면 일반 계산
- 분모가 `0`이면 `1,000,000`으로 cap

즉 `interest_expense = 0`인 경우 이자보상배율이 구조적으로 매우 큰 값으로 들어갑니다.

### 10-4. 성장률

성장률은:
- 현재값과 전기값이 모두 있고
- 전기값이 `0`이 아닐 때만 계산
- 아니면 `NaN`

## 11. 최종 마스터 데이터셋

파일:
- `TS2000_Credit_Model_Dataset.csv`

특징:
- raw + 파생변수를 넓게 포함
- 실험/보류 변수까지 포함
- 설명용/분석용/후속 실험용 마스터 버전

## 12. Model_V1 생성 규칙

스크립트:
- `01_Raw_Data/export_ts2000_model_outputs.py`

`Model_V1`는 마스터 데이터셋에서 아래만 남깁니다.

유지:
- `feature_x`
- `id_reference`
- `time_reference`
- `target_y`

제외:
- `credit_rating`
- `analysis_only_label`
- `feature_deferred`

즉 `Model_V1`는 **마스터에서 공식 모델 입력에 필요한 변수만 추린 버전**입니다.

현재 `Model_V1`:
- 행 수: `5,166`
- feature 수: `137`

## 13. Core29 생성 규칙

`Core29`는 `Model_V1` 전체 137개 변수 중에서,

- 문헌 근거
- 중복 제거
- SHAP / XGBoost importance
- 다중공선성
- 설명 가능성

을 함께 고려해 추린 공식 29개 핵심 변수셋입니다.

## 14. 공식 모델링 규칙

OOT 분할:
- `train 2014~2020`
- `valid 2021~2022`
- `test 2023~2024`

결측 대치:
- 범주형: `market`별 train 최빈값
- 수치형: `market`별 train 중앙값
- 필요 시 전체 train 통계 fallback

## 15. 재현 시 주의할 점

1. `company-year` 단위 유지
   - 같은 기업-연도에서 `03/06/12`가 남지 않도록 latest month만 유지
2. target은 `eval_year - 1` 기준으로 fiscal_year와 매칭
3. market/div에서 `equity_total * 1000` 단위 보정 필요
4. `interest_coverage_ratio`의 cap 처리 때문에 일부 구조적 패턴이 생길 수 있음
5. `Model_V1`와 `Core29`는 raw 직접 결합물이 아니라, 마스터셋에서 한 번 더 정리된 산출물임

## 16. 한 줄 요약

우리 데이터셋은 **raw target과 raw 재무제표를 그대로 붙인 게 아니라**,  
`타겟 정제 -> 거시 전처리 -> company-year 단위 패널 정리 -> 파생변수 생성 -> 마스터셋 생성 -> Model_V1/Core29 export` 순서로 보수적으로 만든 공식 모델링 데이터셋입니다.
