# TS2000 Core29 AI Handoff

이 문서는 팀원이나 LLM이 `TS2000 Core29` 데이터셋을 빠르게 이해하고 바로 사용할 수 있도록 만든 요약 문서입니다.

## 1. 이 패키지의 목적

- 목적: 한국 상장기업의 신용위험을 `투자적격(0)` vs `투기등급(1)` 이진분류로 예측
- 공식 설명용/공유용 핵심 변수셋: `Core29`
- 공식 메인 모델: `XGBoost`

## 2. 먼저 보면 좋은 파일

1. `01_Dataset/TS2000_Credit_Model_Dataset_Model_V1.csv`
2. `02_Core29_Definition/TS2000_Model_Core29_Manifest.json`
3. `02_Core29_Definition/TS2000_Model_Core29_Features.csv`
4. `03_Dictionary/ts2000_column_dictionary_metadata.json`
5. `04_Model_Results/performance_summary.csv`

## 3. 데이터 단위

- 최종 모델 데이터 단위는 `market + stock_code + fiscal_year` 기준 `1행`입니다.
- 원천 재무데이터에는 `회계년도`가 `YYYY/MM` 형태로 들어 있으며, 일부 기업은 같은 연도에 `03`, `06`, `09`, `12`가 함께 존재할 수 있습니다.
- 우리 파이프라인은 같은 `market + stock_code + fiscal_year` 안에서 가장 늦은 `fiscal_month`를 남기고 중복을 제거합니다.
- 따라서 최종 `Model_V1`는 `기업-회계연도(company-year)` 기준으로 정리된 데이터셋입니다.

## 4. 현재 공식 데이터셋

- 파일명: `TS2000_Credit_Model_Dataset_Model_V1.csv`
- 행 수: `5,166`
- 전체 feature 수: `137`
- 공식 Core feature 수: `29`
- ID 컬럼: `stock_code`, `corp_name`
- 시간 컬럼: `fiscal_year`, `eval_year`
- 타겟 컬럼: `is_speculative`

타겟 정의:
- `0`: 투자적격
- `1`: 투기등급

## 5. 전처리/결합 규칙 요약

### 5-1. 원천 데이터

- TS2000 재무원천: BS / IS / CF / Profile
- 신용등급 타겟 데이터
- ECOS 거시변수

### 5-2. 전처리 핵심 규칙

1. 원천 재무데이터의 `회계년도(YYYY/MM)`에서 `fiscal_year`, `fiscal_month`를 추출
2. 같은 `market + stock_code + fiscal_year` 내에서는 가장 늦은 `fiscal_month`만 유지
3. 재무패널과 타겟, 거시변수를 보수적으로 결합
4. 비율, 성장률, 추세, 경고 플래그 변수를 생성
5. 모델 입력용 `Model_V1`를 만들 때 일부 변수는 `deferred`로 분리

### 5-3. 결측 처리 규칙

공식 모델링 시:
- 범주형 결측: `market`별 train 기준 최빈값 대치
- 수치형 결측: `market`별 train 기준 중앙값 대치
- 해당 market 통계가 불안정하면 전체 train 통계로 fallback

## 6. 공식 Core29 변수셋

현재 공식 Core29 변수는 아래 29개입니다.

```text
market
listed_year
firm_size_group
industry_macro_category
current_ratio
cash_ratio
assets_total
equity_ratio
debt_ratio
total_borrowings_ratio
capital_impairment_ratio
net_margin
gross_profit
interest_coverage_ratio
pretax_roa
operating_roa
pretax_roe
ocf_to_total_liabilities
accruals_ratio
depreciation
intangible_assets_ratio
total_debt_turnover
dividend_payer
market_to_book
spec_spread
short_term_borrowings_share
total_assets_growth
net_margin_diff
is_2y_consecutive_ocf_deficit
```

## 7. 변수군 구조

- `context`
  - `market`, `listed_year`, `firm_size_group`, `industry_macro_category`
- `stability_leverage`
  - `current_ratio`, `cash_ratio`, `assets_total`, `equity_ratio`, `debt_ratio`, `total_borrowings_ratio`, `capital_impairment_ratio`
- `profitability_returns`
  - `net_margin`, `gross_profit`, `interest_coverage_ratio`, `pretax_roa`, `operating_roa`, `pretax_roe`
- `cashflow_structure`
  - `ocf_to_total_liabilities`, `accruals_ratio`, `depreciation`, `intangible_assets_ratio`, `total_debt_turnover`
- `market_shareholder`
  - `dividend_payer`, `market_to_book`
- `macro`
  - `spec_spread`
- `trend_early_warning`
  - `short_term_borrowings_share`, `total_assets_growth`, `net_margin_diff`, `is_2y_consecutive_ocf_deficit`

## 8. 공식 OOT 분할

- `train`: `2014~2020`
- `valid`: `2021~2022`
- `test`: `2023~2024`

이 분할은 랜덤 분할이 아니라 과거로 학습하고 미래 연도로 검증하는 `out-of-time validation`입니다.

## 9. 공식 모델 성능

테스트셋 전체 기준:

- `Logistic Regression`
  - `PR-AUC 0.6881`
  - `ROC-AUC 0.8835`
  - `Precision 0.5498`
  - `Recall 0.8313`
- `XGBoost`
  - `PR-AUC 0.7812`
  - `ROC-AUC 0.9088`
  - `Precision 0.6250`
  - `Recall 0.8434`
- `LightGBM`
  - `PR-AUC 0.7722`
  - `ROC-AUC 0.9065`
  - `Precision 0.6520`
  - `Recall 0.8012`

메인 추천 모델:
- `XGBoost`

## 10. XGBoost threshold

- 기본 threshold `0.5`
  - `Precision 0.6250`
  - `Recall 0.8434`
- tuned threshold `0.5437`
  - `Precision 0.6308`
  - `Recall 0.8133`

즉 조기경보 성격을 더 중시하면 `0.5`, precision을 조금 더 높이고 싶으면 `0.5437`을 참고할 수 있습니다.

## 11. LLM/AI 사용 권장 방식

LLM에 바로 넣기 좋은 입력은 다음 조합입니다.

1. `Core29` 변수값
2. 기업 기본 맥락
   - `corp_name`, `market`, `firm_size_group`, `industry_macro_category`
3. 모델 예측확률
4. 상위 `SHAP` 변수 5~10개
5. 실제 금액/비율 단위가 유지된 수치

권장하지 않는 방식:
- 전체 raw 재무제표를 그대로 투입
- 대량 one-hot 컬럼을 그대로 투입
- 복잡한 파생식이나 로그변환값만 주고 원래 금액 복원을 LLM에게 맡기기

## 12. deferred / 해석 주의 변수

현재 `Model_V1`에는 공식 Core29에 포함되지 않는 deferred 변수들도 남아 있습니다.

대표 예시:
- `audit_non_clean_flag`
- `audit_strong_risk_flag`
- `audit_missing_flag`
- `audit_opinion_category`
- `ppi_yoy`
- `operating_margin`
- `total_borrowings_diff`
- `lag1_operating_income`
- `lag1_ocf`
- `rolling_3y_avg_operating_income`
- `rolling_3y_avg_ocf`
- `rolling_3y_obs_count`
- `lag1_interest_coverage_ratio`

이 변수들은 실험 과정에서 보류되었거나, 설명성/중복성/구조적 결측 이슈 때문에 공식 Core29에서는 제외된 변수들입니다.

## 13. 한 줄 요약

이 패키지는 `기업-회계연도 1행`으로 정리된 TS2000 신용위험 데이터셋이며, 공식 모델링 기준은 `Core29 + marketwise train imputation + OOT split + XGBoost`입니다.
