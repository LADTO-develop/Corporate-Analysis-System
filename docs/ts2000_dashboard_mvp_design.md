# TS2000 Dashboard MVP Design

## 목적

이 문서는 `TS2000_Credit_Model_Dataset_Model_V1.csv`와 공식 `Core29` 결과를 활용해, 기업 신용위험을 설명하는 대시보드 MVP를 설계하기 위한 구현 가이드다.

핵심 목표는 세 가지다.

1. 기업별 투자적격성 예측 결과를 한 화면에서 직관적으로 보여준다.
2. SHAP와 핵심 변수값을 이용해 "왜 이런 결과가 나왔는지" 설명한다.
3. LLM이 안전하고 일관되게 설명문을 생성할 수 있도록 입력 구조를 표준화한다.

## 사용 데이터

대시보드 MVP는 아래 공식 파일을 기준으로 한다.

- `data/external/ts2000/TS2000_Credit_Model_Dataset_Model_V1.csv`
- `data/external/ts2000/TS2000_Model_V1_Manifest.json`
- `data/external/ts2000/TS2000_Model_Core29_Manifest.json`
- `data/external/ts2000/model_results/round1_model_comparison/performance_summary.csv`
- `data/external/ts2000/model_results/xgboost_threshold_shap/threshold_summary.csv`
- `data/external/ts2000/model_results/xgboost_threshold_shap/shap_importance_grouped.csv`
- `data/external/ts2000/diagnostics/core29_multicollinearity/core29_vif_summary.csv`

공식 Core29 기준 핵심 수치는 아래와 같다.

- XGBoost test `PR-AUC`: `0.7812`
- XGBoost test `Precision`: `0.6250`
- XGBoost test `Recall`: `0.8434`
- tuned threshold: `0.5437`

## 제품 원칙

### 1. 예측과 설명을 분리한다

- 예측은 머신러닝 모델이 담당한다.
- 설명은 SHAP + Core29 + peer context를 조합해 LLM이 담당한다.
- LLM에게 raw wide feature set 전체를 그대로 주지 않는다.

### 2. 처음부터 산업별 대규모 시뮬레이터로 가지 않는다

MVP는 아래 순서로 확장한다.

1. 단일 기업 진단
2. 동일 산업/시장 비교
3. 변수 기반 what-if 시나리오
4. 산업 단위 배치 시뮬레이션

### 3. 숫자 해석은 Python이, 문장 생성은 LLM이 담당한다

- ratio, percentile, SHAP 정렬, 리스크 밴드 계산은 Python에서 끝낸다.
- LLM은 정리된 payload만 보고 설명문을 만든다.

## 권장 대시보드 IA

MVP는 4개 탭 구조를 추천한다.

### Tab 1. Overview

목적:
- 현재 기업의 위험 수준을 한눈에 보여준다.

표시 요소:
- 기업명
- 시장(`KOSPI`/`KOSDAQ`)
- 산업(`industry_macro_category`)
- 규모(`firm_size_group`)
- 예측확률 `P(speculative)`
- 최종 판정
- threshold
- 리스크 밴드
- LLM 요약 3~4문장

권장 카드:
- `Speculative Probability`
- `Predicted Label`
- `Risk Band`
- `Threshold`

### Tab 2. Drivers

목적:
- 예측 결과의 직접 원인을 설명한다.

표시 요소:
- top SHAP 변수 5~10개
- 위험 증가/완화 방향
- 각 변수의 실제값
- Core29 변수 설명 tooltip
- bar chart 또는 waterfall chart

권장 구성:
- 좌측: SHAP bar/waterfall
- 우측: "핵심 위험 요인" / "완화 요인" 리스트

### Tab 3. Peer Comparison

목적:
- 기업의 상태를 시장/산업 맥락 안에서 해석한다.

표시 요소:
- 동일 산업 median
- 동일 시장 median
- percentile rank
- peer 대비 상/하위 변수

예시 문장:
- "동일 산업 대비 현금비율은 낮고, 단기차입금 비중은 높으며, gross profit은 양호하다."

### Tab 4. Scenario

목적:
- 변수 충격을 가했을 때 예측확률이 어떻게 변하는지 본다.

MVP에서는 거시예측형 시뮬레이터보다 변수 조정형 what-if를 우선한다.

예시 조정 변수:
- `spec_spread`
- `cash_ratio`
- `net_margin`
- `short_term_borrowings_share`
- `capital_impairment_ratio`

예시 preset:
- `Base`
- `Mild Stress`
- `Severe Stress`

## 리스크 밴드 제안

초기 MVP에서는 설명용 risk band를 아래처럼 단순화해도 충분하다.

- `0.00 ~ 0.35`: 안정
- `0.35 ~ 0.65`: 관찰
- `0.65 ~ 1.00`: 고위험

주의:
- 이는 설명용 band다.
- 최종 투자적격/투기등급 판정은 공식 threshold를 따른다.

## 백엔드 아티팩트 설계

대시보드를 빠르게 구현하려면, raw CSV를 매번 직접 읽어 UI에서 모든 계산을 하지 말고 중간 아티팩트를 생성하는 편이 낫다.

### 1. Prediction table

파일 예시:
- `data/outputs/dashboard/predictions_core29.csv`

권장 컬럼:
- `corp_name`
- `stock_code`
- `market`
- `fiscal_year`
- `industry_macro_category`
- `firm_size_group`
- `prob_speculative`
- `predicted_label`
- `threshold`
- `risk_band`

### 2. Top SHAP table

파일 예시:
- `data/outputs/dashboard/top_shap_core29.csv`

권장 컬럼:
- `corp_name`
- `stock_code`
- `fiscal_year`
- `feature`
- `feature_group`
- `feature_value`
- `shap_value`
- `direction`
- `rank`

### 3. Peer percentile table

파일 예시:
- `data/outputs/dashboard/peer_percentiles_core29.csv`

권장 컬럼:
- `corp_name`
- `stock_code`
- `fiscal_year`
- `feature`
- `value`
- `industry_percentile`
- `market_percentile`
- `industry_median`
- `market_median`

### 4. Scenario preset config

파일 예시:
- `data/outputs/dashboard/scenario_presets.json`

예시 구조:

```json
{
  "base": {},
  "mild_stress": {
    "spec_spread": 0.5,
    "cash_ratio": -0.05,
    "net_margin": -0.01
  },
  "severe_stress": {
    "spec_spread": 1.0,
    "cash_ratio": -0.10,
    "net_margin": -0.02,
    "short_term_borrowings_share": 0.05
  }
}
```

## LLM 입력 payload 설계

LLM에는 아래 구조의 요약 payload만 전달한다.

```json
{
  "company_profile": {
    "corp_name": "Example Corp",
    "market": "KOSDAQ",
    "industry_macro_category": "Manufacturing",
    "firm_size_group": "Small"
  },
  "model_output": {
    "prob_speculative": 0.68,
    "predicted_label": 1,
    "threshold": 0.5437,
    "risk_band": "High Risk"
  },
  "key_metrics": {
    "cash_ratio": 0.12,
    "interest_coverage_ratio": 0.84,
    "capital_impairment_ratio": 0.31,
    "net_margin": -0.07
  },
  "top_shap": [
    {
      "feature": "gross_profit",
      "feature_group": "profitability",
      "value": 123.4,
      "shap_value": -0.18,
      "direction": "risk_down"
    },
    {
      "feature": "capital_impairment_ratio",
      "feature_group": "stability",
      "value": 0.31,
      "shap_value": 0.14,
      "direction": "risk_up"
    }
  ],
  "peer_context": {
    "cash_ratio": {
      "industry_percentile": 12,
      "industry_median": 0.19
    }
  },
  "scenario_result": null
}
```

## LLM 프롬프트 초안

### System prompt

```text
You are a credit-risk explanation assistant.
Do not override the model decision.
Explain the result using the provided SHAP drivers, company context, and peer comparison.
Do not invent missing values.
Do not make bankruptcy claims.
Use concise analyst-style Korean.
```

### User prompt template

```text
다음 기업의 투자적격성 예측 결과를 설명해줘.

조건:
- 모델의 예측 결과를 뒤집지 말 것
- SHAP 상위 변수 중심으로 설명할 것
- 숫자를 왜곡하지 말 것
- 1) 요약 2) 핵심 위험 요인 3) 완화 요인 4) 산업 비교 5) 시나리오 시사점 순서로 작성할 것

입력 payload:
{payload_json}
```

### Expected output format

```text
[요약]

[핵심 위험 요인]
- ...

[완화 요인]
- ...

[산업 비교]

[시나리오 시사점]
```

## 구현 권장 구조

현재 저장소 구조에 맞춰 아래 경로를 추천한다.

- `src/cas/dashboard/data_loader.py`
  - TS2000 dataset, manifest, result table 로딩
- `src/cas/dashboard/payloads.py`
  - 기업별 LLM payload 생성
- `src/cas/dashboard/peer_compare.py`
  - 산업/시장 percentile 계산
- `src/cas/dashboard/scenario.py`
  - what-if 시나리오 적용
- `src/cas/dashboard/views/overview.py`
  - overview tab
- `src/cas/dashboard/views/drivers.py`
  - shap tab
- `src/cas/dashboard/views/peers.py`
  - peer comparison tab
- `src/cas/dashboard/views/scenario.py`
  - scenario tab
- `scripts/export_dashboard_inputs.py`
  - 대시보드용 CSV/JSON 아티팩트 생성

## 구현 순서

### Phase 1

- prediction table export
- top SHAP table export
- Streamlit 또는 equivalent UI로 Overview/Drivers 탭 구현
- LLM 설명문 연결

### Phase 2

- peer percentile table export
- Peer Comparison 탭 구현

### Phase 3

- scenario preset config
- Scenario 탭 구현
- 시나리오 전후 delta 설명문 생성

## 첫 구현에서 하지 말 것

- 114개 raw feature를 LLM에 그대로 전달
- 거시변수만 바꿔서 자동으로 재무제표까지 연동되는 것처럼 보이는 시뮬레이션
- 모델 설명과 LLM 자유 서술을 섞어서 모델 판단을 뒤집는 출력

## Definition of Done

아래가 되면 MVP로 본다.

1. 특정 기업을 선택할 수 있다.
2. 예측확률, 판정, risk band를 볼 수 있다.
3. SHAP 상위 변수와 실제값을 볼 수 있다.
4. LLM 설명문이 SHAP과 모순 없이 생성된다.
5. 동일 산업/시장 대비 상대 위치를 볼 수 있다.
6. 최소 2개 scenario preset으로 점수 변화를 볼 수 있다.

## 현재 권장 메인 모델

공식 기준으로는 아래를 기본으로 사용한다.

- feature set: official Core29
- main model: XGBoost
- default threshold: `0.5`
- tuned threshold reference: `0.5437`

XGBoost가 현재 Precision/Recall/PR-AUC 균형이 가장 좋아, 대시보드 MVP의 기본 추론 모델로 적합하다.
