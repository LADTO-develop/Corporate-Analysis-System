# Credit 46 Features Dataset

이 폴더는 `data/raw/ts2000/TS2000_Credit_Model_Dataset_Model_V1.csv`를
바탕으로 만든 공식 `credit_46_features` 입력 파일 모음입니다.
2026-05-26 기준 공식 Stage 1 XGBoost 모델 입력은 43개에서 46개로 승격했습니다.
TS2000 연결재무제표 값이 비어 있는 기업-연도는 OpenDART 사업보고서 값을
먼저 CFS 기준으로 보강하고, CFS가 없을 때만 OFS로 보강한 뒤 재생성합니다.

구성:
- `feature_46_master.csv`: 기업 식별정보와 37개 원천/파생 변수가 함께 들어 있는 기준 테이블
- `feature_46_inference_2026.csv`: 2025 회계연도 원천 재무데이터와
  `data/raw/ts2000/feature_46_inference_2026_aux.csv`, OpenDART CFS/OFS
  보조 원천으로 보정한 2026 예측용 입력 테이블
- `feature_46_list.json`: 원천/파생 변수 37개와 one-hot 이후 모델 입력 46개 정의
- `feature_46_dictionary_metadata.json`: 대시보드에서 쓰는 한글 지표명, 단위, 설명 사전
- `xgb_train.csv`, `xgb_valid.csv`, `xgb_test.csv`: XGBoost 학습용 입력 매트릭스
- `xgb_id_train.csv`, `xgb_id_valid.csv`, `xgb_id_test.csv`: 각 split의 기업 식별정보

2026-05-26 공식 승격으로 추가된 3개 파생 변수:
- `assets_total_industry_year_pct`
- `gross_profit_industry_year_pct`
- `depreciation_industry_year_pct`

세 변수는 같은 `fiscal_year`와 `industry_macro_category` 안에서 원천 금액의
백분위 순위를 계산한 값입니다. 절대 규모가 큰 기업/작은 기업 자체보다
동종업계-동일연도 안의 상대 위치를 보도록 추가했습니다.

기본 분할 규칙:
- train: `fiscal_year <= 2021`
- valid: `fiscal_year == 2022`
- test: `fiscal_year >= 2023`

현재 기준 확인:
- `feature_46_master.csv`: 5,451행
- test split: 924행 (`fiscal_year >= 2023`)
- Model V1 OpenDART 보강: 741개 후보 중 669개 반영, 보강 후 미보강 73개
- 2026 추론 입력 OpenDART 보강: 424개 후보 중 422개 반영, 보강 후 미보강 2개
- 삼성전자(주): 10행
- (주)토마토시스템: 1행 (`2023 -> 2024`)

이 폴더의 파일은 직접 수정하지 말고 아래 스크립트로 재생성합니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/collect_opendart_financial_statements.py --source-kind model-v1 --all-years --fallback-ofs
/opt/anaconda3/envs/aura/bin/python scripts/apply_opendart_financial_supplements.py
/opt/anaconda3/envs/aura/bin/python scripts/rebuild_feature_43_dataset.py
```

2026 추론 입력의 기업규모, 시장가치, 재무제표 보조 원천을 갱신할 때는 아래
순서로 실행합니다. CAS 실행 자체는 갱신된 내부 CSV만 읽습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/import_feature_43_inference_2026_aux.py
/opt/anaconda3/envs/aura/bin/python scripts/build_feature_43_inference_2026.py
/opt/anaconda3/envs/aura/bin/python scripts/collect_opendart_financial_statements.py --source-kind inference --target-fiscal-year 2025 --fallback-ofs
/opt/anaconda3/envs/aura/bin/python scripts/apply_opendart_inference_financial_supplements.py
/opt/anaconda3/envs/aura/bin/python scripts/build_feature_43_inference_2026.py --check-only
```

이 폴더 자체는 대시보드가 직접 읽지 않습니다.
대시보드에서 사용하려면 아래 스크립트로 별도 아티팩트를 생성해야 합니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_dashboard_artifacts.py
```

생성 결과:
- `data/outputs/dashboard/feature_46_mvp`
