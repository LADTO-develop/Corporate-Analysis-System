# credit_44_features 입력 데이터셋

이 폴더는 `data/raw/ts2000/TS2000_Credit_Model_Dataset_Model_V1.csv`를 기반으로 만든
CAS XGBoost 운영 후보 입력 데이터셋입니다.

기존 `credit_43_features`는 기준선으로 남기고, 이 폴더는 신규 파생 변수
`industry_current_ratio_percentile`을 추가한 44개 모델 입력을 별도로 관리합니다.
팀원이 혼동하지 않도록 데이터셋 이름, 파일명, 모델 artifact 이름을 모두 `44`로 맞췄습니다.

## 주요 파일

- `feature_44_master.csv`: 기업-연도 단위 전체 feature master
- `feature_44_inference_2026.csv`: 2025 회계연도 기반 2026 예측 입력 후보
- `feature_44_list.json`: 모델 입력 44개 feature 정의
- `feature_44_dictionary_metadata.json`: 대시보드와 보고서에서 쓰는 지표 설명 사전
- `xgb_train.csv`, `xgb_valid.csv`, `xgb_test.csv`: XGBoost 학습 입력 행렬
- `xgb_id_train.csv`, `xgb_id_valid.csv`, `xgb_id_test.csv`: 각 split의 기업 식별 정보

## 분할 규칙

- train: `fiscal_year <= 2021`
- valid: `fiscal_year == 2022`
- test: `fiscal_year >= 2023`

## 재생성 방법

이 폴더의 CSV는 직접 수정하지 않고 아래 스크립트로 재생성합니다.

```bash
python scripts/rebuild_feature_44_dataset.py
python scripts/build_feature_44_inference_2026.py
python scripts/export_feature_44_dashboard_artifacts.py
```

대시보드용 산출물은 `data/outputs/dashboard/feature_44_mvp`에 생성됩니다.
