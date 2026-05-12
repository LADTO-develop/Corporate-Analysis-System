# 43-Feature Dataset

이 폴더는 `data/raw/ts2000/TS2000_Credit_Model_Dataset_Model_V1.csv`를
바탕으로 만든 공식 `credit_43_features` 입력 파일 모음입니다.

구성:
- `feature_43_master.csv`: 기업 식별정보와 34개 원천 변수가 함께 들어 있는 기준 테이블
- `feature_43_inference_2026.csv`: 2025 회계연도 원천 재무데이터로부터 생성한 2026 예측용 입력 테이블
- `feature_43_list.json`: 원천 변수 34개와 one-hot 이후 모델 입력 43개 정의
- `feature_43_dictionary_metadata.json`: 대시보드에서 쓰는 한글 지표명, 단위, 설명 사전
- `xgb_train.csv`, `xgb_valid.csv`, `xgb_test.csv`: XGBoost 학습용 입력 매트릭스
- `xgb_id_train.csv`, `xgb_id_valid.csv`, `xgb_id_test.csv`: 각 split의 기업 식별정보

기본 분할 규칙:
- train: `fiscal_year <= 2021`
- valid: `fiscal_year == 2022`
- test: `fiscal_year >= 2023`

현재 기준 확인:
- `feature_43_master.csv`: 5,199행
- 삼성전자(주): 10행
- (주)토마토시스템: 1행 (`2023 -> 2024`)

이 폴더의 파일은 직접 수정하지 말고 아래 스크립트로 재생성합니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/rebuild_feature_43_dataset.py
```

이 폴더 자체는 대시보드가 직접 읽지 않습니다.
대시보드에서 사용하려면 아래 스크립트로 별도 아티팩트를 생성해야 합니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_dashboard_artifacts.py
```

생성 결과:
- `data/outputs/dashboard/feature_43_mvp`
