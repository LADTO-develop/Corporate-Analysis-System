# 43-Feature XGBoost Model Artifacts

이 폴더는 `credit_43_features` 데이터를 기준으로 다시 학습한
XGBoost 모델링 산출물을 팀 공유용으로 저장한 결과입니다.

CAS 기준 원본은 `data/raw/ts2000/TS2000_Credit_Model_Dataset_Model_V1.csv`이고,
전체 5,199개 라벨 기업-연도 중 train 3,851개 행으로 학습합니다.

구성:
- `xgboost_model.json`: XGBoost 원본 모델 파일
- `model_artifact_metadata.json`: 사용 변수, 결측 대치값, 기준선 등 메타데이터

이 경로는 모델링 결과 검토, 팀원 handoff, Stage 1 런타임 추론이 함께 사용하는
단일 기준 모델 artifact 위치입니다.
