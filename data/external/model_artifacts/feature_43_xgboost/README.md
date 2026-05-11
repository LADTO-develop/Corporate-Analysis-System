# 43-Feature XGBoost Model Artifacts

이 폴더는 `credit_43_features` 데이터를 기준으로 다시 학습한
XGBoost 모델 artifact를 저장한 결과입니다.

구성:
- `xgboost_model.json`: XGBoost 원본 모델 파일
- `model_artifact_metadata.json`: 사용 변수, 결측 대치값, 기준선 등 메타데이터

이 폴더는 Git에 포함되는 기준 모델 artifact 위치이며,
Stage 1 모델 추론은 이 경로를 직접 참조합니다.
