# 44-Feature XGBoost Model Artifacts

이 폴더는 `credit_44_features` 입력 데이터셋으로 학습한 Stage 1 XGBoost 모델 artifact입니다.
기존 `feature_43_xgboost`는 기준선으로 남겨두고, 이 폴더를 현재 운영 후보로 사용합니다.

## 포함 파일

- `xgboost_model.json`: XGBoost 모델 파일
- `model_artifact_metadata.json`: feature 목록, threshold, calibration, split 성능, model version 정보
- `diagnostics/`: calibration, threshold, segment, 오류 사례 분석 산출물

## 현재 모델 기준

- Dataset: `credit_44_features`
- Model: `feature_44_xgboost`
- Model version: `ts2000_44_xgboost_mvp`
- 추가 feature: `industry_current_ratio_percentile`
- 결측 처리: XGBoost native missing

## 재생성 방법

```bash
python scripts/rebuild_feature_44_dataset.py
python scripts/build_feature_44_inference_2026.py
python scripts/export_feature_44_dashboard_artifacts.py
python scripts/export_feature_44_model_diagnostics.py
python scripts/export_feature_44_threshold_policy_experiments.py
```

## 보조 검토 신호

대시보드 산출물에는 Stage 2 검토 대상을 넓히기 위한 보조 신호가 포함됩니다.
기준 모델은 44개 feature를 사용하고, 보조 검토 신호는 `delta_accruals_ratio`,
`is_3y_consecutive_operating_loss`를 추가한 46개 feature 조합으로 계산합니다.
이 보조 신호는 최종 라벨을 직접 바꾸는 모델이 아니라, 에이전트 위원회 검토 대상을 넓히는 참고 신호입니다.
