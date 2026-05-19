# `data/outputs/`

이 폴더는 CAS 실행 과정에서 생성되는 모델 artifact, 대시보드 산출물, 리포트 파일을 보관합니다.
원본 데이터는 `data/raw/`, 모델 입력 데이터는 `data/input/`에 두고, 이곳에는 재생성 가능한 결과물만 둡니다.

## Git에서 추적하는 산출물

- `modeling/feature_43_xgboost/`
  - 기존 43-feature 기준선 모델입니다.
  - 성능 비교와 회귀 검증을 위해 유지합니다.
- `modeling/feature_44_xgboost/`
  - 현재 런타임이 사용하는 44-feature XGBoost 모델 artifact입니다.
  - `industry_current_ratio_percentile` 파생 변수를 포함한 개선판입니다.
  - `diagnostics/`에는 같은 예측 결과를 기준으로 만든 threshold, calibration, segment, error-case 진단 산출물이 들어갑니다.
- `README.md`
  - 이 폴더의 산출물 기준과 재생성 방법을 설명합니다.

## 로컬에서 생성되는 산출물

- `dashboard/feature_44_mvp/`
  - 대시보드용 회사 목록, 예측 점수, SHAP, peer percentile, manifest입니다.
  - 기본적으로 Git에는 올리지 않고, 아래 스크립트로 다시 만들 수 있습니다.
- `reports/`
  - `scripts/run_agent.py` 또는 대시보드 실행 시 생성되는 기업별 Markdown/JSON 리포트입니다.
  - 실행 결과가 계속 바뀔 수 있으므로 Git 추적 대상이 아닙니다.

## 44-feature 산출물 재생성

```bash
python scripts/rebuild_feature_44_dataset.py
python scripts/build_feature_44_inference_2026.py
python scripts/export_feature_44_dashboard_artifacts.py
python scripts/export_feature_44_model_diagnostics.py
python scripts/export_feature_44_threshold_policy_experiments.py
```

## 정리 기준

- `data/outputs/reports/`는 로컬 실행 산출물이므로 필요하면 삭제해도 됩니다.
- `data/outputs/dashboard/`는 대시보드 재생성 산출물이므로 필요하면 삭제 후 다시 만들 수 있습니다.
- `data/outputs/modeling/feature_44_xgboost/`는 현재 공유용 모델 artifact이므로 임의 삭제하지 않습니다.
- `data/outputs/modeling/feature_43_xgboost/`는 기준선 비교용 artifact이므로 별도 합의 없이 덮어쓰지 않습니다.
