# `data/outputs/`

이 폴더는 CAS 실행 과정에서 생성되는 모델 artifact, 대시보드 산출물, 리포트 파일을 보관합니다.
원본 데이터는 `data/raw/`, 모델 입력 데이터는 `data/input/`에 두고, 이곳에는 재생성 가능한 결과물만 둡니다.

## Git에서 추적하는 산출물

- `modeling/feature_43_xgboost/`
  - 현재 런타임이 사용하는 43-feature XGBoost 모델 artifact입니다.
  - 성능 진단과 대시보드 기본 산출물의 기준입니다.
- `README.md`
  - 이 폴더의 산출물 기준과 재생성 방법을 설명합니다.

## 로컬에서 생성되는 산출물

- `dashboard/feature_43_mvp/`
  - 대시보드용 회사 목록, 예측 점수, SHAP, peer percentile, manifest입니다.
  - 기본적으로 Git에는 올리지 않고, 아래 스크립트로 다시 만들 수 있습니다.
- `reports/`
  - `scripts/run_agent.py` 또는 대시보드 실행 시 생성되는 기업별 Markdown/JSON 리포트입니다.
  - 실행 결과가 계속 바뀔 수 있으므로 Git 추적 대상이 아닙니다.

## 43-feature 산출물 재생성

```bash
python scripts/rebuild_feature_43_dataset.py
python scripts/build_feature_43_inference_2026.py
python scripts/export_feature_43_dashboard_artifacts.py
python scripts/export_feature_43_model_diagnostics.py
python scripts/export_feature_43_threshold_policy_experiments.py
```

## 정리 기준

- `data/outputs/reports/`는 로컬 실행 산출물이므로 필요하면 삭제해도 됩니다.
- `data/outputs/dashboard/`는 대시보드 재생성 산출물이므로 필요하면 삭제 후 다시 만들 수 있습니다.
- `data/outputs/modeling/feature_43_xgboost/`는 현재 공유용 모델 artifact이므로 임의 삭제하지 않습니다.
- 탈락한 후보 변수셋의 전체 artifact는 보관하지 않고, 필요한 경우 43개 기준 diagnostics 안에 작은 비교 요약만 남깁니다.
