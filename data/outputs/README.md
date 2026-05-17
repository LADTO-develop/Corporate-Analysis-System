# `data/outputs/`

이 폴더는 CAS 실행 과정에서 만들어지는 파생 산출물을 둡니다. 원본 데이터는
`data/raw/`, 모델 입력 데이터는 `data/input/`에 두고, 여기에는 대시보드,
모델 artifact, 리포트처럼 재생성 가능한 결과물을 둡니다.

## Git에 남기는 산출물

현재 Git에서 추적하는 산출물은 팀 실행에 꼭 필요한 최소 파일만 남깁니다.

- `modeling/feature_43_xgboost/`
  - Stage 1 43-feature XGBoost 모델 JSON과 메타데이터입니다.
  - 대시보드와 에이전트 파이프라인이 같은 모델 기준을 쓰기 위한 팀 공유용
    artifact입니다.
  - `diagnostics/`에는 같은 예측 결과를 기준으로 만든 성능 진단 리포트와
    segment/threshold/calibration/error-case 테이블, 변수 개선 및 결측 대체
    실험 결과를 둡니다.
- `README.md`
  - 이 폴더의 산출물 기준과 재생성 방법을 설명합니다.

## 로컬에서 생성될 수 있는 산출물

아래 폴더들은 스크립트나 CLI를 실행하면 로컬에 생길 수 있지만, 기본적으로
Git에 올리지 않는 재생성 산출물입니다.

- `dashboard/feature_43_mvp/`
  - 대시보드용 회사 목록, 예측 점수, SHAP, peer percentile, manifest입니다.
  - `scripts/export_feature_43_dashboard_artifacts.py`로 다시 만들 수 있습니다.
- `reports/`
  - `cas-agent` 또는 에이전트 파이프라인 실행 시 생기는 기업별 Markdown/JSON
    리포트입니다.
  - 샘플 실행 결과가 중복으로 쌓일 수 있으므로 커밋하지 않고 필요 시 삭제해도
    됩니다.

## 재생성 방법

대시보드와 모델 artifact를 다시 만들려면 아래 스크립트를 실행합니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_dashboard_artifacts.py
```

모델을 다시 학습하지 않고 현재 예측 결과의 성능 진단만 다시 만들려면 아래
스크립트를 실행합니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_model_diagnostics.py
```

변수 개선 및 결측값 대체 실험을 다시 만들려면 아래 스크립트를 실행합니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_variable_experiments.py
```

기업별 리포트는 CLI 실행 시 자동으로 `data/outputs/reports/` 아래에 생성됩니다.

```bash
/opt/anaconda3/envs/aura/bin/python -m cas.cli --company-id sample-company
```

## 정리 기준

- `data/outputs/reports/`는 로컬 실행 산출물이므로 삭제해도 됩니다.
- `data/outputs/dashboard/`는 대시보드 재생성 산출물이므로 필요하면 삭제 후 다시
  만들 수 있습니다.
- `data/outputs/modeling/feature_43_xgboost/`는 팀 공유용 기준 모델 artifact이므로
  임의 삭제하지 않습니다.
- `data/outputs/modeling/feature_43_xgboost/diagnostics/`는 모델 artifact와 같은
  기준으로 해석해야 하므로, 삭제했다면 위 진단 스크립트로 다시 생성합니다.
