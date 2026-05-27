# `data/outputs/`

이 폴더는 CAS 실행 과정에서 생성되는 모델 artifact, 대시보드 산출물, 리포트 파일을 보관합니다.
원본 데이터는 `data/raw/`, 모델 입력 데이터는 `data/input/`에 두고, 이곳에는 재생성 가능한 결과물만 둡니다.

## Git에서 추적하는 산출물

- `modeling/feature_46_xgboost/`
  - 현재 런타임이 사용하는 46-feature XGBoost baseline 모델 artifact를 보관합니다.
  - Git 추적 대상은 `README.md`, `xgboost_model.json`,
    `model_artifact_metadata.json`, 핵심 diagnostics 리포트/요약입니다.
- `README.md`
  - 이 폴더의 산출물 기준과 재생성 방법을 설명합니다.

상세 정책은 `docs/artifact_versioning_policy_ko.md`를 따릅니다. 현재 저장소는
DVC/Git LFS를 사용하지 않습니다. 팀 공유에 필요한 작은 기준 diagnostics는 Git에
남기고, 대량 row-level 결과와 반복 batch 산출물은 재생성하거나 release artifact로
공유합니다.

## 로컬에서 생성되는 산출물

- `dashboard/feature_46_mvp/`
  - 대시보드용 회사 목록, 예측 점수, SHAP, peer percentile, manifest입니다.
  - 기본적으로 Git에는 올리지 않고, 아래 스크립트로 다시 만들 수 있습니다.
- `reports/`
  - `scripts/run_agent.py` 또는 대시보드 실행 시 생성되는 기업별 Markdown/JSON 리포트입니다.
  - 실행 결과가 계속 바뀔 수 있으므로 Git 추적 대상이 아닙니다.
- `modeling/*/diagnostics/`
  - 모델 성능, SHAP, threshold, Stage 2 agent 평가 산출물입니다.
  - 기준 결과 공유에 필요한 작은 CSV/JSON/Markdown은 Git에 남기고, 큰 score 파일과
    live batch 원자료는 재생성 또는 release artifact로 관리합니다.

## 46-feature 산출물 재생성

```bash
python scripts/collect_opendart_financial_statements.py --source-kind model-v1 --all-years --fallback-ofs
python scripts/apply_opendart_financial_supplements.py
python scripts/rebuild_feature_46_dataset.py
python scripts/import_feature_46_inference_2026_aux.py
python scripts/build_feature_46_inference_2026.py
python scripts/collect_opendart_financial_statements.py --source-kind inference --target-fiscal-year 2025 --fallback-ofs
python scripts/apply_opendart_inference_financial_supplements.py
python scripts/build_feature_46_inference_2026.py --check-only
python scripts/export_feature_46_dashboard_artifacts.py
python scripts/export_feature_46_model_diagnostics.py
python scripts/export_feature_46_threshold_policy_experiments.py
```

## 현재 모델 기준 성능

OpenDART CFS/OFS 보강 후 46-feature XGBoost 모델을 재생성한 기준입니다.

| 기준선 | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| threshold=0.5 | 0.8321 | 0.9415 | 0.7656 | 0.7241 | 0.7443 |
| tuned threshold=0.30 | 0.8321 | 0.9415 | 0.6941 | 0.8719 | 0.7729 |

## 정리 기준

- `data/outputs/reports/`는 로컬 실행 산출물이므로 필요하면 삭제해도 됩니다.
- `data/outputs/dashboard/`는 대시보드 재생성 산출물이므로 필요하면 삭제 후 다시 만들 수 있습니다.
- `data/outputs/modeling/feature_46_xgboost/`의 baseline 모델 파일은 현재 공유용 artifact이므로 임의 삭제하지 않습니다.
- `data/outputs/modeling/feature_46_xgboost/diagnostics/`의 핵심 리포트/요약은 팀 공유용 기준 결과입니다.
- diagnostics의 큰 row-level score, nested live batch 결과는 재생성 산출물이므로 필요하면 삭제 후 다시 만들 수 있습니다.
- 탈락한 후보 변수셋의 전체 artifact는 보관하지 않고, 필요한 경우 공식 diagnostics 안에 작은 비교 요약만 남깁니다.
