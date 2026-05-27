# 46-Feature XGBoost Model Artifacts

이 폴더는 `credit_46_features` 데이터를 기준으로 다시 학습한
XGBoost 모델링 산출물을 저장한 결과입니다. CAS 기준 원본은
`data/raw/ts2000/TS2000_Credit_Model_Dataset_Model_V1.csv`이고,
전체 5,451개 라벨 기업-연도 중 train 3,851개 행으로 학습합니다.
TS2000 연결재무제표 값이 비어 있는 기업-연도는 OpenDART 사업보고서 기준
CFS를 먼저 사용하고, CFS가 없을 때만 OFS로 보강한 뒤 46-feature 입력을
재생성합니다.

구성:
- `xgboost_model.json`: XGBoost 원본 모델 파일
- `model_artifact_metadata.json`: 사용 변수, 결측 처리 전략, 기준선 등 메타데이터
- `diagnostics/`: 연도/시장/산업별 성능, threshold trade-off, calibration,
  대표 오류 사례, threshold 정책을 정리한 공식 46-feature 모델 진단 산출물

이 경로는 팀 공유용 모델링 산출물이자 Stage 1 런타임이 직접 참조하는 기준
모델 artifact 위치입니다.

`prob_speculative`는 검증셋 기준 Platt scaling을 적용한 보정 확률입니다.
결측값은 XGBoost native missing 방향 학습을 사용하며, metadata의
`fill_values`는 진단/후속 비교용 참고값으로만 보존합니다.
`threshold_tuned`는 validation 기준 Recall 0.85 이상을 유지하는 후보 중
Precision이 가장 높은 기준선을 사용합니다.

현재 test 성능은 다음과 같습니다.

| 기준선 | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| threshold=0.5 | 0.8321 | 0.9415 | 0.7656 | 0.7241 | 0.7443 |
| tuned threshold=0.30 | 0.8321 | 0.9415 | 0.6941 | 0.8719 | 0.7729 |

43-feature baseline tuned threshold 기준 test 성능은 PR-AUC 0.8329,
ROC-AUC 0.9415, Precision 0.7004, Recall 0.8522, F1 0.7689였습니다.
46-feature 승격 후 Precision은 소폭 낮아졌지만 Recall은 0.8719로 상승했고,
FN은 30건에서 26건으로 줄었습니다.

Rolling validation은 단일 1년 validation에 대한 과신을 줄이기 위해 사용합니다.
특정 경기/시장 국면에 우연히 잘 맞은 후보 변수를 바로 채택하지 않고, 여러
평가연도에서 반복적으로 안정적인지 확인한 뒤 final test는 마지막 확인용으로만
사용합니다.

Stage 2 에이전트 검토 큐 확장용 보조 트리거는 `full_review_trigger_73`을
사용합니다. 이 트리거는 공식 Stage 1 판정을 덮어쓰지 않고, Stage 1에서 놓칠 수
있는 기업을 추가 검토 대상으로 올리는 recall-oriented 신호입니다. 채택 근거와
현재 valid/test 성능은 `docs/stage2_review_trigger_policy_ko.md`에 보존합니다.

데이터와 모델 artifact 전체 재생성 순서는 아래와 같습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/collect_opendart_financial_statements.py --source-kind model-v1 --all-years --fallback-ofs
/opt/anaconda3/envs/aura/bin/python scripts/apply_opendart_financial_supplements.py
/opt/anaconda3/envs/aura/bin/python scripts/rebuild_feature_46_dataset.py
/opt/anaconda3/envs/aura/bin/python scripts/import_feature_46_inference_2026_aux.py
/opt/anaconda3/envs/aura/bin/python scripts/build_feature_46_inference_2026.py
/opt/anaconda3/envs/aura/bin/python scripts/collect_opendart_financial_statements.py --source-kind inference --target-fiscal-year 2025 --fallback-ofs
/opt/anaconda3/envs/aura/bin/python scripts/export_inference_2026_missing_2024_lag_targets.py
/opt/anaconda3/envs/aura/bin/python scripts/collect_opendart_financial_statements.py --source data/raw/opendart/inference_2026_missing_2024_lag_targets.csv --source-kind inference --target-fiscal-year 2025 --opendart-bsns-year 2024 --fallback-ofs --output-dir data/raw/opendart/lag_2024_tmp
/opt/anaconda3/envs/aura/bin/python scripts/apply_opendart_inference_financial_supplements.py --lag-raw-supplement data/raw/opendart/lag_2024_tmp/financial_statements_inference_2024_cfs_with_ofs_fallback_raw.csv
/opt/anaconda3/envs/aura/bin/python scripts/build_feature_46_inference_2026.py --check-only
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_dashboard_artifacts.py
```

진단 산출물은 모델을 다시 학습하지 않고 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_model_diagnostics.py
```

threshold 정책별 valid/test 성능 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_threshold_policy_experiments.py
```

46-feature 입력을 유지한 XGBoost regularization rolling OOT 실험은 아래 명령으로
재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_regularized_xgboost_experiments.py
```

46-feature 입력에 trend diff와 peer-relative percentile 후보를 추가하는 feature
pack 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_trend_peer_feature_experiments.py
```

46-feature 공식 모델 score를 유지하면서 calibration 후보와 dashboard 운영
threshold mode를 비교하는 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_calibration_operating_policy_experiments.py
```

KOSDAQ 제조업 FN rescue gate의 rolling OOT 실험은 아래 명령으로 재생성할 수
있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_manufacturing_fn_rescue_experiments.py
```

Stage 2 에이전트 검토 트리거 feature set 후보 비교는 아래 명령으로 재생성할 수
있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_stage2_trigger_feature_experiments.py
```

43-feature 기준 중간 실험 diagnostics와 Stage 2 반복 실행 산출물은 46-feature
공식 승격 후 제거했습니다. 승격 판단에 필요한 전후 성능과 근거는
`docs/stage1_46_feature_promotion_ko.md` 문서 본문에 보존합니다.
