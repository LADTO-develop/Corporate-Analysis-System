# 43-Feature XGBoost Model Artifacts

이 폴더는 `credit_43_features` 데이터를 기준으로 다시 학습한
XGBoost 모델링 산출물을 저장한 결과입니다. CAS 기준 원본은
`data/raw/ts2000/TS2000_Credit_Model_Dataset_Model_V1.csv`이고,
전체 5,199개 라벨 기업-연도 중 train 3,851개 행으로 학습합니다.

구성:
- `xgboost_model.json`: XGBoost 원본 모델 파일
- `model_artifact_metadata.json`: 사용 변수, 결측 처리 전략, 기준선 등 메타데이터
- `diagnostics/`: 연도/시장/산업별 성능, threshold trade-off, calibration,
  대표 오류 사례, threshold 정책, FP 집중 구간, SHAP 기반 변수 개선 후보
  실험을 정리한 모델 진단 산출물

이 경로는 팀 공유용 모델링 산출물이자 Stage 1 런타임이 직접 참조하는 기준
모델 artifact 위치입니다.

`prob_speculative`는 검증셋 기준 Platt scaling을 적용한 보정 확률입니다.
결측값은 XGBoost native missing 방향 학습을 사용하며, metadata의
`fill_values`는 진단/후속 비교용 참고값으로만 보존합니다.
`threshold_tuned`는 validation 기준 Recall 0.85 이상을 유지하는 후보 중
Precision이 가장 높은 기준선을 사용합니다.

Rolling validation은 단일 1년 validation에 대한 과신을 줄이기 위해 사용합니다.
특정 경기/시장 국면에 우연히 잘 맞은 후보 변수를 바로 채택하지 않고, 여러
평가연도에서 반복적으로 안정적인지 확인한 뒤 final test는 마지막 확인용으로만
사용합니다.

진단 산출물은 모델을 다시 학습하지 않고 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_model_diagnostics.py
```

threshold 정책별 valid/test 성능 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_threshold_policy_experiments.py
```

오류 사례별 SHAP 패턴 분석은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_error_shap_analysis.py
```

오류 사례별 리뷰 테이블은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_error_case_review.py
```

SHAP 오류 패턴 기반 변수 개선 후보 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_shap_feature_experiments.py
```

원본 Model V1의 미사용 후보 변수를 묶음별로 추가하는 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_candidate_feature_pack_experiments.py
```

단일 후보 변수와 2개 조합 기반 forward selection 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_forward_selection_experiments.py
```

여러 연도 walk-forward rolling OOT validation 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_rolling_validation_experiments.py
```

rolling validation으로 전체 후보를 선별한 뒤 final test 성능을 확인하는 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_rolling_selection_test_experiments.py
```

43개 기준 모델과 45개 변수셋(`delta_accruals_ratio`,
`is_3y_consecutive_operating_loss` 추가)을 직접 비교하는 실험은 아래 명령으로
재생성할 수 있습니다. 이 산출물은 운영 모델 교체가 아니라 Recall 우선 후보
검토용입니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_45_experiment.py
```

45개 변수셋 기준으로 하이퍼파라미터, threshold 정책, Stage 2 보조 트리거 가능성을
비교하는 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_45_improvement_experiments.py
```

XGBoost 하이퍼파라미터 튜닝 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_xgboost_tuning_experiments.py
```
