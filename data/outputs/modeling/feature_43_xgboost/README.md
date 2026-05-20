# 43-Feature XGBoost Model Artifacts

이 폴더는 `credit_43_features` 데이터를 기준으로 다시 학습한
XGBoost 모델링 산출물을 저장한 결과입니다. CAS 기준 원본은
`data/raw/ts2000/TS2000_Credit_Model_Dataset_Model_V1.csv`이고,
전체 5,451개 라벨 기업-연도 중 train 3,851개 행으로 학습합니다.

구성:
- `xgboost_model.json`: XGBoost 원본 모델 파일
- `model_artifact_metadata.json`: 사용 변수, 결측 처리 전략, 기준선 등 메타데이터
- `diagnostics/`: 연도/시장/산업별 성능, threshold trade-off, calibration,
  대표 오류 사례, threshold 정책, FP 집중 구간, 43/44 변수셋 비교 요약을
  정리한 모델 진단 산출물

이 경로는 팀 공유용 모델링 산출물이자 Stage 1 런타임이 직접 참조하는 기준
모델 artifact 위치입니다.

`prob_speculative`는 검증셋 기준 Platt scaling을 적용한 보정 확률입니다.
결측값은 XGBoost native missing 방향 학습을 사용하며, metadata의
`fill_values`는 진단/후속 비교용 참고값으로만 보존합니다.
`threshold_tuned`는 validation 기준 Recall 0.85 이상을 유지하는 후보 중
Precision이 가장 높은 기준선을 사용합니다.

`industry_current_ratio_percentile`을 추가한 44개 후보 변수셋은 성능 비교 후
공식 artifact에서 제외했습니다. 비교 근거는
`diagnostics/feature_43_vs_44_performance_comparison.md`에 남겨두었습니다.

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

공식 43개 test 구간의 FP/FN 집중 구간, 기업규모별 성능, feature profile을
더 자세히 보는 오류 심화 진단은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_error_deep_dive.py
```
