# Feature 46 XGBoost Diagnostics

이 폴더는 공식 46개 변수 XGBoost 모델의 Stage 1 정량 진단 산출물이 생성되는
위치입니다.

Git에는 이 README와 팀 공유에 필요한 핵심 CSV/JSON/Markdown 진단 결과를
남깁니다. 다만 큰 row-level score, live batch 원자료, 반복 실험 디렉터리는
재생성 가능한 산출물이므로 커밋하지 않고, 공유가 필요하면 release artifact로
첨부합니다.

43-feature 기준 legacy diagnostics와 중간 반복 실험 파일은 46-feature 공식
승격 후 제거했습니다. 승격 근거와 전후 성능은
`docs/stage1_46_feature_promotion_ko.md`에 문서 본문으로 보존합니다.
Stage 2 에이전트 검토 트리거는 `full_review_trigger_73`을 사용하며, 채택
기록은 `docs/stage2_review_trigger_policy_ko.md`에 보존합니다.

## 폴더 기준

| 위치 | 내용 |
|---|---|
| `diagnostics/` | 공식 46-feature Stage 1 모델 성능, threshold, calibration, 대표 오류 사례 |

## 자주 보는 Stage 1 파일

| 파일 | 용도 |
|---|---|
| `model_diagnostics_report.md` | 공식 46개 XGBoost 모델의 전체 성능 요약 |
| `model_diagnostics_summary.json` | 공식 모델 성능과 기준선 메타데이터 |
| `threshold_sweep.csv` | threshold별 precision/recall/F1 trade-off |
| `calibration_bins.csv` | 확률 보정 구간별 관측 부실률 |
| `segment_performance.csv` | 연도/시장/산업 등 세그먼트별 성능 |
| `error_cases.csv` | tuned threshold 기준 대표 FP/FN 사례 |
| `threshold_policy_experiment_report.md` | global/segment threshold 정책 비교 |
| `threshold_policy_experiment_*.csv/json` | threshold 정책 실험의 정량 결과 |
| `regularized_xgboost_rolling_tuning_report.md` | 46-feature XGBoost regularization 후보의 rolling OOT 비교 |
| `regularized_xgboost_rolling_tuning_*.csv/json` | regularization tuning fold/final/summary 정량 결과 |
| `trend_peer_feature_pack_report.md` | trend diff/peer-relative feature pack 후보의 rolling OOT 비교 |
| `trend_peer_feature_pack_*.csv/json` | trend/peer feature pack fold/final/segment/summary 정량 결과 |
| `calibration_operating_policy_report.md` | calibration 후보와 dashboard 운영 threshold mode의 rolling/Final Test 비교 |
| `calibration_operating_policy_*.csv/json` | calibration, bin, operating mode, rolling fold/summary, segment mode 정량 결과 |
| `macro_interaction_feature_pack_report.md` | macro regime 변화량과 macro shock × 재무 취약도 interaction 후보의 rolling OOT 비교 |
| `macro_interaction_feature_pack_*.csv/json` | macro interaction feature pack fold/final/segment/summary 정량 결과 |
| `manufacturing_fn_rescue_gate_report.md` | KOSDAQ 제조업 FN rescue gate의 rolling OOT 및 Final Test 비교 |
| `manufacturing_fn_rescue_gate_*.csv/json` | 제조업 FN rescue gate 후보별 fold/final/summary 정량 결과 |
| `stage2_trigger_feature_set_report.md` | `stage2_aux_48` 대비 Stage2 보조 트리거 후보 비교와 `full_review_trigger_73` 채택 근거 |
| `stage2_trigger_feature_set_*.csv/json` | Stage2 trigger feature set 후보별 fold/final/summary 정량 결과 |
