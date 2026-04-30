# Round 1 Model Comparison

- Dataset: `TS2000_Credit_Model_Dataset_Model_V1.csv`
- Feature set manifest: `TS2000_Model_Core29_Manifest.json`
- Split: `train=2014~2020`, `valid=2021~2022`, `test=2023~2024`
- Imputation: `marketwise_train_stats`
- Models: `logistic_regression`, `xgboost`, `lightgbm`

## Test Overall

- `logistic_regression`: PR-AUC 0.6881, ROC-AUC 0.8835, Precision@0.5 0.5498, Recall@0.5 0.8313
- `xgboost`: PR-AUC 0.7812, ROC-AUC 0.9088, Precision@0.5 0.6250, Recall@0.5 0.8434
- `lightgbm`: PR-AUC 0.7722, ROC-AUC 0.9065, Precision@0.5 0.6520, Recall@0.5 0.8012

## Versus Logistic Baseline

- `xgboost`: PR-AUC Δ +0.0931, ROC-AUC Δ +0.0253, Precision Δ +0.0752, Recall Δ +0.0120
- `lightgbm`: PR-AUC Δ +0.0841, ROC-AUC Δ +0.0229, Precision Δ +0.1022, Recall Δ -0.0301
