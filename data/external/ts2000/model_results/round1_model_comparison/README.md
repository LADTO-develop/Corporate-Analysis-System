# Round 1 Model Comparison

- Dataset: `TS2000_Credit_Model_Dataset_Model_V1.csv`
- Feature set manifest: `TS2000_Model_Core29_Manifest.json`
- Split: `train=2014~2020`, `valid=2021~2022`, `test=2023~2024`
- Imputation: `marketwise_train_stats`
- Models: `logistic_regression`, `xgboost`, `lightgbm`

## Test Overall

- `logistic_regression`: PR-AUC 0.6839, ROC-AUC 0.8783, Precision@0.5 0.5472, Recall@0.5 0.8323
- `xgboost`: PR-AUC 0.7621, ROC-AUC 0.8998, Precision@0.5 0.6233, Recall@0.5 0.8323
- `lightgbm`: PR-AUC 0.7661, ROC-AUC 0.9040, Precision@0.5 0.6373, Recall@0.5 0.7784

## Versus Logistic Baseline

- `xgboost`: PR-AUC Δ +0.0782, ROC-AUC Δ +0.0214, Precision Δ +0.0761, Recall Δ +0.0000
- `lightgbm`: PR-AUC Δ +0.0821, ROC-AUC Δ +0.0256, Precision Δ +0.0900, Recall Δ -0.0539
