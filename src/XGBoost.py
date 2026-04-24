import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import shap
import optuna

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =========================================================
# 0. 경로 설정
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 0-1. 최종 확정 threshold
# =========================================================
THRESHOLD_KQ = 0.39
THRESHOLD_KP = 0.45

# =========================================================
# 0-2. Optuna trial 수
# =========================================================
N_TRIALS_KQ = 40
N_TRIALS_KP = 40

# =========================================================
# 1. 데이터 로드
# =========================================================
kosdaq = pd.read_csv(DATA_DIR / "kosdaq" / "kosdaq_train.csv")
kospi = pd.read_csv(DATA_DIR / "kospi" / "kospi_train.csv")

print("[DATA SHAPE]")
print("KOSDAQ:", kosdaq.shape)
print("KOSPI :", kospi.shape)

# =========================================================
# 2. 학습용 컬럼 분리
# =========================================================
DROP_COLS = [
    "stock_code",
    "company_name",
    "market",
    "fiscal_year",
    "target_binary",
]

def split_xy(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].copy()
    y = df["target_binary"].copy()
    return X, y, feature_cols

def split_by_year(df: pd.DataFrame):
    train_df = df[(df["fiscal_year"] >= 2014) & (df["fiscal_year"] <= 2020)].copy()
    valid_df = df[(df["fiscal_year"] >= 2021) & (df["fiscal_year"] <= 2022)].copy()
    test_df = df[df["fiscal_year"] == 2023].copy()
    return train_df, valid_df, test_df

# =========================================================
# 3. 평가 함수
# =========================================================
def evaluate_model(model, X, y, threshold, dataset_name="test"):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    result = {
        "dataset": dataset_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
        "pr_auc": average_precision_score(y, y_prob),
        "confusion_matrix": confusion_matrix(y, y_pred),
    }
    return result

def print_eval_result(result: dict):
    print(f"\n{result['dataset']}")
    print("threshold:", result["threshold"])
    print("accuracy :", round(result["accuracy"], 4))
    print("precision:", round(result["precision"], 4))
    print("recall   :", round(result["recall"], 4))
    print("f1       :", round(result["f1"], 4))
    print("roc_auc  :", round(result["roc_auc"], 4))
    print("pr_auc   :", round(result["pr_auc"], 4))
    print("cm:\n", result["confusion_matrix"])

# =========================================================
# 4. Feature importance 저장
# =========================================================
def save_feature_importance(model, feature_cols, market_name: str):
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\n[{market_name} Feature Importance Top 20]")
    print(fi.head(20))

    fi.to_csv(
        MODEL_DIR / f"feature_importance_{market_name.lower()}.csv",
        index=False,
        encoding="utf-8-sig"
    )
    return fi

# =========================================================
# 5. SHAP 분석 및 저장
# =========================================================
def run_shap_analysis(model, X_test, market_name: str, idx: int = 0):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"{market_name} SHAP Summary")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_summary_{market_name.lower()}.png", dpi=200, bbox_inches="tight")
    plt.close()

    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[idx],
        feature_names=X_test.columns,
    )

    plt.figure()
    shap.waterfall_plot(explanation, show=False)
    plt.title(f"{market_name} SHAP Waterfall (idx={idx})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_waterfall_{market_name.lower()}_idx{idx}.png", dpi=200, bbox_inches="tight")
    plt.close()

    return explainer, shap_values

# =========================================================
# 6. 데이터 split
# =========================================================
train_kq, valid_kq, test_kq = split_by_year(kosdaq)
X_train_kq, y_train_kq, feature_cols_kq = split_xy(train_kq)
X_valid_kq, y_valid_kq, _ = split_xy(valid_kq)
X_test_kq, y_test_kq, _ = split_xy(test_kq)

print("\n[KOSDAQ SPLIT]")
print("train:", X_train_kq.shape, y_train_kq.value_counts().to_dict())
print("valid:", X_valid_kq.shape, y_valid_kq.value_counts().to_dict())
print("test :", X_test_kq.shape, y_test_kq.value_counts().to_dict())

train_kp, valid_kp, test_kp = split_by_year(kospi)
X_train_kp, y_train_kp, feature_cols_kp = split_xy(train_kp)
X_valid_kp, y_valid_kp, _ = split_xy(valid_kp)
X_test_kp, y_test_kp, _ = split_xy(test_kp)

neg = (y_train_kp == 0).sum()
pos = (y_train_kp == 1).sum()
scale_pos_weight_kp = neg / pos

print("\n[KOSPI SPLIT]")
print("train:", X_train_kp.shape, y_train_kp.value_counts().to_dict())
print("valid:", X_valid_kp.shape, y_valid_kp.value_counts().to_dict())
print("test :", X_test_kp.shape, y_test_kp.value_counts().to_dict())
print("scale_pos_weight_kp:", scale_pos_weight_kp)

# =========================================================
# 7. Optuna objective - KOSDAQ
# =========================================================
def objective_kq(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = XGBClassifier(**params)
    model.fit(X_train_kq, y_train_kq, verbose=False)

    result = evaluate_model(
        model, X_valid_kq, y_valid_kq,
        threshold=THRESHOLD_KQ,
        dataset_name="KOSDAQ valid"
    )
    return result["f1"]

# =========================================================
# 8. Optuna objective - KOSPI
# =========================================================
def objective_kp(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
        "scale_pos_weight": scale_pos_weight_kp,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = XGBClassifier(**params)
    model.fit(X_train_kp, y_train_kp, verbose=False)

    result = evaluate_model(
        model, X_valid_kp, y_valid_kp,
        threshold=THRESHOLD_KP,
        dataset_name="KOSPI valid"
    )
    return result["pr_auc"]

# =========================================================
# 9. Optuna 실행
# =========================================================
print("\n[OPTUNA] KOSDAQ tuning start")
study_kq = optuna.create_study(direction="maximize")
study_kq.optimize(objective_kq, n_trials=N_TRIALS_KQ)

print("\n[KOSDAQ BEST]")
print("best_value:", study_kq.best_value)
print("best_params:", study_kq.best_params)

print("\n[OPTUNA] KOSPI tuning start")
study_kp = optuna.create_study(direction="maximize")
study_kp.optimize(objective_kp, n_trials=N_TRIALS_KP)

print("\n[KOSPI BEST]")
print("best_value:", study_kp.best_value)
print("best_params:", study_kp.best_params)

# =========================================================
# 10. 최적 파라미터로 최종 모델 학습
# =========================================================
best_params_kq = study_kq.best_params.copy()
best_params_kq.update({
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
})

best_params_kp = study_kp.best_params.copy()
best_params_kp.update({
    "scale_pos_weight": scale_pos_weight_kp,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
})

model_kq = XGBClassifier(**best_params_kq)
model_kq.fit(
    X_train_kq,
    y_train_kq,
    eval_set=[(X_train_kq, y_train_kq), (X_valid_kq, y_valid_kq)],
    verbose=False,
)

model_kp = XGBClassifier(**best_params_kp)
model_kp.fit(
    X_train_kp,
    y_train_kp,
    eval_set=[(X_train_kp, y_train_kp), (X_valid_kp, y_valid_kp)],
    verbose=False,
)

# =========================================================
# 11. 최종 성능 평가
# =========================================================
result_kq_valid = evaluate_model(
    model_kq, X_valid_kq, y_valid_kq,
    threshold=THRESHOLD_KQ,
    dataset_name="KOSDAQ valid"
)
result_kq_test = evaluate_model(
    model_kq, X_test_kq, y_test_kq,
    threshold=THRESHOLD_KQ,
    dataset_name="KOSDAQ test"
)

result_kp_valid = evaluate_model(
    model_kp, X_valid_kp, y_valid_kp,
    threshold=THRESHOLD_KP,
    dataset_name="KOSPI valid"
)
result_kp_test = evaluate_model(
    model_kp, X_test_kp, y_test_kp,
    threshold=THRESHOLD_KP,
    dataset_name="KOSPI test"
)

for r in [result_kq_valid, result_kq_test, result_kp_valid, result_kp_test]:
    print_eval_result(r)

# =========================================================
# 12. Feature importance
# =========================================================
feature_importance_kq = save_feature_importance(model_kq, feature_cols_kq, "KOSDAQ")
feature_importance_kp = save_feature_importance(model_kp, feature_cols_kp, "KOSPI")

# =========================================================
# 13. SHAP 분석
# =========================================================
explainer_kq, shap_values_kq = run_shap_analysis(
    model=model_kq,
    X_test=X_test_kq,
    market_name="KOSDAQ",
    idx=0,
)

explainer_kp, shap_values_kp = run_shap_analysis(
    model=model_kp,
    X_test=X_test_kp,
    market_name="KOSPI",
    idx=0,
)

# =========================================================
# 14. 모델 저장
# =========================================================
joblib.dump(model_kq, MODEL_DIR / "xgb_kosdaq.pkl")
joblib.dump(model_kp, MODEL_DIR / "xgb_kospi.pkl")

pd.Series(feature_cols_kq).to_csv(
    MODEL_DIR / "feature_cols_kosdaq.csv",
    index=False,
    header=["feature"],
    encoding="utf-8-sig"
)
pd.Series(feature_cols_kp).to_csv(
    MODEL_DIR / "feature_cols_kospi.csv",
    index=False,
    header=["feature"],
    encoding="utf-8-sig"
)

with open(MODEL_DIR / "threshold_kosdaq.txt", "w", encoding="utf-8") as f:
    f.write(str(THRESHOLD_KQ))

with open(MODEL_DIR / "threshold_kospi.txt", "w", encoding="utf-8") as f:
    f.write(str(THRESHOLD_KP))

pd.DataFrame([best_params_kq]).to_csv(
    MODEL_DIR / "best_params_kosdaq.csv",
    index=False,
    encoding="utf-8-sig"
)
pd.DataFrame([best_params_kp]).to_csv(
    MODEL_DIR / "best_params_kospi.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[saved]")
