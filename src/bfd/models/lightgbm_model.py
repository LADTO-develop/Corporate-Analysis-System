"""LightGBM wrapper with early stopping via native callbacks.

Reference: https://lightgbm.readthedocs.io/en/latest/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from bfd.models.base import TabularClassifier
from bfd.utils.io import read_yaml
from bfd.utils.logging import get_logger

logger = get_logger(__name__)


class LightGBMModel(TabularClassifier):
    """LightGBM binary classifier adapter."""

    name = "lightgbm"

    def __init__(
        self,
        config_path: str | Path = "configs/model/lightgbm.yaml",
        overrides: dict[str, Any] | None = None,
    ) -> None:
        self.config: dict[str, Any] = read_yaml(config_path)
        if overrides:
            self.config.update(overrides)
        self._booster: lgb.Booster | None = None
        self._feature_names: list[str] | None = None
        self._categorical_features: list[str] = list(self.config.get("categorical_features", []))

    # ------------------------------------------------------------------
    def _params(self) -> dict[str, Any]:
        """Return the LightGBM param dict. Excludes fit-level settings."""
        fit_only = {"early_stopping_rounds", "n_estimators", "categorical_features"}
        return {k: v for k, v in self.config.items() if k not in fit_only}

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "LightGBMModel":
        self._feature_names = list(X.columns)

        cat_indices = [X.columns.get_loc(c) for c in self._categorical_features if c in X.columns]

        train_ds = lgb.Dataset(
            X,
            label=y,
            categorical_feature=cat_indices or "auto",
            feature_name=list(X.columns),
            free_raw_data=False,
        )

        valid_sets: list[lgb.Dataset] = [train_ds]
        valid_names: list[str] = ["train"]
        if eval_set is not None:
            X_val, y_val = eval_set
            val_ds = lgb.Dataset(
                X_val,
                label=y_val,
                categorical_feature=cat_indices or "auto",
                feature_name=list(X_val.columns),
                reference=train_ds,
                free_raw_data=False,
            )
            valid_sets.append(val_ds)
            valid_names.append("valid")

        callbacks: list[Any] = [lgb.log_evaluation(period=0)]
        es_rounds = self.config.get("early_stopping_rounds")
        if es_rounds and eval_set is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=es_rounds, verbose=False))

        n_estimators = int(self.config.get("n_estimators", 500))

        logger.info("lgbm_fit", n=len(X), n_features=X.shape[1], n_estimators=n_estimators)
        self._booster = lgb.train(
            params=self._params(),
            train_set=train_ds,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Call .fit() before .predict_proba().")
        p1 = np.asarray(self._booster.predict(X, num_iteration=self._booster.best_iteration))
        return np.column_stack([1.0 - p1, p1])

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if self._booster is None:
            raise RuntimeError("No booster to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save the booster in LightGBM's native format plus a sidecar for metadata
        self._booster.save_model(str(path))
        (path.with_suffix(path.suffix + ".features.json")).write_text(
            __import__("json").dumps(self._feature_names, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "LightGBMModel":
        path = Path(path)
        model = cls(**kwargs)
        model._booster = lgb.Booster(model_file=str(path))
        features_sidecar = path.with_suffix(path.suffix + ".features.json")
        if features_sidecar.exists():
            import json

            model._feature_names = json.loads(features_sidecar.read_text(encoding="utf-8"))
        return model

    # ------------------------------------------------------------------
    # Convenience for feature importance / inspection
    # ------------------------------------------------------------------
    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        if self._booster is None:
            raise RuntimeError("Call .fit() first.")
        scores = self._booster.feature_importance(importance_type=importance_type)
        return pd.Series(scores, index=self._feature_names, name=importance_type).sort_values(
            ascending=False
        )
