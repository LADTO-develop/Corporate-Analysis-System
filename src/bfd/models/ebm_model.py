"""Explainable Boosting Machine wrapper.

Reference: https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html

EBM is kept in the ensemble as both a predictor and as the dedicated
"interpretability channel" — its global shape functions are always exported
to the final report via ``bfd.reporting.explanations``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

from bfd.models.base import TabularClassifier
from bfd.utils.io import read_yaml
from bfd.utils.logging import get_logger

logger = get_logger(__name__)


class EBMModel(TabularClassifier):
    """Wrapper around ``interpret.glassbox.ExplainableBoostingClassifier``."""

    name = "ebm"

    def __init__(
        self,
        config_path: str | Path = "configs/model/ebm.yaml",
        overrides: dict[str, Any] | None = None,
    ) -> None:
        self.config: dict[str, Any] = read_yaml(config_path)
        if overrides:
            self.config.update(overrides)
        self._clf: ExplainableBoostingClassifier | None = None
        self._feature_names: list[str] | None = None

    # ------------------------------------------------------------------
    def _build(self) -> ExplainableBoostingClassifier:
        return ExplainableBoostingClassifier(
            max_bins=int(self.config.get("max_bins", 1024)),
            max_interaction_bins=int(self.config.get("max_interaction_bins", 64)),
            interactions=self.config.get("interactions", 15),
            learning_rate=float(self.config.get("learning_rate", 0.01)),
            max_rounds=int(self.config.get("max_rounds", 25000)),
            early_stopping_rounds=int(self.config.get("early_stopping_rounds", 50)),
            early_stopping_tolerance=float(self.config.get("early_stopping_tolerance", 1e-5)),
            outer_bags=int(self.config.get("outer_bags", 8)),
            inner_bags=int(self.config.get("inner_bags", 0)),
            min_samples_leaf=int(self.config.get("min_samples_leaf", 4)),
            max_leaves=int(self.config.get("max_leaves", 3)),
            validation_size=float(self.config.get("validation_size", 0.15)),
            random_state=int(self.config.get("random_state", 42)),
            n_jobs=int(self.config.get("n_jobs", -1)),
        )

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "EBMModel":
        del eval_set  # EBM handles its own validation split internally
        self._feature_names = list(X.columns)
        self._clf = self._build()
        logger.info("ebm_fit", n=len(X), n_features=X.shape[1])
        self._clf.fit(X, y)
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Call .fit() before .predict_proba().")
        return np.asarray(self._clf.predict_proba(X))

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if self._clf is None:
            raise RuntimeError("No model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"config": self.config, "clf": self._clf, "feature_names": self._feature_names},
                f,
            )

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "EBMModel":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        model = cls.__new__(cls)
        model.config = blob["config"]
        model._clf = blob["clf"]
        model._feature_names = blob["feature_names"]
        return model

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------
    def explain_global(self) -> Any:
        """Return the EBM global explanation object (InterpretML Explanation)."""
        if self._clf is None:
            raise RuntimeError("Call .fit() first.")
        return self._clf.explain_global()

    def explain_local(self, X: pd.DataFrame, y: pd.Series | None = None) -> Any:
        """Return the EBM local explanation object for the given samples."""
        if self._clf is None:
            raise RuntimeError("Call .fit() first.")
        return self._clf.explain_local(X, y)

    def term_importances(self) -> pd.Series:
        """Return each term's mean absolute contribution."""
        if self._clf is None:
            raise RuntimeError("Call .fit() first.")
        scores = np.asarray(self._clf.term_importances())
        names = list(self._clf.term_names_)
        return pd.Series(scores, index=names, name="mean_abs_score").sort_values(ascending=False)
