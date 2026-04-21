"""Common interface for all supervised tabular models in the project.

All concrete models (``TabPFNModel``, ``LightGBMModel``, ``EBMModel``)
conform to this minimal contract so they can be stacked / ensembled
uniformly in ``ensemble.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class TabularClassifier(ABC):
    """Minimal sklearn-like interface required by the ensemble."""

    #: set by subclass — used for logging and artifact paths
    name: str = "base"

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "TabularClassifier":
        """Fit the model to training data."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return a 2-column array of ``[P(y=0), P(y=1)]``."""

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Default threshold-based binary prediction."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the fitted model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "TabularClassifier":
        """Load a previously saved model."""
