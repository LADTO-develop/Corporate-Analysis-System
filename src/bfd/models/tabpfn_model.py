"""TabPFN v2.5 wrapper.

Reference: https://github.com/PriorLabs/TabPFN

Canonical instantiation::

    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5)

TabPFN expects raw (unscaled, un-one-hot-encoded) data and does its own
preprocessing internally. Do not wrap it in a pipeline that scales features.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from bfd.models.base import TabularClassifier
from bfd.utils.io import read_yaml
from bfd.utils.logging import get_logger

logger = get_logger(__name__)


class TabPFNModel(TabularClassifier):
    """Thin adapter from TabPFNClassifier to our ``TabularClassifier`` contract."""

    name = "tabpfn"

    def __init__(
        self,
        config_path: str | Path = "configs/model/tabpfn.yaml",
        overrides: dict[str, Any] | None = None,
    ) -> None:
        self.config = read_yaml(config_path)
        if overrides:
            self.config.update(overrides)
        self._clf: TabPFNClassifier | None = None
        self._feature_names: list[str] | None = None

    # ------------------------------------------------------------------
    def _build(self) -> TabPFNClassifier:
        version_name = self.config["model_version"]
        version = getattr(ModelVersion, version_name)
        kwargs = dict(self.config.get("constructor_kwargs", {}))

        if kwargs.get("device") == "auto":
            try:
                import torch

                kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                kwargs["device"] = "cpu"

        logger.info("tabpfn_build", version=version_name, device=kwargs.get("device"))
        return TabPFNClassifier.create_default_for_version(version, **kwargs)

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "TabPFNModel":
        del eval_set  # TabPFN does not use early stopping
        self._feature_names = list(X.columns)
        self._clf = self._build()
        logger.info("tabpfn_fit", n=len(X), n_features=X.shape[1])
        self._clf.fit(X.values, y.values)
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Call .fit() before .predict_proba().")

        batch_size = int(self.config.get("inference", {}).get("predict_batch_size", 1000))
        if len(X) <= batch_size:
            return np.asarray(self._clf.predict_proba(X.values))

        # Prior Labs recommends batched predict for large test sets —
        # each predict call recomputes over the training set.
        out: list[np.ndarray] = []
        for start in range(0, len(X), batch_size):
            chunk = X.iloc[start : start + batch_size]
            out.append(np.asarray(self._clf.predict_proba(chunk.values)))
        return np.vstack(out)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "config": self.config,
                    "clf": self._clf,
                    "feature_names": self._feature_names,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "TabPFNModel":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        model = cls.__new__(cls)
        model.config = blob["config"]
        model._clf = blob["clf"]
        model._feature_names = blob["feature_names"]
        return model
