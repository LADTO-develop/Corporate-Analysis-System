"""Out-of-fold stacking ensemble over TabPFN / LightGBM / EBM.

The metalearner is trained on OOF predictions from the walk-forward CV,
so it sees the base models' behaviour *before* they overfit to the
training fold. At inference, base models' predictions (trained on all
available data up to the target year) are fed into the metalearner.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from bfd.models.base import TabularClassifier
from bfd.models.ebm_model import EBMModel
from bfd.models.lightgbm_model import LightGBMModel
from bfd.models.tabpfn_model import TabPFNModel
from bfd.utils.io import read_yaml
from bfd.utils.logging import get_logger

logger = get_logger(__name__)


MODEL_CLASSES: dict[str, type[TabularClassifier]] = {
    "tabpfn": TabPFNModel,
    "lightgbm": LightGBMModel,
    "ebm": EBMModel,
}


@dataclass
class BaseModelEntry:
    """One slot in the ensemble."""

    name: str
    model: TabularClassifier
    weight: float
    enabled: bool = True


class StackingEnsemble:
    """Out-of-fold stacking with a logistic-regression meta-learner by default."""

    def __init__(
        self,
        config_path: str | Path = "configs/model/ensemble.yaml",
    ) -> None:
        self.config = read_yaml(config_path)
        self._bases: list[BaseModelEntry] = []
        self._meta: LogisticRegression | None = None
        self._strategy: str = self.config.get("strategy", "stacking")

    # ------------------------------------------------------------------
    def _instantiate_bases(self) -> list[BaseModelEntry]:
        out: list[BaseModelEntry] = []
        for entry in self.config["base_models"]:
            if not entry.get("enabled", True):
                continue
            cls = MODEL_CLASSES[entry["name"]]
            out.append(
                BaseModelEntry(
                    name=entry["name"],
                    model=cls(),
                    weight=float(entry.get("weight", 1.0)),
                    enabled=True,
                )
            )
        return out

    # ------------------------------------------------------------------
    def fit_bases(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> None:
        """Fit each base model on the given (fold) training data."""
        self._bases = self._instantiate_bases()
        for entry in self._bases:
            logger.info("ensemble_fit_base", base=entry.name)
            entry.model.fit(X, y, eval_set=eval_set)

    def predict_proba_bases(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return each base model's predict_proba output."""
        return {entry.name: entry.model.predict_proba(X) for entry in self._bases}

    # ------------------------------------------------------------------
    def fit_meta_on_oof(self, oof_probs: pd.DataFrame, y: pd.Series) -> None:
        """Train the meta-learner on OOF prediction columns.

        ``oof_probs`` has one column per base model: ``p_tabpfn``, ``p_lightgbm``,
        ``p_ebm`` — each is the P(y=1) from that base model on OOF folds.
        """
        if self._strategy != "stacking":
            return

        meta_cfg = self.config.get("meta_learner", {})
        self._meta = LogisticRegression(
            C=float(meta_cfg.get("C", 1.0)),
            penalty=meta_cfg.get("penalty", "l2"),
            class_weight=meta_cfg.get("class_weight", "balanced"),
            max_iter=int(meta_cfg.get("max_iter", 1000)),
            solver="lbfgs",
        )
        logger.info("ensemble_fit_meta", n=len(oof_probs), columns=list(oof_probs.columns))
        self._meta.fit(oof_probs.values, y.values)

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base_probs = self.predict_proba_bases(X)

        if self._strategy == "weighted_average":
            total_w = sum(e.weight for e in self._bases)
            p1 = np.zeros(len(X))
            for entry in self._bases:
                p1 += entry.weight * base_probs[entry.name][:, 1]
            p1 = p1 / total_w
            return np.column_stack([1.0 - p1, p1])

        if self._strategy == "stacking":
            if self._meta is None:
                raise RuntimeError("Stacking selected but meta model has not been fit.")
            meta_input = pd.DataFrame(
                {f"p_{entry.name}": base_probs[entry.name][:, 1] for entry in self._bases}
            )
            return np.asarray(self._meta.predict_proba(meta_input.values))

        if self._strategy == "single_best":
            best_entry = max(self._bases, key=lambda e: e.weight)
            return base_probs[best_entry.name]

        raise ValueError(f"Unknown strategy: {self._strategy}")

    # ------------------------------------------------------------------
    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        for entry in self._bases:
            entry.model.save(dir_path / f"base_{entry.name}.bin")
        with open(dir_path / "meta.pkl", "wb") as f:
            pickle.dump({"config": self.config, "meta": self._meta}, f)

    @classmethod
    def load(cls, dir_path: str | Path) -> "StackingEnsemble":
        dir_path = Path(dir_path)
        ens = cls()
        ens._bases = []
        for entry_cfg in ens.config["base_models"]:
            if not entry_cfg.get("enabled", True):
                continue
            name = entry_cfg["name"]
            model_cls = MODEL_CLASSES[name]
            ens._bases.append(
                BaseModelEntry(
                    name=name,
                    model=model_cls.load(dir_path / f"base_{name}.bin"),
                    weight=float(entry_cfg.get("weight", 1.0)),
                )
            )
        with open(dir_path / "meta.pkl", "rb") as f:
            blob = pickle.load(f)
            ens._meta = blob["meta"]
        return ens
