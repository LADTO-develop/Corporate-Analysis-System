"""Feature registry — central catalogue of derived features.

Every feature-generating function in ``bfd.features`` registers itself here
with its semantics, data source, and target market scope. The registry is
the single place the pipeline consults to decide which features to compute
for a given ``feature_subset`` (referenced by ``configs/market/*.yaml``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

FeatureSource = Literal[
    "balance_sheet",
    "income_statement",
    "cash_flow",
    "equity_changes",
    "footnotes",
    "macro",
    "news",
]

FeatureKind = Literal["numeric", "categorical", "boolean"]


@dataclass(frozen=True)
class FeatureSpec:
    """Description of a single derived feature."""

    name: str
    source: FeatureSource
    kind: FeatureKind
    description: str
    fn: Callable[..., pd.Series]
    subsets: tuple[str, ...] = ("kospi_v1", "kosdaq_v1")
    dependencies: tuple[str, ...] = field(default_factory=tuple)


class FeatureRegistry:
    """Registry holding every feature spec."""

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Duplicate feature registration: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> FeatureSpec:
        return self._specs[name]

    def list_subset(self, subset: str) -> list[FeatureSpec]:
        return [s for s in self._specs.values() if subset in s.subsets]

    def names_for_subset(self, subset: str) -> list[str]:
        return [s.name for s in self.list_subset(subset)]

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)


# Module-level singleton
REGISTRY = FeatureRegistry()


def feature(
    *,
    source: FeatureSource,
    kind: FeatureKind,
    description: str,
    subsets: tuple[str, ...] = ("kospi_v1", "kosdaq_v1"),
    dependencies: tuple[str, ...] = (),
) -> Callable[[Callable[..., pd.Series]], Callable[..., pd.Series]]:
    """Decorator registering a feature function in the global ``REGISTRY``.

    The function name becomes the feature name.
    """

    def deco(fn: Callable[..., pd.Series]) -> Callable[..., pd.Series]:
        REGISTRY.register(
            FeatureSpec(
                name=fn.__name__,
                source=source,
                kind=kind,
                description=description,
                fn=fn,
                subsets=subsets,
                dependencies=dependencies,
            )
        )
        return fn

    return deco
