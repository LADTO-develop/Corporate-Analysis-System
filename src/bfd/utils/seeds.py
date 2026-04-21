"""Seed management for reproducibility across numpy, random, and torch."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seeds(seed: int | None = None) -> int:
    """Set global random seeds on stdlib, numpy, and torch if available.

    Args:
        seed: explicit seed. If None, uses ``BFD_RANDOM_SEED`` env var (default 42).

    Returns:
        The seed actually used.
    """
    if seed is None:
        seed = int(os.getenv("BFD_RANDOM_SEED", "42"))

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:  # torch is an indirect dependency (via tabpfn) but may be absent at install time
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms slow things down; only flip on demand via env.
        if os.getenv("BFD_TORCH_DETERMINISTIC", "0") == "1":
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    return seed
