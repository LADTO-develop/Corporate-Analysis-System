from __future__ import annotations

import warnings

from export_feature_46_threshold_policy_experiments import *  # noqa: F403
from export_feature_46_threshold_policy_experiments import main

if __name__ == "__main__":
    warnings.warn(
        "scripts/export_feature_43_threshold_policy_experiments.py is deprecated; "
        "use scripts/export_feature_46_threshold_policy_experiments.py instead.",
        FutureWarning,
        stacklevel=2,
    )
    main()
