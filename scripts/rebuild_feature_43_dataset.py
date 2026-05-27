from __future__ import annotations

import warnings

from rebuild_feature_46_dataset import *  # noqa: F403
from rebuild_feature_46_dataset import main

if __name__ == "__main__":
    warnings.warn(
        "scripts/rebuild_feature_43_dataset.py is deprecated; "
        "use scripts/rebuild_feature_46_dataset.py instead.",
        FutureWarning,
        stacklevel=2,
    )
    main()
