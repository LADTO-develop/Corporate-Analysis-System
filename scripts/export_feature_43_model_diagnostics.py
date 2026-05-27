from __future__ import annotations

import warnings

from export_feature_46_model_diagnostics import *  # noqa: F403
from export_feature_46_model_diagnostics import main

if __name__ == "__main__":
    warnings.warn(
        "scripts/export_feature_43_model_diagnostics.py is deprecated; "
        "use scripts/export_feature_46_model_diagnostics.py instead.",
        FutureWarning,
        stacklevel=2,
    )
    main()
