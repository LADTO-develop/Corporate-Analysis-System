# `data/input/`

Model input datasets live here. Files in this directory are derived from
canonical raw sources and should be regenerated through scripts rather than
edited by hand.

Current datasets:
- `credit_43_features/`: current 43-feature XGBoost input set used by the runtime.
- `companies/`: reserved for sample or custom company input payloads.

Candidate feature experiments are not retained as separate input folders. Keep
small comparison summaries under `data/outputs/modeling/feature_43_xgboost/diagnostics/`
when a rejected candidate needs to be explained.
