"""Load TS2000 dashboard artifacts for the Streamlit MVP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cas.utils.io import read_json

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_DIR = ROOT / "data" / "outputs" / "dashboard" / "ts2000_core29_mvp"


@dataclass(slots=True)
class DashboardArtifacts:
    """In-memory dashboard artifact bundle."""

    artifact_dir: Path
    company_universe: pd.DataFrame
    company_latest: pd.DataFrame
    peer_percentiles: pd.DataFrame
    feature_dictionary: pd.DataFrame
    global_shap_reference: pd.DataFrame
    scenario_presets: dict[str, object]
    llm_payload_template: dict[str, object]
    model_summary: dict[str, object]
    export_manifest: dict[str, object]
    prediction_scores: pd.DataFrame | None
    local_shap: pd.DataFrame | None
    industry_year_summary: pd.DataFrame | None
    industry_latest_summary: pd.DataFrame | None
    industry_shap_summary: pd.DataFrame | None


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV if it exists, otherwise return None."""
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def load_dashboard_artifacts(artifact_dir: Path | None = None) -> DashboardArtifacts:
    """Load dashboard artifacts from disk."""
    base_dir = artifact_dir or DEFAULT_ARTIFACT_DIR

    company_universe = pd.read_csv(base_dir / "company_universe_core29.csv", encoding="utf-8-sig")
    company_latest = pd.read_csv(base_dir / "company_latest_core29.csv", encoding="utf-8-sig")
    peer_percentiles = pd.read_csv(base_dir / "peer_percentiles_core29.csv", encoding="utf-8-sig")
    feature_dictionary = pd.read_csv(
        base_dir / "feature_dictionary_core29.csv",
        encoding="utf-8-sig",
    )
    global_shap_reference = pd.read_csv(
        base_dir / "global_shap_reference_core29.csv",
        encoding="utf-8-sig",
    )

    scenario_presets = read_json(base_dir / "scenario_presets_core29.json")
    llm_payload_template = read_json(base_dir / "llm_payload_template_core29.json")
    model_summary = read_json(base_dir / "model_summary_core29.json")
    export_manifest = read_json(base_dir / "dashboard_export_manifest.json")

    prediction_scores = _read_optional_csv(base_dir / "prediction_scores_core29.csv")
    local_shap = _read_optional_csv(base_dir / "local_shap_core29.csv")
    industry_year_summary = _read_optional_csv(base_dir / "industry_year_summary_core29.csv")
    industry_latest_summary = _read_optional_csv(base_dir / "industry_latest_summary_core29.csv")
    industry_shap_summary = _read_optional_csv(base_dir / "industry_shap_summary_core29.csv")

    return DashboardArtifacts(
        artifact_dir=base_dir,
        company_universe=company_universe,
        company_latest=company_latest,
        peer_percentiles=peer_percentiles,
        feature_dictionary=feature_dictionary,
        global_shap_reference=global_shap_reference,
        scenario_presets=scenario_presets,
        llm_payload_template=llm_payload_template,
        model_summary=model_summary,
        export_manifest=export_manifest,
        prediction_scores=prediction_scores,
        local_shap=local_shap,
        industry_year_summary=industry_year_summary,
        industry_latest_summary=industry_latest_summary,
        industry_shap_summary=industry_shap_summary,
    )
