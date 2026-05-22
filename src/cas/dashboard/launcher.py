"""Operational launcher for the Streamlit credit-risk dashboard."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from cas.dashboard.data_loader import DEFAULT_ARTIFACT_DIR

ROOT = Path(__file__).resolve().parents[3]
APP_PATH = ROOT / "src" / "cas" / "dashboard" / "credit_app.py"
EXPORT_SCRIPT_PATH = ROOT / "scripts" / "export_feature_43_dashboard_artifacts.py"
INFERENCE_2026_EXPORT_SCRIPT_PATH = (
    ROOT / "scripts" / "export_feature_43_inference_2026_dashboard_artifacts.py"
)

REQUIRED_DASHBOARD_ARTIFACTS = (
    "company_universe.csv",
    "company_latest.csv",
    "peer_percentiles.csv",
    "feature_dictionary.csv",
    "global_shap_reference.csv",
    "scenario_presets.json",
    "llm_payload_template.json",
    "model_summary.json",
    "dashboard_export_manifest.json",
    "prediction_scores.csv",
    "local_shap.csv",
    "industry_year_summary.csv",
    "industry_latest_summary.csv",
    "industry_shap_summary.csv",
)


def parse_args() -> argparse.Namespace:
    """Parse launcher arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare dashboard artifacts when needed, then launch the CAS Streamlit dashboard."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Dashboard artifact directory to load or generate.",
    )
    parser.add_argument(
        "--rebuild-artifacts",
        action="store_true",
        help="Regenerate dashboard artifacts before launching Streamlit.",
    )
    parser.add_argument(
        "--no-auto-build",
        action="store_true",
        help="Do not generate missing artifacts automatically.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Optional Streamlit server address, for example 127.0.0.1 or 0.0.0.0.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional Streamlit server port. Omit to let Streamlit choose its default.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Streamlit without opening a browser window.",
    )
    parser.add_argument(
        "--stage2-runner",
        choices=("deterministic", "agno"),
        default=None,
        help="Override the Stage 2 runner used by dashboard committee previews.",
    )
    parser.add_argument(
        "--stage2-model",
        default=None,
        help="Override CAS_STAGE2_MODEL for Agno dashboard runs.",
    )
    parser.add_argument(
        "--stage2-agno-mode",
        choices=("single", "multi_llm_committee"),
        default=None,
        help=(
            "Route Agno Stage 2 through one provider or Claude/GPT/Gemini committee mode."
        ),
    )
    parser.add_argument(
        "--stage2-model-provider",
        default=None,
        help="Default Agno model provider: anthropic/claude, openai/gpt, or google/gemini.",
    )
    parser.add_argument(
        "--stage2-quant-model",
        default=None,
        help="Override the QuantCreditAgent model in multi-LLM committee mode.",
    )
    parser.add_argument(
        "--stage2-evidence-model",
        default=None,
        help="Override the EvidenceAuditAgent model in multi-LLM committee mode.",
    )
    parser.add_argument(
        "--stage2-chair-model",
        default=None,
        help="Override the ChairReportAgent synthesis model in multi-LLM committee mode.",
    )
    parser.add_argument(
        "--strict-agno",
        action="store_true",
        help="Fail visibly instead of falling back to deterministic Stage 2 when Agno errors.",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare the operational dashboard and launch Streamlit."""
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()

    missing = missing_dashboard_artifacts(artifact_dir)
    if args.rebuild_artifacts or (missing and not args.no_auto_build):
        export_dashboard_artifacts(artifact_dir)
        missing = missing_dashboard_artifacts(artifact_dir)

    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(
            "Dashboard artifacts are not ready. "
            f"Missing file(s): {missing_text}. "
            "Run with --rebuild-artifacts or omit --no-auto-build."
        )

    launch_streamlit(
        artifact_dir=artifact_dir,
        host=args.host,
        port=args.port,
        headless=args.headless,
        stage2_runner=args.stage2_runner,
        stage2_model=args.stage2_model,
        stage2_agno_mode=args.stage2_agno_mode,
        stage2_model_provider=args.stage2_model_provider,
        stage2_quant_model=args.stage2_quant_model,
        stage2_evidence_model=args.stage2_evidence_model,
        stage2_chair_model=args.stage2_chair_model,
        strict_agno=args.strict_agno,
    )


def missing_dashboard_artifacts(artifact_dir: Path) -> list[str]:
    """Return required artifact file names that do not exist."""
    return [name for name in REQUIRED_DASHBOARD_ARTIFACTS if not (artifact_dir / name).exists()]


def export_dashboard_artifacts(artifact_dir: Path) -> None:
    """Generate dashboard artifacts through the existing export script."""
    export_script = (
        INFERENCE_2026_EXPORT_SCRIPT_PATH
        if artifact_dir.name == "feature_43_inference_2026"
        else EXPORT_SCRIPT_PATH
    )
    command = [
        sys.executable,
        str(export_script),
    ]
    if export_script == EXPORT_SCRIPT_PATH:
        command.extend(["--output-dir", str(artifact_dir)])
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(
            "Dashboard artifact export failed. "
            "Install dashboard and ML dependencies, then rerun this launcher."
        )


def launch_streamlit(
    *,
    artifact_dir: Path,
    host: str | None,
    port: int | None,
    headless: bool,
    stage2_runner: str | None,
    stage2_model: str | None,
    stage2_agno_mode: str | None,
    stage2_model_provider: str | None,
    stage2_quant_model: str | None,
    stage2_evidence_model: str | None,
    stage2_chair_model: str | None,
    strict_agno: bool,
) -> None:
    """Launch Streamlit with the prepared artifact directory."""
    try:
        from streamlit.web import cli as stcli
    except ImportError as error:
        raise SystemExit(
            "Streamlit is not installed in the current environment. "
            'Install it with: python -m pip install -e ".[dashboard]"'
        ) from error

    os.environ["CAS_DASHBOARD_ARTIFACT_DIR"] = str(artifact_dir)
    if stage2_runner:
        os.environ["CAS_STAGE2_RUNNER"] = stage2_runner
        os.environ["CAS_DASHBOARD_STAGE2_RUNNER"] = stage2_runner
    if stage2_model:
        os.environ["CAS_STAGE2_MODEL"] = stage2_model
    if stage2_agno_mode:
        os.environ["CAS_STAGE2_AGNO_MODE"] = stage2_agno_mode
    if stage2_model_provider:
        os.environ["CAS_STAGE2_MODEL_PROVIDER"] = stage2_model_provider
    if stage2_quant_model:
        os.environ["CAS_STAGE2_QUANT_MODEL"] = stage2_quant_model
    if stage2_evidence_model:
        os.environ["CAS_STAGE2_EVIDENCE_MODEL"] = stage2_evidence_model
    if stage2_chair_model:
        os.environ["CAS_STAGE2_CHAIR_MODEL"] = stage2_chair_model
    if strict_agno:
        os.environ["CAS_STAGE2_FALLBACK_ON_ERROR"] = "0"
    argv = ["streamlit", "run", str(APP_PATH)]
    if host:
        argv.extend(["--server.address", host])
    if port is not None:
        argv.extend(["--server.port", str(port)])
    if headless:
        argv.extend(["--server.headless", "true"])
    sys.argv = argv
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
