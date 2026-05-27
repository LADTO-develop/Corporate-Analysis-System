"""Run the Stage 2 feature_46/full_review_trigger_73 evaluation harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import export_stage2_rolling_validation_samples as sample_export  # noqa: E402
import run_committee_review_evaluation_batch as batch_runner  # noqa: E402
from cas.agents.stage2_evaluation_harness import (  # noqa: E402
    build_harness_report,
    provider_summary_frame,
    summarize_batch_results,
    summarize_by_category,
)
from cas.agents.stage2_policy import stage2_policy_version  # noqa: E402
from cas.agents.stage2_prompt_contracts import (  # noqa: E402
    all_stage2_prompt_contract_versions,
)
from cas.llm.model_catalog import (  # noqa: E402
    stage2_role_model_default,
    stage2_single_model_default,
)

STAGE2_DIR = ROOT / "data/outputs/modeling/feature_46_xgboost/diagnostics/stage2_agents"
DEFAULT_OUTPUT_DIR = STAGE2_DIR / "feature46_full_review_trigger_73_harness"
DEFAULT_POLICY = "feature46_full_review_trigger_73"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_PROVIDERS = ["deterministic", "openai", "gemini", "multi_role"]
FULL_SAMPLE_ROWS = 75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--providers", nargs="+", default=DEFAULT_PROVIDERS)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--eval-years", type=int, nargs="+", default=sample_export.ROLLING_EVAL_YEARS)
    parser.add_argument("--sample-per-category", type=int, default=15)
    parser.add_argument("--batch-per-category", type=int, default=15)
    parser.add_argument("--max-cases", type=int, default=FULL_SAMPLE_ROWS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retry-failed-attempts", type=int, default=1)
    parser.add_argument("--retry-failed-delay-seconds", type=float, default=1.0)
    parser.add_argument(
        "--reuse-existing-runs",
        action="store_true",
        help="Reuse existing provider result CSVs instead of calling the provider again.",
    )
    parser.add_argument(
        "--mark-skipped",
        action="append",
        default=[],
        metavar="PROVIDER=REASON",
        help="Record a provider as skipped without running it. May be passed multiple times.",
    )
    parser.add_argument(
        "--stage2-fallback-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow deterministic fallback when a live Stage 2 provider fails. "
            "Default false keeps provider failures visible in harness metrics."
        ),
    )
    parser.add_argument(
        "--stage2-llm-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use cached Stage 2 LLM responses. Default false forces live provider calls.",
    )
    parser.add_argument("--live-external-evidence", action="store_true")
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument(
        "--role-assignment-id",
        help="Stable run id suffix for a multi-role provider assignment experiment.",
    )
    parser.add_argument("--quant-provider")
    parser.add_argument("--quant-model")
    parser.add_argument("--evidence-provider")
    parser.add_argument("--evidence-model")
    parser.add_argument("--chair-provider")
    parser.add_argument("--chair-model")
    parser.add_argument(
        "--strict-provider-keys",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail instead of skipping a provider when its API key is missing.",
    )
    return parser.parse_args()


def generate_rolling_samples(output_dir: Path, args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    rolling_dir = output_dir / "rolling_samples"
    master = sample_export.read_master(sample_export.MASTER_PATH)
    frame = sample_export.build_feature_frame(master, sample_export.RAW_TS2000_PATH)
    base_columns = sample_export.read_stage1_feature_columns(sample_export.FEATURE_LIST_PATH, frame)
    scores, folds = sample_export.rolling_scores(
        frame,
        base_columns=base_columns,
        stage2_trigger_candidate_id=sample_export.STAGE2_TRIGGER_CANDIDATE_ID,
        eval_years=args.eval_years,
        seed=sample_export.DEFAULT_STAGE1_RANDOM_STATE,
    )
    scores = sample_export.attach_rating_reference(scores, sample_export.TARGET_LABEL_REFERENCE_PATH)
    scores = sample_export.add_sample_policy_flags(scores)
    samples = sample_export.build_samples(scores, args.sample_per_category)
    sample_export.write_outputs(
        scores=scores,
        samples=samples,
        folds=folds,
        output_dir=rolling_dir,
    )
    sample_counts = (
        samples.groupby(["committee_policy", "sample_category"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    sample_counts_path = rolling_dir / "stage2_harness_sample_counts.csv"
    sample_counts.to_csv(sample_counts_path, index=False, encoding="utf-8-sig")
    summary = {
        "sample_rows": len(samples),
        "score_rows": len(scores),
        "policy": args.policy,
        "stage2_policy_version": stage2_policy_version(),
        "prompt_contract_versions": all_stage2_prompt_contract_versions(),
        "eval_years": ", ".join(str(year) for year in args.eval_years),
        "sample_counts_path": _relative(sample_counts_path),
    }
    return rolling_dir / "committee_review_rolling_validation_tuning_samples.csv", summary


def run_provider(
    *,
    provider: str,
    samples_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, pd.DataFrame, dict[str, str] | None]:
    normalized = _canonical_provider(provider)
    run_id = _run_id(normalized, args)
    runtime = _runtime_selection(normalized, args)
    run_dir = output_dir / "runs" / run_id
    result_path = run_dir / "committee_review_batch_results.csv"
    if args.reuse_existing_runs and result_path.exists():
        results = pd.read_csv(result_path)
        summary = summarize_batch_results(
            results,
            run_id=run_id,
            runner=runtime["stage2_runner"],
            provider=runtime["summary_provider"],
            model=runtime["summary_model"],
            result_path=_relative(result_path),
            stage2_policy_version=stage2_policy_version(),
        )
        category_summary = summarize_by_category(results, run_id=run_id)
        return summary, category_summary, None

    if normalized in {"openai", "gemini", "google"} and not _provider_has_key(normalized):
        skipped = {
            "run_id": run_id,
            "provider": normalized,
            "reason": "API key not found",
        }
        if args.strict_provider_keys:
            raise RuntimeError(f"Provider key missing for {normalized}: {skipped}")
        return None, pd.DataFrame(), skipped
    if normalized in {"multi", "multi_role", "multi-role", "multi_llm", "multi_llm_committee"}:
        missing_keys = _missing_multi_role_keys()
        if missing_keys:
            skipped = {
                "run_id": run_id,
                "provider": "multi_role",
                "reason": "API key not found: " + ", ".join(missing_keys),
            }
            if args.strict_provider_keys:
                raise RuntimeError(f"Provider key missing for multi_role: {skipped}")
            return None, pd.DataFrame(), skipped

    samples = batch_runner.read_samples(samples_path)
    batch = batch_runner.select_batch(
        samples,
        policy=args.policy,
        per_category=args.batch_per_category,
        max_cases=args.max_cases,
    )
    runtime_config = batch_runner.configure_runtime(
        live_external_evidence=args.live_external_evidence,
        stage2_runner=runtime["stage2_runner"],
        stage2_agno_mode=runtime["stage2_agno_mode"],
        stage2_model_provider=runtime["runtime_model_provider"],
        stage2_model=runtime["runtime_model"],
        stage2_llm_cache=args.stage2_llm_cache,
    )
    runtime_config = replace(
        runtime_config,
        fallback_on_error=bool(args.stage2_fallback_on_error),
        review_qa_fallback_on_error=bool(args.stage2_fallback_on_error),
        risk_recall_qa_fallback_on_error=bool(args.stage2_fallback_on_error),
        **_role_override_fields(normalized, args),
    )
    results = batch_runner.run_batch(
        batch,
        use_sample_model_view=True,
        workers=args.workers,
        runtime_config=runtime_config,
    )
    results, retry_reports = batch_runner.retry_failed_cases(
        batch,
        results,
        use_sample_model_view=True,
        attempts=args.retry_failed_attempts,
        workers=1,
        delay_seconds=args.retry_failed_delay_seconds,
        output_dir=run_dir,
        write_artifacts=True,
        runtime_config=runtime_config,
    )
    batch_runner.write_outputs(results, output_dir=run_dir, retry_reports=retry_reports)
    summary = summarize_batch_results(
        results,
        run_id=run_id,
        runner=runtime["stage2_runner"],
        provider=runtime["summary_provider"],
        model=runtime["summary_model"],
        result_path=_relative(result_path),
        stage2_policy_version=stage2_policy_version(),
    )
    category_summary = summarize_by_category(results, run_id=run_id)
    return summary, category_summary, None


def write_harness_outputs(
    *,
    output_dir: Path,
    provider_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    sample_summary: dict[str, Any],
    skipped_runs: list[dict[str, str]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    provider_summary_path = output_dir / "stage2_provider_summary.csv"
    category_summary_path = output_dir / "stage2_category_summary.csv"
    summary_path = output_dir / "stage2_harness_summary.json"
    report_path = output_dir / "stage2_feature46_full_review_trigger_73_report.md"
    provider_summary.to_csv(provider_summary_path, index=False, encoding="utf-8-sig")
    category_summary.to_csv(category_summary_path, index=False, encoding="utf-8-sig")
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_version": "feature_46_xgboost",
        "stage2_trigger": "full_review_trigger_73",
        "stage2_policy_version": stage2_policy_version(),
        "prompt_contract_versions": all_stage2_prompt_contract_versions(),
        "sample_summary": sample_summary,
        "providers": provider_summary.to_dict("records"),
        "skipped_runs": skipped_runs,
        "outputs": {
            "provider_summary": _relative(provider_summary_path),
            "category_summary": _relative(category_summary_path),
            "report": _relative(report_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(
        build_harness_report(
            provider_summary=provider_summary,
            category_summary=category_summary,
            sample_summary=sample_summary,
            skipped_runs=skipped_runs,
            output_dir=output_dir,
            stage2_policy_version=stage2_policy_version(),
            prompt_contract_versions=all_stage2_prompt_contract_versions(),
        ),
        encoding="utf-8",
    )
    return {
        "provider_summary": _relative(provider_summary_path),
        "category_summary": _relative(category_summary_path),
        "summary": _relative(summary_path),
        "report": _relative(report_path),
    }


def _runtime_selection(provider: str, args: argparse.Namespace) -> dict[str, str]:
    if provider == "deterministic":
        return {
            "stage2_runner": "deterministic",
            "stage2_agno_mode": "single",
            "runtime_model_provider": "deterministic",
            "runtime_model": "deterministic",
            "summary_provider": "deterministic",
            "summary_model": "deterministic",
        }
    if provider == "openai":
        return {
            "stage2_runner": "agno",
            "stage2_agno_mode": "single",
            "runtime_model_provider": "openai",
            "runtime_model": args.openai_model,
            "summary_provider": "openai",
            "summary_model": args.openai_model,
        }
    if provider in {"gemini", "google"}:
        return {
            "stage2_runner": "agno",
            "stage2_agno_mode": "single",
            "runtime_model_provider": "google",
            "runtime_model": args.gemini_model,
            "summary_provider": "google",
            "summary_model": args.gemini_model,
        }
    if provider in {"multi", "multi_role", "multi-role", "multi_llm", "multi_llm_committee"}:
        single_default = stage2_single_model_default()
        return {
            "stage2_runner": "agno",
            "stage2_agno_mode": "multi_llm_committee",
            "runtime_model_provider": single_default.provider,
            "runtime_model": single_default.model,
            "summary_provider": "multi_role",
            "summary_model": _multi_role_model_label(args),
        }
    raise ValueError(f"Unsupported provider: {provider}")


def _canonical_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "google":
        return "gemini"
    if normalized in {"multi", "multi-role", "multi_llm", "multi_llm_committee"}:
        return "multi_role"
    return normalized


def _provider_has_key(provider: str) -> bool:
    env = _provider_env()
    if provider == "openai":
        return bool(env.get("OPENAI_API_KEY"))
    if provider in {"gemini", "google"}:
        return bool(env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY"))
    return True


def _missing_multi_role_keys() -> list[str]:
    env = _provider_env()
    missing: list[str] = []
    if not env.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not env.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not (env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY")):
        missing.append("GOOGLE_API_KEY/GEMINI_API_KEY")
    return missing


def _provider_env() -> dict[str, str]:
    env = {key: value for key, value in dotenv_values(ROOT / ".env").items() if value}
    env.update({key: value for key, value in os.environ.items() if value})
    return env


def _multi_role_model_label(args: argparse.Namespace) -> str:
    roles = ("quant_credit", "evidence_audit", "chair_report")
    parts = []
    for role in roles:
        provider, model = _role_provider_model(role, args)
        parts.append(f"{role}={provider}:{model}")
    return "; ".join(parts)


def _run_id(provider: str, args: argparse.Namespace) -> str:
    if provider == "deterministic":
        return "deterministic"
    if provider == "openai":
        return f"openai_{_slug(args.openai_model)}"
    if provider in {"gemini", "google"}:
        return f"gemini_{_slug(args.gemini_model)}"
    if provider in {"multi", "multi_role", "multi-role", "multi_llm", "multi_llm_committee"}:
        if args.role_assignment_id:
            return f"multi_role_{_slug(args.role_assignment_id)}"
        return "multi_role_catalog_defaults"
    return _slug(provider)


def _role_override_fields(provider: str, args: argparse.Namespace) -> dict[str, str | None]:
    if provider != "multi_role":
        return {}
    return {
        "quant_provider": args.quant_provider,
        "quant_model": args.quant_model,
        "evidence_provider": args.evidence_provider,
        "evidence_model": args.evidence_model,
        "chair_provider": args.chair_provider,
        "chair_model": args.chair_model,
    }


def _role_provider_model(role: str, args: argparse.Namespace) -> tuple[str, str]:
    default = stage2_role_model_default(role)
    provider_attr = {
        "quant_credit": "quant_provider",
        "evidence_audit": "evidence_provider",
        "chair_report": "chair_provider",
    }[role]
    model_attr = {
        "quant_credit": "quant_model",
        "evidence_audit": "evidence_model",
        "chair_report": "chair_model",
    }[role]
    return (
        getattr(args, provider_attr) or default.provider,
        getattr(args, model_attr) or default.model,
    )


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    samples_path, sample_summary = generate_rolling_samples(output_dir, args)
    marked_skips = _parse_marked_skips(args.mark_skipped)
    summaries: list[dict[str, Any]] = []
    category_frames: list[pd.DataFrame] = []
    skipped_runs: list[dict[str, str]] = []
    for provider in args.providers:
        normalized = _canonical_provider(provider)
        if normalized in marked_skips:
            skipped = _skipped_run(normalized, args, marked_skips[normalized])
            skipped_runs.append(skipped)
            summaries.append(_skipped_provider_summary(skipped, args))
            continue
        summary, category_summary, skipped = run_provider(
            provider=provider,
            samples_path=samples_path,
            output_dir=output_dir,
            args=args,
        )
        if skipped is not None:
            skipped_runs.append(skipped)
            summaries.append(_skipped_provider_summary(skipped, args))
            continue
        if summary is not None:
            summaries.append(summary)
        if not category_summary.empty:
            category_frames.append(category_summary)
    provider_summary = provider_summary_frame(summaries)
    category_summary = (
        pd.concat(category_frames, ignore_index=True) if category_frames else pd.DataFrame()
    )
    outputs = write_harness_outputs(
        output_dir=output_dir,
        provider_summary=provider_summary,
        category_summary=category_summary,
        sample_summary=sample_summary,
        skipped_runs=skipped_runs,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


def _parse_marked_skips(values: list[str]) -> dict[str, str]:
    skips: dict[str, str] = {}
    for value in values:
        provider, sep, reason = value.partition("=")
        if not sep:
            raise ValueError(f"--mark-skipped must use PROVIDER=REASON: {value}")
        skips[_canonical_provider(provider)] = reason.strip() or "manually skipped"
    return skips


def _skipped_run(provider: str, args: argparse.Namespace, reason: str) -> dict[str, str]:
    return {
        "run_id": _run_id(provider, args),
        "provider": provider,
        "reason": reason,
    }


def _skipped_provider_summary(skipped: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    runtime = _runtime_selection(skipped["provider"], args)
    return {
        "run_id": skipped["run_id"],
        "runner": runtime["stage2_runner"],
        "provider": runtime["summary_provider"],
        "model": runtime["summary_model"],
        "run_status": "skipped",
        "skip_reason": skipped["reason"],
        "rows": 0,
        "stage2_policy_version": stage2_policy_version(),
    }


if __name__ == "__main__":
    main()
