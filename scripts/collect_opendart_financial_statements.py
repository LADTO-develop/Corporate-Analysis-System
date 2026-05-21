from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INFERENCE_PATH = (
    ROOT / "data" / "input" / "credit_43_features" / "feature_43_inference_2026.csv"
)
DEFAULT_MODEL_V1_PATH = (
    ROOT / "data" / "raw" / "ts2000" / "TS2000_Credit_Model_Dataset_Model_V1.csv"
)
DEFAULT_CORP_CODE_PATH = ROOT / "data" / "external" / "opendart" / "corp_codes.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "opendart"
OPENDART_FINANCIAL_URL = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"

KEY_ACCOUNT_CANDIDATES = {
    "assets_total": {
        "statement": "BS",
        "account_ids": {"ifrs-full_Assets"},
        "account_names": {"자산총계"},
    },
    "liabilities_total": {
        "statement": "BS",
        "account_ids": {"ifrs-full_Liabilities"},
        "account_names": {"부채총계"},
    },
    "equity_total": {
        "statement": "BS",
        "account_ids": {"ifrs-full_Equity"},
        "account_names": {"자본총계"},
    },
    "current_assets": {
        "statement": "BS",
        "account_ids": {"ifrs-full_CurrentAssets"},
        "account_names": {"유동자산"},
    },
    "current_liabilities": {
        "statement": "BS",
        "account_ids": {"ifrs-full_CurrentLiabilities"},
        "account_names": {"유동부채"},
    },
    "cash_and_equivalents": {
        "statement": "BS",
        "account_ids": {"ifrs-full_CashAndCashEquivalents"},
        "account_names": {"현금및현금성자산", "현금 및 현금성자산"},
    },
    "revenue": {
        "statement": "IS",
        "account_ids": {"ifrs-full_Revenue"},
        "account_names": {"매출액", "수익(매출액)", "영업수익"},
    },
    "cost_of_sales": {
        "statement": "IS",
        "account_ids": {"ifrs-full_CostOfSales"},
        "account_names": {"매출원가", "영업비용"},
    },
    "gross_profit": {
        "statement": "IS",
        "account_ids": {"ifrs-full_GrossProfit"},
        "account_names": {"매출총이익", "매출총이익(손실)"},
    },
    "operating_income": {
        "statement": "IS",
        "account_ids": {
            "dart_OperatingIncomeLoss",
            "ifrs-full_ProfitLossFromOperatingActivities",
        },
        "account_names": {"영업이익", "영업이익(손실)"},
    },
    "pretax_income": {
        "statement": "IS",
        "account_ids": {"ifrs-full_ProfitLossBeforeTax"},
        "account_names": {"법인세비용차감전순이익", "법인세비용차감전순이익(손실)"},
    },
    "net_income": {
        "statement": "IS",
        "account_ids": {"ifrs-full_ProfitLoss"},
        "account_names": {"당기순이익", "당기순이익(손실)"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect OpenDART annual financial statements for inference rows whose "
            "TS2000 financial statement source is missing. CFS is used by default."
        )
    )
    parser.add_argument("--inference", type=Path, default=DEFAULT_INFERENCE_PATH)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--source-kind", choices=["inference", "model-v1"], default="inference")
    parser.add_argument("--corp-codes", type=Path, default=DEFAULT_CORP_CODE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-fiscal-year", type=int, default=2025)
    parser.add_argument("--all-years", action="store_true")
    parser.add_argument(
        "--opendart-bsns-year",
        type=int,
        default=None,
        help="Override the OpenDART bsns_year while keeping the selected inference rows.",
    )
    parser.add_argument("--reprt-code", default="11011", help="11011 means annual report.")
    parser.add_argument("--fs-div", default="CFS", choices=["CFS", "OFS"])
    parser.add_argument(
        "--fallback-ofs",
        action="store_true",
        help="If CFS has no OpenDART data, collect OFS and mark the actual fs_div used.",
    )
    parser.add_argument("--stock-code", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--include-all", action="store_true")
    return parser.parse_args()


def normalize_stock_code(value: object) -> str:
    text = str(value).replace("\ufeff", "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6) if text else ""


def parse_amount(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "nan", "None"}:
        return None
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"
    try:
        return float(text)
    except ValueError:
        return None


def load_inference_targets(args: argparse.Namespace) -> pd.DataFrame:
    source_path = args.source
    if source_path is None:
        source_path = DEFAULT_MODEL_V1_PATH if args.source_kind == "model-v1" else args.inference
    source = pd.read_csv(source_path, encoding="utf-8-sig", dtype={"stock_code": "string"})
    source["stock_code"] = source["stock_code"].map(normalize_stock_code)
    source["fiscal_year"] = pd.to_numeric(source["fiscal_year"], errors="coerce").astype("Int64")
    if not args.all_years:
        source = source.loc[source["fiscal_year"].eq(args.target_fiscal_year)].copy()

    if args.stock_code:
        requested = {normalize_stock_code(code) for code in args.stock_code}
        source = source.loc[source["stock_code"].isin(requested)].copy()

    if not args.include_all:
        source = source.loc[missing_financial_statement_source(source)].copy()

    if args.limit is not None:
        source = source.head(args.limit).copy()
    return source


def missing_financial_statement_source(frame: pd.DataFrame) -> pd.Series:
    assets_zero = pd.to_numeric(frame.get("assets_total"), errors="coerce").fillna(0).eq(0)
    gross_profit_zero = pd.to_numeric(frame.get("gross_profit"), errors="coerce").fillna(0).eq(0)
    current_missing = pd.to_numeric(frame.get("current_ratio"), errors="coerce").isna()
    cash_missing = pd.to_numeric(frame.get("cash_ratio"), errors="coerce").isna()
    net_margin_missing = pd.to_numeric(frame.get("net_margin"), errors="coerce").isna()
    return assets_zero & gross_profit_zero & current_missing & cash_missing & net_margin_missing


def load_corp_code_map(path: Path) -> dict[str, dict[str, str]]:
    corp_codes = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": "string"})
    corp_codes["stock_code"] = corp_codes["stock_code"].map(normalize_stock_code)
    output: dict[str, dict[str, str]] = {}
    for record in corp_codes.to_dict(orient="records"):
        stock_code = str(record.get("stock_code") or "")
        if stock_code:
            output[stock_code] = {
                "corp_code": str(record.get("corp_code") or "").zfill(8),
                "opendart_corp_name": str(record.get("corp_name") or ""),
            }
    return output


def collect_financial_rows(
    *,
    api_key: str,
    corp_code: str,
    bsns_year: int,
    reprt_code: str,
    fs_div: str,
) -> tuple[str, str, list[dict[str, Any]]]:
    response = requests.get(
        OPENDART_FINANCIAL_URL,
        params={
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "bsns_year": str(bsns_year),
            "reprt_code": reprt_code,
            "fs_div": fs_div,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    status = str(payload.get("status") or "")
    message = str(payload.get("message") or "")
    rows = payload.get("list") if isinstance(payload.get("list"), list) else []
    return status, message, rows


def summarize_key_accounts(raw_rows: list[dict[str, Any]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for metric, spec in KEY_ACCOUNT_CANDIDATES.items():
        matched = select_account_row(
            raw_rows,
            statement=str(spec["statement"]),
            account_ids=set(spec["account_ids"]),
            account_names=set(spec["account_names"]),
        )
        summary[metric] = parse_amount(matched.get("thstrm_amount")) if matched else None
        summary[f"{metric}_account_nm"] = matched.get("account_nm") if matched else ""
        summary[f"{metric}_account_id"] = matched.get("account_id") if matched else ""
    return summary


def select_account_row(
    rows: list[dict[str, Any]],
    *,
    statement: str,
    account_ids: set[str],
    account_names: set[str],
) -> dict[str, Any] | None:
    statement_rows = [row for row in rows if str(row.get("sj_div") or "") == statement]
    for row in statement_rows:
        if str(row.get("account_id") or "") in account_ids:
            return row
    for row in statement_rows:
        if str(row.get("account_nm") or "").strip() in account_names:
            return row
    for row in rows:
        if str(row.get("account_id") or "") in account_ids:
            return row
    for row in rows:
        if str(row.get("account_nm") or "").strip() in account_names:
            return row
    return None


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    api_key = os.environ.get("OPENDART_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENDART_API_KEY is required in .env or the environment.")

    targets = load_inference_targets(args)
    corp_code_map = load_corp_code_map(args.corp_codes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []

    for index, target in enumerate(targets.to_dict(orient="records"), start=1):
        stock_code = normalize_stock_code(target["stock_code"])
        corp_meta = corp_code_map.get(stock_code, {})
        corp_code = corp_meta.get("corp_code", "")
        base = {
            "market": target.get("market"),
            "stock_code": stock_code,
            "corp_name": target.get("corp_name"),
            "fiscal_year": int(target.get("fiscal_year")),
            "eval_year": target.get("eval_year"),
            "opendart_bsns_year": args.opendart_bsns_year or int(target.get("fiscal_year")),
            "corp_code": corp_code,
            "opendart_corp_name": corp_meta.get("opendart_corp_name", ""),
            "reprt_code": args.reprt_code,
            "fs_div_requested": args.fs_div,
            "fs_div": args.fs_div,
        }
        if not corp_code:
            summary_records.append({**base, "opendart_status": "missing_corp_code"})
            continue

        status, message, rows = collect_financial_rows(
            api_key=api_key,
            corp_code=corp_code,
            bsns_year=int(base["opendart_bsns_year"]),
            reprt_code=args.reprt_code,
            fs_div=args.fs_div,
        )
        if args.fallback_ofs and args.fs_div == "CFS" and (status != "000" or not rows):
            fallback_status, fallback_message, fallback_rows = collect_financial_rows(
                api_key=api_key,
                corp_code=corp_code,
                bsns_year=int(base["opendart_bsns_year"]),
                reprt_code=args.reprt_code,
                fs_div="OFS",
            )
            if fallback_status == "000" and fallback_rows:
                status = fallback_status
                message = f"{fallback_message} (CFS unavailable, OFS fallback used)"
                rows = fallback_rows
                base["fs_div"] = "OFS"
        for row in rows:
            raw_records.append(
                {
                    **base,
                    "opendart_status": status,
                    "opendart_message": message,
                    "sj_div": row.get("sj_div"),
                    "sj_nm": row.get("sj_nm"),
                    "account_id": row.get("account_id"),
                    "account_nm": row.get("account_nm"),
                    "account_detail": row.get("account_detail"),
                    "thstrm_nm": row.get("thstrm_nm"),
                    "thstrm_amount": row.get("thstrm_amount"),
                    "frmtrm_nm": row.get("frmtrm_nm"),
                    "frmtrm_amount": row.get("frmtrm_amount"),
                    "ord": row.get("ord"),
                }
            )
        summary_records.append(
            {
                **base,
                "opendart_status": status,
                "opendart_message": message,
                "raw_account_rows": len(rows),
                **summarize_key_accounts(rows),
            }
        )
        print(
            f"[{index:>4}/{len(targets):>4}] {stock_code} {target.get('corp_name')} "
            f"status={status} rows={len(rows)}"
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    suffix_year = (
        "all_years"
        if args.all_years and args.opendart_bsns_year is None
        else str(args.opendart_bsns_year or args.target_fiscal_year)
    )
    suffix_fs_div = (
        f"{args.fs_div.lower()}_with_ofs_fallback"
        if args.fallback_ofs and args.fs_div == "CFS"
        else args.fs_div.lower()
    )
    suffix = f"{args.source_kind}_{suffix_year}_{suffix_fs_div}"
    raw_path = args.output_dir / f"financial_statements_{suffix}_raw.csv"
    summary_path = args.output_dir / f"financial_statements_{suffix}_summary.csv"
    pd.DataFrame(raw_records).to_csv(raw_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_records).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {raw_path} ({len(raw_records):,} rows)")
    print(f"[Saved] {summary_path} ({len(summary_records):,} rows)")


if __name__ == "__main__":
    main()
