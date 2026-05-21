from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_V1_PATH = ROOT / "data" / "raw" / "ts2000" / "TS2000_Credit_Model_Dataset_Model_V1.csv"
DEFAULT_RAW_SUPPLEMENT_PATH = (
    ROOT
    / "data"
    / "raw"
    / "opendart"
    / "financial_statements_model-v1_all_years_cfs_with_ofs_fallback_raw.csv"
)
DEFAULT_AUDIT_PATH = ROOT / "data" / "raw" / "opendart" / "model_v1_opendart_supplement_audit.csv"
INF_CAP = 1_000_000.0
CV_CAP = 10.0
KEY_COLUMNS = ["market", "stock_code", "fiscal_year"]
MATCH_COLUMNS = ["market", "_stock_code_key", "fiscal_year"]

BASE_STATEMENT_COLUMNS = [
    "assets_total",
    "liabilities_total",
    "equity_total",
    "current_assets",
    "current_liabilities",
    "noncurrent_liabilities",
    "cash_and_equivalents",
    "accounts_receivable",
    "inventories",
    "contract_assets",
    "short_term_borrowings",
    "long_term_borrowings",
    "bonds_payable",
    "accounts_payable",
    "property_plant_equipment",
    "intangible_assets",
    "advances_from_customers",
    "capital_stock",
    "revenue",
    "cost_of_sales",
    "gross_profit",
    "operating_income",
    "pretax_income",
    "net_income",
    "interest_expense",
    "ocf",
    "icf",
    "cff",
    "net_change_in_cash",
    "depreciation",
    "amortization_dev_cost",
    "amortization_other_intangibles",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply OpenDART financial statement supplements to TS2000 Model_V1 and "
            "recompute financial ratios/trend columns."
        )
    )
    parser.add_argument("--source", type=Path, default=MODEL_V1_PATH)
    parser.add_argument("--raw-supplement", type=Path, default=DEFAULT_RAW_SUPPLEMENT_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_V1_PATH)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_stock_code(value: object) -> str:
    text = str(value).replace("\ufeff", "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6) if text else ""


def parse_amount_won(value: object) -> float | None:
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


def won_to_thousand(value: float | None) -> float | None:
    return None if value is None else value / 1000.0


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid = numerator.notna() & denominator.notna() & denominator.ne(0)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def capped_ratio(numerator: pd.Series, denominator: pd.Series, cap: float = INF_CAP) -> pd.Series:
    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid = numerator.notna() & denominator.notna() & denominator.ne(0)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    zero_denominator = denominator.notna() & denominator.eq(0)
    result.loc[zero_denominator] = cap
    return result


def growth_ratio(current: pd.Series, previous: pd.Series, *, abs_base: bool = False) -> pd.Series:
    base = previous.abs() if abs_base else previous
    result = pd.Series(np.nan, index=current.index, dtype=float)
    valid = current.notna() & previous.notna() & base.notna() & base.ne(0)
    result.loc[valid] = (current.loc[valid] - previous.loc[valid]) / base.loc[valid]
    return result


def missing_financial_statement_source(frame: pd.DataFrame) -> pd.Series:
    assets_zero = pd.to_numeric(frame.get("assets_total"), errors="coerce").fillna(0).eq(0)
    gross_profit_zero = pd.to_numeric(frame.get("gross_profit"), errors="coerce").fillna(0).eq(0)
    current_missing = pd.to_numeric(frame.get("current_ratio"), errors="coerce").isna()
    cash_missing = pd.to_numeric(frame.get("cash_ratio"), errors="coerce").isna()
    net_margin_missing = pd.to_numeric(frame.get("net_margin"), errors="coerce").isna()
    return assets_zero & gross_profit_zero & current_missing & cash_missing & net_margin_missing


def load_supplement_rows(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": "string"})
    if raw.empty:
        return raw
    raw["stock_code"] = raw["stock_code"].map(normalize_stock_code)
    raw["fiscal_year"] = pd.to_numeric(raw["fiscal_year"], errors="coerce").astype("Int64")
    raw["thstrm_amount_number"] = raw["thstrm_amount"].map(parse_amount_won)
    status = raw["opendart_status"].astype(str).str.strip().str.zfill(3)
    return raw.loc[status.eq("000")].copy()


def first_amount(
    rows: pd.DataFrame,
    *,
    statements: Iterable[str],
    account_ids: Iterable[str] = (),
    account_names: Iterable[str] = (),
) -> float | None:
    if rows.empty:
        return None
    subset = rows.loc[rows["sj_div"].astype(str).isin(set(statements))].copy()
    if subset.empty:
        subset = rows.copy()
    id_set = set(account_ids)
    if id_set:
        id_match = subset.loc[subset["account_id"].astype(str).isin(id_set)]
        if not id_match.empty:
            return parse_amount_won(id_match.iloc[0]["thstrm_amount"])
    name_set = set(account_names)
    if name_set:
        name_match = subset.loc[subset["account_nm"].astype(str).str.strip().isin(name_set)]
        if not name_match.empty:
            return parse_amount_won(name_match.iloc[0]["thstrm_amount"])
    return None


def sum_amount_by_name(
    rows: pd.DataFrame,
    *,
    statements: Iterable[str],
    include_terms: Iterable[str],
    exclude_terms: Iterable[str] = (),
) -> float | None:
    if rows.empty:
        return None
    subset = rows.loc[rows["sj_div"].astype(str).isin(set(statements))].copy()
    if subset.empty:
        return None
    names = subset["account_nm"].astype(str)
    include_mask = pd.Series(False, index=subset.index)
    for term in include_terms:
        include_mask = include_mask | names.str.contains(term, regex=False, na=False)
    exclude_mask = pd.Series(False, index=subset.index)
    for term in exclude_terms:
        exclude_mask = exclude_mask | names.str.contains(term, regex=False, na=False)
    matched = subset.loc[include_mask & ~exclude_mask]
    if matched.empty:
        return None
    values = matched["thstrm_amount_number"].dropna()
    if values.empty:
        return None
    return float(values.sum())


def extract_statement_values(rows: pd.DataFrame) -> dict[str, float | None]:
    values_won: dict[str, float | None] = {
        "assets_total": first_amount(
            rows,
            statements=["BS"],
            account_ids=["ifrs-full_Assets"],
            account_names=["자산총계"],
        ),
        "liabilities_total": first_amount(
            rows,
            statements=["BS"],
            account_ids=["ifrs-full_Liabilities"],
            account_names=["부채총계"],
        ),
        "equity_total": first_amount(
            rows,
            statements=["BS", "SCE"],
            account_ids=["ifrs-full_Equity"],
            account_names=["자본총계"],
        ),
        "current_assets": first_amount(
            rows,
            statements=["BS"],
            account_ids=["ifrs-full_CurrentAssets"],
            account_names=["유동자산"],
        ),
        "current_liabilities": first_amount(
            rows,
            statements=["BS"],
            account_ids=["ifrs-full_CurrentLiabilities"],
            account_names=["유동부채"],
        ),
        "noncurrent_liabilities": first_amount(
            rows,
            statements=["BS"],
            account_ids=["ifrs-full_NoncurrentLiabilities"],
            account_names=["비유동부채"],
        ),
        "cash_and_equivalents": first_amount(
            rows,
            statements=["BS"],
            account_ids=["ifrs-full_CashAndCashEquivalents"],
            account_names=["현금및현금성자산", "현금 및 현금성자산"],
        ),
        "accounts_receivable": first_amount(
            rows,
            statements=["BS"],
            account_ids=["dart_ShortTermTradeReceivable"],
            account_names=["매출채권"],
        ),
        "inventories": first_amount(
            rows,
            statements=["BS"],
            account_ids=["ifrs-full_Inventories"],
            account_names=["재고자산"],
        ),
        "contract_assets": first_amount(
            rows,
            statements=["BS"],
            account_names=["계약자산", "유동계약자산"],
        ),
        "accounts_payable": first_amount(
            rows,
            statements=["BS"],
            account_ids=["dart_ShortTermTradePayables"],
            account_names=["매입채무"],
        ),
        "property_plant_equipment": first_amount(
            rows,
            statements=["BS"],
            account_ids=["ifrs-full_PropertyPlantAndEquipment"],
            account_names=["유형자산"],
        ),
        "intangible_assets": first_amount(
            rows,
            statements=["BS"],
            account_ids=["ifrs-full_IntangibleAssetsOtherThanGoodwill"],
            account_names=["무형자산"],
        ),
        "advances_from_customers": first_amount(
            rows,
            statements=["BS"],
            account_ids=["dart_ShortTermAdvancesCustomers"],
            account_names=["선수금"],
        ),
        "capital_stock": first_amount(
            rows,
            statements=["BS", "SCE"],
            account_ids=["dart_IssuedCapitalOfCommonStock"],
            account_names=["자본금"],
        ),
        "revenue": first_amount(
            rows,
            statements=["IS", "CIS"],
            account_ids=["ifrs-full_Revenue"],
            account_names=["매출액", "수익(매출액)", "영업수익"],
        ),
        "cost_of_sales": first_amount(
            rows,
            statements=["IS", "CIS"],
            account_ids=["ifrs-full_CostOfSales"],
            account_names=["매출원가", "영업비용"],
        ),
        "gross_profit": first_amount(
            rows,
            statements=["IS", "CIS"],
            account_ids=["ifrs-full_GrossProfit"],
            account_names=["매출총이익", "매출총이익(손실)"],
        ),
        "operating_income": first_amount(
            rows,
            statements=["IS", "CIS"],
            account_ids=["dart_OperatingIncomeLoss", "ifrs-full_ProfitLossFromOperatingActivities"],
            account_names=["영업이익", "영업이익(손실)"],
        ),
        "pretax_income": first_amount(
            rows,
            statements=["IS", "CIS"],
            account_ids=["ifrs-full_ProfitLossBeforeTax"],
            account_names=["법인세비용차감전순이익", "법인세비용차감전순이익(손실)"],
        ),
        "net_income": first_amount(
            rows,
            statements=["IS", "CIS", "CF"],
            account_ids=["ifrs-full_ProfitLoss"],
            account_names=["당기순이익", "당기순이익(손실)"],
        ),
        "interest_expense": first_amount(
            rows,
            statements=["CF", "IS", "CIS"],
            account_ids=["dart_AdjustmentsForInterestExpenses", "ifrs-full_FinanceCosts"],
            account_names=["이자비용", "금융비용"],
        ),
        "ocf": first_amount(
            rows,
            statements=["CF"],
            account_ids=["ifrs-full_CashFlowsFromUsedInOperatingActivities"],
            account_names=["영업활동현금흐름"],
        ),
        "icf": first_amount(
            rows,
            statements=["CF"],
            account_ids=["ifrs-full_CashFlowsFromUsedInInvestingActivities"],
            account_names=["투자활동현금흐름"],
        ),
        "cff": first_amount(
            rows,
            statements=["CF"],
            account_ids=["ifrs-full_CashFlowsFromUsedInFinancingActivities"],
            account_names=["재무활동현금흐름"],
        ),
        "net_change_in_cash": first_amount(
            rows,
            statements=["CF"],
            account_ids=["ifrs-full_IncreaseDecreaseInCashAndCashEquivalents"],
            account_names=["현금및현금성자산의순증가(감소)"],
        ),
        "depreciation": first_amount(
            rows,
            statements=["CF"],
            account_ids=["ifrs-full_AdjustmentsForDepreciationExpense"],
            account_names=["감가상각비"],
        ),
        "amortization_dev_cost": first_amount(
            rows,
            statements=["CF"],
            account_names=["개발비상각비"],
        ),
        "amortization_other_intangibles": first_amount(
            rows,
            statements=["CF"],
            account_ids=["ifrs-full_AdjustmentsForAmortisationExpense"],
            account_names=["무형자산상각비"],
        ),
    }
    values_won["short_term_borrowings"] = sum_amount_by_name(
        rows,
        statements=["BS"],
        include_terms=["단기차입금", "유동성장기차입금", "유동성사채", "유동성전환사채"],
        exclude_terms=["상환", "발행"],
    )
    values_won["long_term_borrowings"] = sum_amount_by_name(
        rows,
        statements=["BS"],
        include_terms=["장기차입금"],
        exclude_terms=["대여금", "유동성"],
    )
    values_won["bonds_payable"] = sum_amount_by_name(
        rows,
        statements=["BS"],
        include_terms=["사채", "전환사채", "신주인수권부사채"],
        exclude_terms=["유동성", "상환", "발행"],
    )
    return {column: won_to_thousand(value) for column, value in values_won.items()}


def build_supplement_frame(raw: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if raw.empty:
        return pd.DataFrame(columns=[*KEY_COLUMNS, "fs_div", *BASE_STATEMENT_COLUMNS])
    group_columns = ["market", "stock_code", "fiscal_year"]
    for keys, group in raw.groupby(group_columns, dropna=False):
        market, stock_code, fiscal_year = keys
        extracted = extract_statement_values(group)
        records.append(
            {
                "market": market,
                "stock_code": normalize_stock_code(stock_code),
                "fiscal_year": int(fiscal_year),
                "corp_name": group["corp_name"].dropna().iloc[0]
                if group["corp_name"].notna().any()
                else "",
                "corp_code": group["corp_code"].dropna().iloc[0]
                if group["corp_code"].notna().any()
                else "",
                "fs_div": group["fs_div"].dropna().iloc[0] if group["fs_div"].notna().any() else "",
                "fs_div_requested": group["fs_div_requested"].dropna().iloc[0]
                if "fs_div_requested" in group and group["fs_div_requested"].notna().any()
                else "",
                "raw_account_rows": len(group),
                **extracted,
            }
        )
    return pd.DataFrame(records)


def apply_supplements(
    model_v1: pd.DataFrame, supplements: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = model_v1.copy()
    output["_stock_code_key"] = output["stock_code"].map(normalize_stock_code)
    output["fiscal_year"] = pd.to_numeric(output["fiscal_year"], errors="coerce").astype(int)
    output = output.sort_values(["market", "_stock_code_key", "fiscal_year"]).reset_index(drop=True)
    for column in BASE_STATEMENT_COLUMNS:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce").astype("float64")
    missing_mask = missing_financial_statement_source(output)

    supplements = supplements.copy()
    supplements["_stock_code_key"] = supplements["stock_code"].map(normalize_stock_code)
    supplement_lookup = supplements.set_index(MATCH_COLUMNS, drop=False)
    audit_records: list[dict[str, object]] = []
    for index, row in output.loc[missing_mask].iterrows():
        key = (row["market"], row["_stock_code_key"], row["fiscal_year"])
        if key not in supplement_lookup.index:
            continue
        supplement = supplement_lookup.loc[key]
        if isinstance(supplement, pd.DataFrame):
            supplement = supplement.iloc[0]
        changed_columns: list[str] = []
        for column in BASE_STATEMENT_COLUMNS:
            value = supplement.get(column)
            if value is None or pd.isna(value):
                continue
            if column not in output.columns:
                continue
            old_value = output.at[index, column]
            output.at[index, column] = value
            if pd.isna(old_value) or float(old_value) != float(value):
                changed_columns.append(column)
        audit_records.append(
            {
                "market": row["market"],
                "stock_code": row["stock_code"],
                "corp_name": row["corp_name"],
                "fiscal_year": row["fiscal_year"],
                "eval_year": row.get("eval_year"),
                "fs_div_requested": supplement.get("fs_div_requested", ""),
                "fs_div_used": supplement.get("fs_div", ""),
                "raw_account_rows": supplement.get("raw_account_rows", 0),
                "changed_column_count": len(changed_columns),
                "changed_columns": "|".join(changed_columns),
            }
        )
    audit = pd.DataFrame(audit_records)
    output = output.drop(columns=["_stock_code_key"])
    return output, audit


def recompute_derived_columns(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.copy()
    numeric_columns = [
        column
        for column in output.columns
        if column
        not in {"market", "stock_code", "corp_name", "firm_size_group", "industry_macro_category"}
    ]
    for column in numeric_columns:
        converted = pd.to_numeric(output[column], errors="coerce")
        if converted.notna().any() or output[column].isna().all():
            output[column] = converted
    output = output.copy()

    output["total_borrowings"] = (
        output["short_term_borrowings"] + output["long_term_borrowings"] + output["bonds_payable"]
    )
    output["intangible_amortization"] = (
        output["amortization_dev_cost"] + output["amortization_other_intangibles"]
    )
    output["ebitda"] = (
        output["operating_income"] + output["depreciation"] + output["intangible_amortization"]
    )

    output["current_ratio"] = safe_ratio(output["current_assets"], output["current_liabilities"])
    output["debt_ratio"] = safe_ratio(output["liabilities_total"], output["equity_total"])
    output["total_borrowings_ratio"] = safe_ratio(
        output["total_borrowings"], output["assets_total"]
    )
    output["short_term_borrowings_share"] = safe_ratio(
        output["short_term_borrowings"], output["total_borrowings"]
    )
    output["noncurrent_liabilities_share"] = safe_ratio(
        output["noncurrent_liabilities"], output["liabilities_total"]
    )
    output["cash_ratio"] = safe_ratio(output["cash_and_equivalents"], output["current_liabilities"])
    output["equity_ratio"] = safe_ratio(output["equity_total"], output["assets_total"])
    output["capital_impairment_ratio"] = safe_ratio(
        output["capital_stock"] - output["equity_total"], output["capital_stock"]
    )

    output["operating_margin"] = safe_ratio(output["operating_income"], output["revenue"])
    output["net_margin"] = safe_ratio(output["net_income"], output["revenue"])
    output["gross_margin"] = safe_ratio(output["gross_profit"], output["revenue"])
    output["interest_burden_ratio"] = safe_ratio(output["interest_expense"], output["revenue"])
    output["cost_of_sales_ratio"] = safe_ratio(output["cost_of_sales"], output["revenue"])
    output["ebitda_margin"] = safe_ratio(output["ebitda"], output["revenue"])
    output["roe"] = safe_ratio(output["net_income"], output["equity_total"])
    output["pretax_roa"] = safe_ratio(output["pretax_income"], output["assets_total"])
    output["operating_roa"] = safe_ratio(output["operating_income"], output["assets_total"])
    output["pretax_roe"] = safe_ratio(output["pretax_income"], output["equity_total"])
    output["operating_roe"] = safe_ratio(output["operating_income"], output["equity_total"])
    output["asset_turnover"] = safe_ratio(output["revenue"], output["assets_total"])
    output["ppe_turnover"] = safe_ratio(output["revenue"], output["property_plant_equipment"])
    output["total_debt_turnover"] = safe_ratio(output["revenue"], output["liabilities_total"])

    group = output.groupby(["market", "stock_code"], sort=False)
    prev_revenue = group["revenue"].shift(1)
    prev_operating_income = group["operating_income"].shift(1)
    prev_net_income = group["net_income"].shift(1)
    prev_pretax_income = group["pretax_income"].shift(1)
    prev_equity_total = group["equity_total"].shift(1)
    prev_total_borrowings = group["total_borrowings"].shift(1)
    prev_current_ratio = group["current_ratio"].shift(1)
    prev_ocf = group["ocf"].shift(1)

    output["revenue_growth"] = growth_ratio(output["revenue"], prev_revenue, abs_base=False)
    output["operating_income_growth"] = growth_ratio(
        output["operating_income"], prev_operating_income, abs_base=True
    )
    output["net_income_growth"] = growth_ratio(output["net_income"], prev_net_income, abs_base=True)
    output["pretax_income_growth"] = growth_ratio(
        output["pretax_income"], prev_pretax_income, abs_base=True
    )
    output["equity_growth"] = growth_ratio(output["equity_total"], prev_equity_total, abs_base=True)
    output["total_borrowings_growth"] = growth_ratio(
        output["total_borrowings"], prev_total_borrowings, abs_base=False
    )
    output["lag1_current_ratio"] = prev_current_ratio
    output["is_2y_consecutive_operating_loss"] = (
        (output["operating_income"].lt(0) & prev_operating_income.lt(0)).fillna(False).astype(int)
    )
    output["is_2y_consecutive_ocf_deficit"] = (
        (output["ocf"].lt(0) & prev_ocf.lt(0)).fillna(False).astype(int)
    )

    output["interest_coverage_ratio"] = capped_ratio(
        output["operating_income"], output["interest_expense"]
    )
    output["ocf_to_sales"] = safe_ratio(output["ocf"], output["revenue"])
    output["ocf_to_total_borrowings"] = safe_ratio(output["ocf"], output["total_borrowings"])
    output["ocf_to_total_liabilities"] = safe_ratio(output["ocf"], output["liabilities_total"])
    output["ocf_to_total_assets"] = safe_ratio(output["ocf"], output["assets_total"])
    output["ocf_deficit_flag"] = output["ocf"].lt(0).fillna(False).astype(int)
    output["cashflow_coverage_ratio"] = capped_ratio(output["ocf"], output["interest_expense"])

    output["accounts_receivable_ratio"] = safe_ratio(
        output["accounts_receivable"], output["assets_total"]
    )
    output["inventory_ratio"] = safe_ratio(output["inventories"], output["assets_total"])
    output["contract_assets_ratio"] = safe_ratio(output["contract_assets"], output["revenue"])
    output["ar_days"] = safe_ratio(output["accounts_receivable"] * 365.0, output["revenue"])
    output["inventory_days"] = safe_ratio(output["inventories"] * 365.0, output["cost_of_sales"])
    output["ap_days"] = safe_ratio(output["accounts_payable"] * 365.0, output["cost_of_sales"])
    output["ppe_ratio"] = safe_ratio(output["property_plant_equipment"], output["assets_total"])
    output["intangible_assets_ratio"] = safe_ratio(
        output["intangible_assets"], output["assets_total"]
    )
    output["advances_from_customers_ratio"] = safe_ratio(
        output["advances_from_customers"], output["liabilities_total"]
    )
    output["ppe_intensity"] = safe_ratio(output["property_plant_equipment"], output["revenue"])
    output["intangible_intensity"] = safe_ratio(output["intangible_assets"], output["revenue"])

    output["icr_under_1"] = output["interest_coverage_ratio"].lt(1).fillna(False).astype(int)
    group = output.groupby(["market", "stock_code"], sort=False)
    output["is_zombie_3y"] = (
        group["icr_under_1"]
        .transform(lambda s: s.rolling(window=3, min_periods=3).sum())
        .eq(3)
        .astype(int)
    )
    output["accruals_ratio"] = safe_ratio(
        output["net_income"] - output["ocf"], output["assets_total"]
    )
    output["delta_accruals_ratio"] = group["accruals_ratio"].transform(lambda s: s.diff())
    output["non_paid_in_equity_ratio"] = safe_ratio(
        output["equity_total"] - output["capital_stock"], output["assets_total"]
    )
    output["delta_non_paid_in_equity_ratio"] = group["non_paid_in_equity_ratio"].transform(
        lambda s: s.diff()
    )
    output["delta_st_borrowings_share"] = group["short_term_borrowings_share"].transform(
        lambda s: s.diff()
    )
    output["operating_margin_diff"] = group["operating_margin"].transform(lambda s: s.diff())
    output["ebitda_margin_diff"] = group["ebitda_margin"].transform(lambda s: s.diff())
    output["equity_ratio_diff"] = group["equity_ratio"].transform(lambda s: s.diff())
    output["current_ratio_diff"] = group["current_ratio"].transform(lambda s: s.diff())
    output["capital_impairment_diff"] = group["capital_impairment_ratio"].transform(
        lambda s: s.diff()
    )
    output["ocf_to_total_borrowings_diff"] = group["ocf_to_total_borrowings"].transform(
        lambda s: s.diff()
    )
    output["net_margin_diff"] = group["net_margin"].transform(lambda s: s.diff())
    output["cash_ratio_diff"] = group["cash_ratio"].transform(lambda s: s.diff())
    output["total_borrowings_ratio_diff"] = group["total_borrowings_ratio"].transform(
        lambda s: s.diff()
    )
    output["ocf_to_total_liabilities_diff"] = group["ocf_to_total_liabilities"].transform(
        lambda s: s.diff()
    )
    output["lag1_equity_ratio"] = group["equity_ratio"].shift(1)

    om_mean = group["operating_margin"].transform(
        lambda s: s.rolling(window=3, min_periods=2).mean()
    )
    om_std = group["operating_margin"].transform(lambda s: s.rolling(window=3, min_periods=2).std())
    output["rolling_3y_cv_operating_margin"] = safe_ratio(om_std, om_mean.abs()).clip(upper=CV_CAP)
    otb_mean = group["ocf_to_total_borrowings"].transform(
        lambda s: s.rolling(window=3, min_periods=2).mean()
    )
    otb_std = group["ocf_to_total_borrowings"].transform(
        lambda s: s.rolling(window=3, min_periods=2).std()
    )
    output["rolling_3y_cv_ocf_to_total_borrowings"] = safe_ratio(otb_std, otb_mean.abs()).clip(
        upper=CV_CAP
    )
    rg_mean = group["revenue_growth"].transform(lambda s: s.rolling(window=3, min_periods=2).mean())
    rg_std = group["revenue_growth"].transform(lambda s: s.rolling(window=3, min_periods=2).std())
    output["rolling_3y_cv_revenue_growth"] = safe_ratio(rg_std, rg_mean.abs()).clip(upper=CV_CAP)

    prev2_operating_income = group["operating_income"].shift(2)
    prev2_ocf = group["ocf"].shift(2)
    output["is_3y_consecutive_operating_loss"] = (
        (
            output["operating_income"].lt(0)
            & prev_operating_income.lt(0)
            & prev2_operating_income.lt(0)
        )
        .fillna(False)
        .astype(int)
    )
    output["is_3y_consecutive_ocf_deficit"] = (
        (output["ocf"].lt(0) & prev_ocf.lt(0) & prev2_ocf.lt(0)).fillna(False).astype(int)
    )
    output["total_assets_growth"] = growth_ratio(
        output["assets_total"], group["assets_total"].shift(1), abs_base=False
    )
    output["negative_equity_flag"] = output["equity_total"].lt(0).fillna(False).astype(int)
    output["is_operating_income_turn_negative"] = (
        (output["operating_income"].lt(0) & prev_operating_income.ge(0)).fillna(False).astype(int)
    )
    output["is_ocf_turn_negative"] = (
        (output["ocf"].lt(0) & prev_ocf.ge(0)).fillna(False).astype(int)
    )
    output["is_current_ratio_below_1"] = (
        (output["current_ratio"].lt(1) & prev_current_ratio.ge(1)).fillna(False).astype(int)
    )
    output["is_negative_equity_entry"] = (
        (output["equity_total"].lt(0) & prev_equity_total.ge(0)).fillna(False).astype(int)
    )
    output["ar_days_diff"] = group["ar_days"].transform(lambda s: s.diff())
    output["inventory_days_diff"] = group["inventory_days"].transform(lambda s: s.diff())
    output["ap_days_diff"] = group["ap_days"].transform(lambda s: s.diff())

    output["industry_current_ratio_percentile"] = (
        output.groupby(["fiscal_year", "industry_macro_category"])["current_ratio"]
        .rank(pct=True)
        .mul(100.0)
    )
    return output


def main() -> None:
    args = parse_args()
    model_v1 = pd.read_csv(args.source, encoding="utf-8-sig", low_memory=False)
    original_columns = model_v1.columns.tolist()
    raw = load_supplement_rows(args.raw_supplement)
    supplements = build_supplement_frame(raw)
    supplemented, audit = apply_supplements(model_v1, supplements)
    supplemented = recompute_derived_columns(supplemented)
    supplemented = supplemented.loc[:, original_columns]

    print(f"[Supplements] raw_rows={len(raw):,}, companies={len(supplements):,}")
    print(f"[Applied] rows={len(audit):,}")
    if not audit.empty:
        print(
            audit[
                [
                    "market",
                    "stock_code",
                    "corp_name",
                    "fiscal_year",
                    "fs_div_used",
                    "changed_column_count",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    if args.dry_run:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    supplemented.to_csv(args.output, index=False, encoding="utf-8-sig")
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.audit_output, index=False, encoding="utf-8-sig")
    print(f"[Saved] {args.output} ({len(supplemented):,} rows)")
    print(f"[Saved] {args.audit_output} ({len(audit):,} rows)")


if __name__ == "__main__":
    main()
