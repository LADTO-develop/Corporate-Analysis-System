import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path

st.set_page_config(
    page_title="기업 신용위험 예측 시스템",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 경로 설정
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

PROCESSED_DIR = DATA_DIR / "processed"
KOSDAQ_DATA_PATH = PROCESSED_DIR / "kosdaq" / "kosdaq_train.csv"
KOSPI_DATA_PATH = PROCESSED_DIR / "kospi" / "kospi_train.csv"

KOSDAQ_MODEL_PATH = MODEL_DIR / "xgb_kosdaq.pkl"
KOSPI_MODEL_PATH = MODEL_DIR / "xgb_kospi.pkl"

KOSDAQ_FEATURE_PATH = MODEL_DIR / "feature_cols_kosdaq.csv"
KOSPI_FEATURE_PATH = MODEL_DIR / "feature_cols_kospi.csv"

KOSDAQ_THRESHOLD_PATH = MODEL_DIR / "threshold_kosdaq.txt"
KOSPI_THRESHOLD_PATH = MODEL_DIR / "threshold_kospi.txt"

# =========================================================
# 변수명 한글 매핑
# =========================================================
FEATURE_NAME_MAP = {
    "OperINC_InterEXP_Ratio_win": "영업이익/이자비용 비율",
    "OCF_InterEXP_Ratio_win": "영업현금흐름/이자비용 비율",
    "InterEXP_SalesINC_Ratio_win": "이자비용/매출 비율",
    "NI_SalesINC_Ratio_win": "순이익/매출 비율",
    "OperINC_SalesINC_Ratio_win": "영업이익/매출 비율",
    "GrossINC_SalesINC_Ratio_win": "매출총이익/매출 비율",
    "Total_EQ": "자본총계",
    "Total_AST": "총자산",
    "Total_LIAB": "총부채",
    "Current_AST": "유동자산",
    "Current_LIAB": "유동부채",
    "NI_INC": "순이익",
    "Operating_INC": "영업이익",
    "Sales_INC": "매출액",
    "OCF": "영업현금흐름",
    "Interest_EXP": "이자비용",
    "Debt_Ratio_win": "부채비율",
    "CurAST_CurLIAB_Ratio_win": "유동비율",
    "TotalEQ_TotalAST_Ratio_win": "자기자본비율",
    "TD_TotalAST_Ratio_win": "총차입금/총자산 비율",
    "ShortLIAB_TD_Ratio_win": "단기부채/총차입금 비율",
    "CCE_CurLIAB_Ratio_win": "현금성자산/유동부채 비율",
    "OCF_SalesINC_Ratio_win": "영업현금흐름/매출 비율",
    "OCF_TD_Ratio_win": "영업현금흐름/총차입금 비율",
    "OCF_TotalLIAB_Ratio_win": "영업현금흐름/총부채 비율",
    "TradeAST_TotalAST_Ratio_win": "매출채권/총자산 비율",
    "INVAST_TotalAST_Ratio_win": "재고자산/총자산 비율",
    "ContractAST_SalesINC_Ratio_win": "계약자산/매출 비율",
    "WorkingCapital_TotalAST_Ratio_win": "운전자본/총자산 비율",
    "NetDebt_TotalAST_Ratio_win": "순차입금/총자산 비율",
    "log_Total_AST": "총자산 로그값",
    "SalesINC_Growth_win": "매출 증가율",
    "OperINC_Growth_win": "영업이익 증가율",
    "OCF_3Y_Mean": "3년 평균 영업현금흐름",
    "OperINC_InterEXP_Ratio_Trend": "이자보상능력 추세",
    "firm_age": "기업 업력",
    "BaseRate": "기준금리",
    "3YRate_KTB": "국고채 3년 금리",
    "3YRate_CorpBond_AA": "회사채 AA- 3년 금리",
    "3YRate_CorpBond_BBB": "회사채 BBB- 3년 금리",
    "USDKRW": "원/달러 환율",
    "PPI": "생산자물가지수",
    "Spread_Credit": "신용스프레드",
    "Spread_Quality": "등급스프레드",
}

def to_korean_feature_name(feature_name: str) -> str:
    return FEATURE_NAME_MAP.get(feature_name, feature_name)

# =========================================================
# 스타일
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fb;
        color: #18212f;
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 251, 0.85);
    }

    .block-container {
        padding-top: 3.5rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }

    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #3457d5 100%);
        border-radius: 24px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 12px 32px rgba(29, 78, 216, 0.18);
    }

    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 8px;
    }

    .hero-desc {
        font-size: 0.98rem;
        color: #dbe7ff;
        line-height: 1.6;
    }

    .section-card {
        background: white;
        border: 1px solid #e6ebf3;
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04);
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #18212f;
        margin-bottom: 8px;
    }

    .section-sub {
        font-size: 0.93rem;
        color: #5f6b7c;
        margin-bottom: 16px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        background: linear-gradient(135deg, #0f172a 0%, #3457d5 100%);
        color: white;
        border: none;
        box-shadow: 0 10px 22px rgba(52, 87, 213, 0.22);
        font-weight: 700;
    }

    .mini-kpi {
        background: #f8fafc;
        border: 1px solid #e8edf5;
        border-radius: 18px;
        padding: 18px;
        min-height: 120px;
    }

    .mini-label {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 8px;
    }

    .mini-value {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
        color: #0f172a;
    }

    .mini-foot {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 8px;
    }

    .blue { color: #2563eb; }
    .orange { color: #ea580c; }
    .green { color: #059669; }
    .purple { color: #7c3aed; }

    .badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: #eaf2ff;
        color: #2563eb;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .risk-item {
        background: #f8fafc;
        border-left: 4px solid #2563eb;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }

    .risk-title {
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 4px;
    }

    .risk-desc {
        color: #5f6b7c;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .report-box {
        background: #fff;
        border: 1px solid #e6ebf3;
        border-radius: 18px;
        padding: 18px 20px;
        color: #334155;
        line-height: 1.75;
    }

    .score-pill {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 로딩 함수
# =========================================================
@st.cache_resource
def load_models():
    model_kq = joblib.load(KOSDAQ_MODEL_PATH)
    model_kp = joblib.load(KOSPI_MODEL_PATH)
    return model_kq, model_kp

@st.cache_data
def load_feature_cols():
    feature_cols_kq = pd.read_csv(KOSDAQ_FEATURE_PATH)["feature"].tolist()
    feature_cols_kp = pd.read_csv(KOSPI_FEATURE_PATH)["feature"].tolist()
    return feature_cols_kq, feature_cols_kp

@st.cache_data
def load_thresholds():
    with open(KOSDAQ_THRESHOLD_PATH, "r", encoding="utf-8") as f:
        th_kq = float(f.read().strip())
    with open(KOSPI_THRESHOLD_PATH, "r", encoding="utf-8") as f:
        th_kp = float(f.read().strip())
    return th_kq, th_kp

@st.cache_data
def load_data():
    kosdaq = pd.read_csv(KOSDAQ_DATA_PATH)
    kospi = pd.read_csv(KOSPI_DATA_PATH)
    return kosdaq, kospi

# =========================================================
# 유틸 함수
# =========================================================
ID_COLS = ["stock_code", "company_name", "market", "fiscal_year", "target_binary"]
SCENARIO_COLS = ["BaseRate", "Spread_Credit", "USDKRW"]

def get_company_column(df: pd.DataFrame):
    for col in ["company_name", "company_name_target", "corp_name", "name"]:
        if col in df.columns:
            return col
    return None

def get_stock_column(df: pd.DataFrame):
    for col in ["stock_code", "code", "ticker"]:
        if col in df.columns:
            return col
    return None

def normalize_code(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(6) if s.isdigit() else s

def prepare_feature_row(row_df: pd.DataFrame, feature_cols: list):
    X = row_df.copy()

    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0

    X = X[feature_cols].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(0.0)
    return X

def apply_scenario_to_row(row_df: pd.DataFrame, base_rate_delta=0.0, spread_delta=0.0, usdkrw_delta=0.0):
    row_df = row_df.copy()

    if "BaseRate" in row_df.columns:
        row_df["BaseRate"] = pd.to_numeric(row_df["BaseRate"], errors="coerce").fillna(0) + base_rate_delta

    if "Spread_Credit" in row_df.columns:
        row_df["Spread_Credit"] = pd.to_numeric(row_df["Spread_Credit"], errors="coerce").fillna(0) + spread_delta

    if "USDKRW" in row_df.columns:
        row_df["USDKRW"] = pd.to_numeric(row_df["USDKRW"], errors="coerce").fillna(0) + usdkrw_delta

    return row_df

def find_company_row(df: pd.DataFrame, company_query: str, fiscal_year: int):
    company_col = get_company_column(df)
    stock_col = get_stock_column(df)

    query = str(company_query).strip()
    query_norm = normalize_code(query)

    tmp = df.copy()
    if stock_col is not None:
        tmp["_stock_norm"] = tmp[stock_col].apply(normalize_code)
    else:
        tmp["_stock_norm"] = ""

    cond_year = tmp["fiscal_year"] == fiscal_year

    cond_name = pd.Series([False] * len(tmp))
    if company_col is not None:
        cond_name = tmp[company_col].astype(str).str.contains(query, case=False, na=False)

    cond_code = tmp["_stock_norm"] == query_norm

    result = tmp[cond_year & (cond_name | cond_code)].copy()
    return result

def predict_one(row_df, market, model_kq, model_kp, feature_cols_kq, feature_cols_kp, th_kq, th_kp):
    if market == "KOSDAQ":
        model = model_kq
        feature_cols = feature_cols_kq
        threshold = th_kq
    else:
        model = model_kp
        feature_cols = feature_cols_kp
        threshold = th_kp

    X = prepare_feature_row(row_df, feature_cols)
    prob_down = float(model.predict_proba(X)[0, 1])
    prob_keep = 1.0 - prob_down
    pred_label = int(prob_down >= threshold)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_series = pd.Series(shap_values[0], index=X.columns)
    top5 = (
        shap_series.abs()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    shap_top_df = pd.DataFrame({
        "feature": top5,
        "shap_value": [float(shap_series[f]) for f in top5],
        "feature_value": [float(X.iloc[0][f]) for f in top5],
    })

    return {
        "market": market,
        "threshold": threshold,
        "prob_down": prob_down,
        "prob_keep": prob_keep,
        "pred_label": pred_label,
        "shap_top_df": shap_top_df,
        "X": X,
        "explainer": explainer,
        "shap_values": shap_values,
    }

def run_scenario_prediction(row_df, market, model_kq, model_kp, feature_cols_kq, feature_cols_kp, th_kq, th_kp,
                            base_rate_delta=0.0, spread_delta=0.0, usdkrw_delta=0.0):
    scenario_row = apply_scenario_to_row(
        row_df,
        base_rate_delta=base_rate_delta,
        spread_delta=spread_delta,
        usdkrw_delta=usdkrw_delta
    )
    return predict_one(
        row_df=scenario_row,
        market=market,
        model_kq=model_kq,
        model_kp=model_kp,
        feature_cols_kq=feature_cols_kq,
        feature_cols_kp=feature_cols_kp,
        th_kq=th_kq,
        th_kp=th_kp,
    )

def build_summary_text(result, company_name, fiscal_year):
    risk_pct = result["prob_down"] * 100
    keep_pct = result["prob_keep"] * 100
    market = result["market"]

    top_rows = result["shap_top_df"].copy()
    top_rows["direction"] = np.where(top_rows["shap_value"] > 0, "위험 증가", "위험 완화")
    top_rows["feature_kr"] = top_rows["feature"].apply(to_korean_feature_name)

    top3 = top_rows.head(3)

    bullet_text = []
    for _, r in top3.iterrows():
        bullet_text.append(f"- {r['feature_kr']}: {r['direction']}")

    bullet_text = "\n".join(bullet_text)

    text = f"""
{company_name}의 {fiscal_year}년 기준 {market} 모델 예측 결과, 투기등급 강등 확률은 {risk_pct:.2f}%이며 투자적격 유지 확률은 {keep_pct:.2f}%입니다.
상위 설명 변수 기준으로는 아래 항목들이 예측에 큰 영향을 주었습니다.

{bullet_text}

본 결과는 저장된 XGBoost 모델을 기반으로 산출되었으며, SHAP 값을 이용해 개별 기업 수준에서 예측 원인을 해석했습니다.
    """.strip()

    return text

# =========================================================
# 초기 로딩
# =========================================================
try:
    model_kq, model_kp = load_models()
    feature_cols_kq, feature_cols_kp = load_feature_cols()
    th_kq, th_kp = load_thresholds()
    kosdaq_df, kospi_df = load_data()
except Exception as e:
    st.error(f"모델/데이터 로딩 중 오류가 발생했습니다: {e}")
    st.stop()

# =========================================================
# 사이드바
# =========================================================
st.sidebar.title("설정")
page = st.sidebar.radio("화면 선택", ["회사 검색", "CSV 업로드"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("모델 정보")
st.sidebar.caption(f"KOSDAQ threshold: {th_kq}")
st.sidebar.caption(f"KOSPI threshold: {th_kp}")

st.sidebar.markdown("---")
st.sidebar.subheader("시나리오 입력")
base_rate_delta = st.sidebar.number_input("기준금리 변화(%p)", value=0.0, step=0.1)
spread_delta = st.sidebar.number_input("신용스프레드 변화(%p)", value=0.0, step=0.1)
usdkrw_delta = st.sidebar.number_input("환율 변화(원)", value=0.0, step=10.0)

# =========================================================
# 헤더
# =========================================================
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">기업 신용위험 예측 시스템</div>
        <div class="hero-desc">
            저장된 XGBoost 모델을 기반으로 기업의 투자적격 유지 확률과 투기등급 강등 확률을 예측하고, SHAP 기반 핵심 원인과 시나리오 변화를 함께 제공합니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 회사 검색 모드
# =========================================================
if page == "회사 검색":
    st.markdown('<div class="section-title">기업 단건 조회</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">회사명 또는 종목코드와 기준연도를 입력하면 저장된 모델로 즉시 예측합니다.</div>', unsafe_allow_html=True)

    company_query = st.text_input("회사명 또는 종목코드", placeholder="예: 삼성전자 또는 005930")
    fiscal_year = st.selectbox("기준 연도", sorted(kospi_df["fiscal_year"].dropna().unique().tolist(), reverse=True))

    if st.button("리포트 생성", type="primary", use_container_width=True):
        if not company_query.strip():
            st.warning("회사명 또는 종목코드를 입력해주세요.")
            st.stop()

        candidates_kq = find_company_row(kosdaq_df, company_query, fiscal_year)
        candidates_kp = find_company_row(kospi_df, company_query, fiscal_year)

        total_found = len(candidates_kq) + len(candidates_kp)

        if total_found == 0:
            st.error("해당 기업/연도 데이터를 찾지 못했습니다.")
            st.stop()

        if len(candidates_kq) > 0 and len(candidates_kp) > 0:
            st.warning("동일 조건에 대해 KOSDAQ/KOSPI 모두 검색되었습니다. 첫 번째 결과를 사용합니다.")

        if len(candidates_kq) > 0:
            row_df = candidates_kq.iloc[[0]].copy()
            market = "KOSDAQ"
        else:
            row_df = candidates_kp.iloc[[0]].copy()
            market = "KOSPI"

        company_col = get_company_column(row_df)
        stock_col = get_stock_column(row_df)

        company_name = row_df.iloc[0][company_col] if company_col else company_query
        stock_code = row_df.iloc[0][stock_col] if stock_col else "-"

        base_result = predict_one(
            row_df=row_df,
            market=market,
            model_kq=model_kq,
            model_kp=model_kp,
            feature_cols_kq=feature_cols_kq,
            feature_cols_kp=feature_cols_kp,
            th_kq=th_kq,
            th_kp=th_kp,
        )

        scenario_result = run_scenario_prediction(
            row_df=row_df,
            market=market,
            model_kq=model_kq,
            model_kp=model_kp,
            feature_cols_kq=feature_cols_kq,
            feature_cols_kp=feature_cols_kp,
            th_kq=th_kq,
            th_kp=th_kp,
            base_rate_delta=base_rate_delta,
            spread_delta=spread_delta,
            usdkrw_delta=usdkrw_delta,
        )

        st.markdown("---")
        st.markdown(f'<div class="section-title">{company_name} | {market}</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <span class="badge">{stock_code}</span>
            <span class="badge">{fiscal_year}</span>
            <span class="badge">threshold {base_result['threshold']}</span>
            """,
            unsafe_allow_html=True,
        )

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f'<div class="mini-kpi"><div class="mini-label">투자적격 유지 확률</div><div class="mini-value blue">{base_result["prob_keep"]*100:.2f}%</div><div class="mini-foot">1 - 강등확률</div></div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f'<div class="mini-kpi"><div class="mini-label">투기등급 강등 확률</div><div class="mini-value orange">{base_result["prob_down"]*100:.2f}%</div><div class="mini-foot">모델 예측 확률</div></div>',
                unsafe_allow_html=True,
            )
        with k3:
            risk_label = "강등 위험" if base_result["pred_label"] == 1 else "투자적격 유지"
            st.markdown(
                f'<div class="mini-kpi"><div class="mini-label">최종 판정</div><div class="mini-value green">{risk_label}</div><div class="mini-foot">threshold 적용 결과</div></div>',
                unsafe_allow_html=True,
            )
        with k4:
            change = (scenario_result["prob_down"] - base_result["prob_down"]) * 100
            sign = "+" if change >= 0 else ""
            st.markdown(
                f'<div class="mini-kpi"><div class="mini-label">시나리오 변화폭</div><div class="mini-value purple">{sign}{change:.2f}%p</div><div class="mini-foot">강등확률 기준</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown('<div class="section-title">핵심 원인 (SHAP 상위 5개)</div>', unsafe_allow_html=True)
            shap_top_df = base_result["shap_top_df"].copy()
            shap_top_df["feature_kr"] = shap_top_df["feature"].apply(to_korean_feature_name)
            shap_top_df["영향 방향"] = np.where(shap_top_df["shap_value"] > 0, "위험 증가", "위험 완화")
            shap_top_df["절대영향"] = shap_top_df["shap_value"].abs()
            st.dataframe(
                shap_top_df[["feature_kr", "feature_value", "shap_value", "영향 방향"]]
                .rename(columns={
                    "feature_kr": "변수",
                    "feature_value": "현재값",
                    "shap_value": "SHAP값",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with c2:
            st.markdown('<div class="section-title">시나리오 결과</div>', unsafe_allow_html=True)
            scenario_df = pd.DataFrame({
                "구분": ["기준", "시나리오 반영"],
                "투자적격 유지 확률": [
                    round(base_result["prob_keep"] * 100, 2),
                    round(scenario_result["prob_keep"] * 100, 2),
                ],
                "투기등급 강등 확률": [
                    round(base_result["prob_down"] * 100, 2),
                    round(scenario_result["prob_down"] * 100, 2),
                ],
            })
            st.dataframe(scenario_df, use_container_width=True, hide_index=True)

            st.caption(
                f"적용 시나리오: 기준금리 {base_rate_delta:+.2f}%p / "
                f"신용스프레드 {spread_delta:+.2f}%p / "
                f"환율 {usdkrw_delta:+.0f}원"
            )

        st.markdown("---")
        st.markdown('<div class="section-title">최종 해석 리포트</div>', unsafe_allow_html=True)

        summary_text = build_summary_text(base_result, company_name, fiscal_year)
        st.markdown(
            f"""
            <div class="report-box">
            {summary_text.replace(chr(10), "<br>")}
            </div>
            """,
            unsafe_allow_html=True,
        )

        download_shap_df = base_result["shap_top_df"].copy()
        download_shap_df["feature"] = download_shap_df["feature"].apply(to_korean_feature_name)

        report_text = f"""
기업명: {company_name}
종목코드: {stock_code}
시장: {market}
기준연도: {fiscal_year}

투자적격 유지 확률: {base_result["prob_keep"]*100:.2f}%
투기등급 강등 확률: {base_result["prob_down"]*100:.2f}%
threshold: {base_result["threshold"]}

시나리오 반영 후 강등 확률: {scenario_result["prob_down"]*100:.2f}%

상위 SHAP 변수:
{download_shap_df.to_string(index=False)}
        """.strip()

        st.download_button(
            "리포트 다운로드",
            data=report_text,
            file_name=f"risk_report_{company_name}_{fiscal_year}.txt",
            mime="text/plain",
            use_container_width=True,
        )

# =========================================================
# CSV 업로드
# =========================================================
else:
    st.markdown('<div class="section-title">CSV 일괄 예측</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">시장별 모델을 자동 분기하여 업로드된 기업 목록에 대해 일괄 예측합니다.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("CSV 업로드", type=["csv"])

    st.caption(
        "권장 컬럼 예시: market, stock_code, company_name, fiscal_year + "
        "학습 feature 컬럼들(BaseRate, Spread_Credit, USDKRW 포함 가능)"
    )

    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"CSV를 읽는 중 오류가 발생했습니다: {e}")
            st.stop()

        st.markdown('<div class="section-title">업로드 데이터 미리보기</div>', unsafe_allow_html=True)
        st.dataframe(input_df.head(), use_container_width=True)

        if st.button("배치 예측 실행", type="primary", use_container_width=True):
            if "market" not in input_df.columns:
                st.error("CSV에 market 컬럼이 필요합니다. (KOSDAQ / KOSPI)")
                st.stop()

            results = []
            errors = []

            for idx, row in input_df.iterrows():
                row_df = pd.DataFrame([row]).copy()
                market = str(row.get("market", "")).strip().upper()

                if market not in ["KOSDAQ", "KOSPI"]:
                    errors.append({"row": idx, "error": "market 값이 KOSDAQ/KOSPI가 아님"})
                    continue

                try:
                    pred = predict_one(
                        row_df=row_df,
                        market=market,
                        model_kq=model_kq,
                        model_kp=model_kp,
                        feature_cols_kq=feature_cols_kq,
                        feature_cols_kp=feature_cols_kp,
                        th_kq=th_kq,
                        th_kp=th_kp,
                    )

                    row_result = {
                        "row": idx,
                        "market": market,
                        "company_name": row.get("company_name", ""),
                        "stock_code": row.get("stock_code", ""),
                        "fiscal_year": row.get("fiscal_year", ""),
                        "prob_keep": pred["prob_keep"],
                        "prob_down": pred["prob_down"],
                        "pred_label": pred["pred_label"],
                        "top1_feature": to_korean_feature_name(pred["shap_top_df"].iloc[0]["feature"]),
                        "top1_shap": pred["shap_top_df"].iloc[0]["shap_value"],
                    }
                    results.append(row_result)

                except Exception as e:
                    errors.append({"row": idx, "error": str(e)})

            result_df = pd.DataFrame(results)
            error_df = pd.DataFrame(errors)

            st.markdown("---")
            st.markdown('<div class="section-title">예측 결과</div>', unsafe_allow_html=True)

            if len(result_df) > 0:
                display_df = result_df.copy()
                display_df["투자적격 유지 확률(%)"] = (display_df["prob_keep"] * 100).round(2)
                display_df["투기등급 강등 확률(%)"] = (display_df["prob_down"] * 100).round(2)
                display_df["최종 판정"] = np.where(display_df["pred_label"] == 1, "강등 위험", "투자적격 유지")

                st.dataframe(
                    display_df[
                        [
                            "market", "company_name", "stock_code", "fiscal_year",
                            "투자적격 유지 확률(%)", "투기등급 강등 확률(%)",
                            "최종 판정", "top1_feature", "top1_shap"
                        ]
                    ].rename(columns={"top1_feature": "주요 변수"}),
                    use_container_width=True,
                    hide_index=True,
                )

                csv_bytes = display_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "예측 결과 CSV 다운로드",
                    data=csv_bytes,
                    file_name="batch_prediction_result.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            if len(error_df) > 0:
                st.markdown('<div class="section-title">오류 행</div>', unsafe_allow_html=True)
                st.dataframe(error_df, use_container_width=True, hide_index=True)