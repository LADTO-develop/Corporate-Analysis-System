"""Theme CSS for the Streamlit credit dashboard."""

from __future__ import annotations

import streamlit as st


def inject_dashboard_theme(theme_mode: str = "light") -> None:
    normalized_theme = str(theme_mode or "light").strip().lower()

    if normalized_theme == "dark":
        cas_theme_vars = """
          color-scheme: dark;
          --cas-page-bg: #080b12;
          --cas-base-text: #f8fafc;

          --cas-blue: #facc15;
          --cas-accent: #facc15;
          --cas-accent-text: #fde68a;
          --cas-accent-text-strong: #fff7d6;
          --cas-accent-soft: rgba(250, 204, 21, 0.14);
          --cas-accent-surface: rgba(250, 204, 21, 0.12);
          --cas-accent-surface-strong: rgba(250, 204, 21, 0.18);
          --cas-accent-border: rgba(250, 204, 21, 0.38);
          --cas-accent-border-strong: rgba(250, 204, 21, 0.58);
          --cas-tab-selected-bg: rgba(250, 204, 21, 0.16);
          --cas-step-track-bg: rgba(250, 204, 21, 0.10);
          --cas-live-step-bg: rgba(250, 204, 21, 0.10);

          --cas-risk: #fb7185;
          --cas-risk-text: #fda4af;
          --cas-success: #34d399;
          --cas-warning: #f59e0b;
          --cas-neutral: #facc15;

          --cas-text: var(--cas-base-text);
          --cas-muted: rgba(203, 213, 225, 0.86);
          --cas-panel: rgba(255, 255, 255, 0.055);
          --cas-panel-strong: rgba(255, 255, 255, 0.085);
          --cas-border: rgba(250, 204, 21, 0.22);
          --cas-border-soft: rgba(250, 204, 21, 0.14);
          --cas-shadow: 0 1px 2px rgba(0, 0, 0, 0.32);
          --cas-card-bg: rgba(255, 255, 255, 0.045);
          --cas-card-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);

          --cas-risk-soft: rgba(251, 113, 133, 0.14);
          --cas-risk-border: rgba(251, 113, 133, 0.40);
          --cas-success-soft: rgba(52, 211, 153, 0.14);
          --cas-success-border: rgba(52, 211, 153, 0.38);
          --cas-warning-soft: rgba(245, 158, 11, 0.16);
          --cas-warning-border: rgba(245, 158, 11, 0.48);
          --cas-neutral-soft: rgba(250, 204, 21, 0.10);
          --cas-neutral-border: rgba(250, 204, 21, 0.32);
        """
    else:
        cas_theme_vars = """
          color-scheme: light;
          --cas-page-bg: #ffffff;
          --cas-base-text: #0f172a;

          --cas-blue: var(--st-primary-color, var(--primary-color, #1d4ed8));
          --cas-accent: #0284c7;
          --cas-accent-text: #0369a1;
          --cas-accent-text-strong: #075985;
          --cas-accent-soft: rgba(2, 132, 199, 0.10);
          --cas-accent-surface: rgba(224, 242, 254, 0.48);
          --cas-accent-surface-strong: rgba(224, 242, 254, 0.66);
          --cas-accent-border: rgba(2, 132, 199, 0.24);
          --cas-accent-border-strong: rgba(2, 132, 199, 0.34);
          --cas-tab-selected-bg: rgba(224, 242, 254, 0.58);
          --cas-step-track-bg: rgba(248, 250, 252, 0.74);
          --cas-live-step-bg: rgba(248, 250, 252, 0.78);

          --cas-risk: #c85050;
          --cas-risk-text: #d14a4a;
          --cas-success: #2f9e5b;
          --cas-warning: #b7791f;
          --cas-neutral: #4f6fad;

          --cas-text: var(--cas-base-text);
          --cas-muted: rgba(71, 85, 105, 0.76);
          --cas-panel: rgba(15, 23, 42, 0.035);
          --cas-panel-strong: rgba(15, 23, 42, 0.06);
          --cas-border: rgba(148, 163, 184, 0.34);
          --cas-border-soft: rgba(148, 163, 184, 0.22);
          --cas-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
          --cas-card-bg: rgba(15, 23, 42, 0.025);
          --cas-card-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);

          --cas-risk-soft: rgba(200, 80, 80, 0.14);
          --cas-risk-border: rgba(200, 80, 80, 0.38);
          --cas-success-soft: rgba(47, 158, 91, 0.14);
          --cas-success-border: rgba(47, 158, 91, 0.38);
          --cas-warning-soft: rgba(183, 121, 31, 0.14);
          --cas-warning-border: rgba(183, 121, 31, 0.38);
          --cas-neutral-soft: rgba(128, 128, 128, 0.10);
          --cas-neutral-border: rgba(128, 128, 128, 0.28);
        """
    """Apply dashboard styling without forcing a fixed light theme.

    Streamlit's Settings menu changes the app theme on the client side.  CSS
    inserted through ``st.markdown`` cannot reliably read that setting in every
    Streamlit release, so this stylesheet deliberately avoids hard-coded page
    backgrounds.  Custom CAS cards are rendered as subtle currentColor-based
    translucent surfaces, which makes them follow both Streamlit light and dark
    themes automatically.
    """
    css ="""
        <style>
        :root,
        .stApp {
        __CAS_THEME_VARS__
        }

        .stApp,
        div[data-testid="stAppViewContainer"],
        div[data-testid="stMain"],
        div[data-testid="stMainBlockContainer"],
        .main,
        .main .block-container {
          background: var(--cas-page-bg) !important;
          color: var(--cas-text) !important;
        }

        .main .block-container {
          max-width: 1480px;
          padding-top: 1.25rem;
          padding-bottom: 2rem;
        }

        .main .block-container, .main .block-container * {
          letter-spacing: 0 !important;
        }

        h1 {
          color: inherit;
          font-size: 1.72rem !important;
          line-height: 1.25 !important;
          margin: 0 0 0.25rem 0 !important;
        }

        h2, h3 {
          color: inherit;
          line-height: 1.35 !important;
        }

        h2 {
          border-top: 1px solid var(--cas-border-soft);
          font-size: 1.12rem !important;
          margin-top: 1.1rem !important;
          padding-top: 0.8rem !important;
        }

        h3 {
          font-size: 1rem !important;
        }

        div[data-testid="stCaptionContainer"] {
          color: var(--cas-muted);
          font-size: 0.9rem;
          line-height: 1.58;
          max-width: 1120px;
        }

        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {
          line-height: 1.62;
        }

        section[data-testid="stSidebar"] {
          border-right: 1px solid var(--cas-border);
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
          color: var(--cas-muted);
          font-size: 0.88rem;
          line-height: 1.5;
        }

        div[data-testid="stTabs"] [role="tablist"] {
          align-items: center;
          background: var(--cas-panel-strong);
          border: 1px solid var(--cas-border);
          border-radius: 10px;
          box-shadow: var(--cas-card-shadow);
          gap: 0.35rem;
          margin: 0.35rem 0 1rem 0;
          max-width: 100%;
          padding: 0.42rem;
          position: sticky;
          top: 0.35rem;
          z-index: 10;
        }

        button[role="tab"] {
          background: var(--cas-card-bg) !important;
          border: 1px solid var(--cas-border-soft) !important;
          border-radius: 8px !important;
          color: var(--cas-text) !important;
          flex: 1 1 0;
          font-size: 0.96rem !important;
          font-weight: 850 !important;
          justify-content: center;
          min-height: 2.72rem;
          min-width: 9.5rem;
          padding: 0.58rem 0.95rem !important;
          transition:
            background 0.18s ease,
            color 0.18s ease,
            border-color 0.18s ease,
            box-shadow 0.18s ease,
            transform 0.18s ease;
          white-space: nowrap;
        }

        button[role="tab"]:hover {
          background: var(--cas-panel-strong) !important;
          border-color: var(--cas-neutral-border) !important;
        }

        button[role="tab"][aria-selected="true"] {
          background: var(--cas-tab-selected-bg) !important;
          border-color: var(--cas-accent-border-strong) !important;
          box-shadow: var(--cas-card-shadow), inset 0 -3px 0 var(--cas-accent-text) !important;
          color: var(--cas-accent-text) !important;
          transform: translateY(-1px);
        }

        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
          background: var(--cas-accent-border-strong) !important;
          height: 2px !important;
        }

        div[data-testid="stTabs"] button[role="tab"]::after,
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]::after {
          background: var(--cas-accent-border-strong) !important;
          border-color: var(--cas-accent-border-strong) !important;
        }

        button[role="tab"] p {
          color: inherit !important;
          font-weight: inherit !important;
          line-height: 1.25 !important;
          margin: 0 !important;
        }

        div[data-testid="stHorizontalBlock"] {
          gap: 0.8rem;
        }

        div[data-testid="stExpander"] {
          background: var(--cas-panel);
          border: 1px solid var(--cas-border) !important;
          border-radius: 8px !important;
          box-shadow: var(--cas-shadow);
        }

        div[data-testid="stDataFrame"] {
          border: 1px solid var(--cas-border);
          border-radius: 8px;
          overflow: hidden;
        }

        /* CAS controlled text/readability fixes */
        div[data-testid="stMarkdownContainer"],
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li,
        div[data-testid="stCaptionContainer"],
        label,
        span,
        p {
          color: var(--cas-text) !important;
        }

        div[data-testid="stCaptionContainer"],
        .stCaption,
        small {
          color: var(--cas-muted) !important;
        }

        section[data-testid="stSidebar"] {
          background: var(--cas-page-bg) !important;
          color: var(--cas-text) !important;
        }

        section[data-testid="stSidebar"] * {
          color: var(--cas-text) !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] small,
        section[data-testid="stSidebar"] label {
          color: var(--cas-muted) !important;
        }

        /* Native Streamlit widgets */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea,
        input {
          background: var(--cas-panel-strong) !important;
          border-color: var(--cas-border) !important;
          color: var(--cas-text) !important;
        }

        div[data-baseweb="select"] *,
        div[data-baseweb="input"] *,
        textarea,
        input {
          color: var(--cas-text) !important;
        }

        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"] {
          background: var(--cas-page-bg) !important;
          border: 1px solid var(--cas-border) !important;
          color: var(--cas-text) !important;
        }

        div[data-baseweb="popover"] *,
        div[data-baseweb="menu"] *,
        ul[role="listbox"] * {
          color: var(--cas-text) !important;
        }

        /* Dataframe/table readability */
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
          background: var(--cas-panel) !important;
          border: 1px solid var(--cas-border) !important;
          border-radius: 8px !important;
          overflow: hidden !important;
        }

        div[data-testid="stDataFrame"] *,
        div[data-testid="stTable"] * {
          color: var(--cas-text) !important;
        }

        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] thead,
        div[data-testid="stTable"] thead {
          background: var(--cas-panel-strong) !important;
          color: var(--cas-text) !important;
        }

        /* Slider label/readability */
        div[data-testid="stSlider"] label,
        div[data-testid="stSlider"] p,
        div[data-testid="stSlider"] span {
          color: var(--cas-text) !important;
        }

        div[data-testid="stSlider"] [data-testid="stTickBar"] {
          color: var(--cas-muted) !important;
        }

        /* Buttons */
        div.stButton > button,
        div.stDownloadButton > button,
        button[kind],
        [data-testid="stBaseButton-secondary"] {
          background: var(--cas-panel-strong) !important;
          border: 1px solid var(--cas-border) !important;
          color: var(--cas-text) !important;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover,
        button[kind]:hover,
        [data-testid="stBaseButton-secondary"]:hover {
          background: var(--cas-accent-soft) !important;
          border-color: var(--cas-accent-border-strong) !important;
          color: var(--cas-accent-text-strong) !important;
        }

        /* CAS-controlled native Streamlit widgets */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea,
        input {
          background: var(--cas-panel-strong) !important;
          border-color: var(--cas-border) !important;
          color: var(--cas-text) !important;
        }

        div[data-baseweb="select"] *,
        div[data-baseweb="input"] *,
        textarea,
        input {
          color: var(--cas-text) !important;
        }

        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"] {
          background: var(--cas-page-bg) !important;
          border: 1px solid var(--cas-border) !important;
          color: var(--cas-text) !important;
        }

        div[data-baseweb="popover"] *,
        div[data-baseweb="menu"] *,
        ul[role="listbox"] * {
          color: var(--cas-text) !important;
        }

        div.stButton > button,
        div.stDownloadButton > button,
        button[kind],
        [data-testid="stBaseButton-secondary"] {
          background: var(--cas-panel-strong) !important;
          border: 1px solid var(--cas-border) !important;
          color: var(--cas-text) !important;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover,
        button[kind]:hover,
        [data-testid="stBaseButton-secondary"]:hover {
          background: var(--cas-accent-soft) !important;
          border-color: var(--cas-accent-border-strong) !important;
          color: var(--cas-accent-text-strong) !important;
        }

        section[data-testid="stSidebar"] {
          background: var(--cas-page-bg) !important;
          color: var(--cas-text) !important;
          border-right: 1px solid var(--cas-border);
        }

        section[data-testid="stSidebar"] * {
          color: inherit;
        }

        div.stButton > button,
        div.stDownloadButton > button {
          border-radius: 8px !important;
          font-weight: 700;
          min-height: 2.6rem;
          width: 100%;
        }

        div[data-testid="stAlert"] {
          border-radius: 8px;
        }

        .committee-live-note {
          background: var(--cas-card-bg);
          border: 1px solid var(--cas-border-soft);
          border-left: 4px solid var(--cas-neutral);
          border-radius: 10px;
          box-shadow: var(--cas-shadow);
          margin: 0 0 0.55rem 0;
          min-height: 92px;
          padding: 0.82rem 0.95rem;
        }

        .committee-live-note.running {
          border-left-color: var(--cas-warning);
        }

        .committee-live-note.ready {
          border-left-color: var(--cas-accent);
        }

        .committee-live-note.error {
          border-left-color: var(--cas-risk);
        }

        .committee-live-note-badge {
          background: rgba(241, 245, 249, 0.82);
          border: 1px solid rgba(148, 163, 184, 0.34);
          border-radius: 999px;
          color: #475569;
          display: inline-flex;
          font-size: 0.74rem;
          font-weight: 760;
          line-height: 1.25;
          margin-bottom: 0.42rem;
          max-width: 100%;
          padding: 0.18rem 0.5rem;
          word-break: keep-all;
        }

        .committee-live-note-title {
          color: inherit;
          font-size: 0.92rem;
          font-weight: 850;
          line-height: 1.35;
          margin-bottom: 0.24rem;
          word-break: keep-all;
        }

        .committee-live-note-body {
          color: var(--cas-muted);
          font-size: 0.9rem;
          line-height: 1.55;
          word-break: keep-all;
        }

        .committee-live-note-loader {
          align-items: center;
          display: grid;
          gap: 0.55rem;
          grid-template-columns: auto minmax(0, 1fr);
          margin-top: 0.65rem;
        }

        .committee-live-note-spinner,
        .committee-live-loading-spinner {
          animation: cas-live-spin 0.9s linear infinite;
          border: 2px solid rgba(148, 163, 184, 0.28);
          border-radius: 999px;
          border-top-color: var(--cas-warning);
          display: inline-block;
          height: 18px;
          width: 18px;
        }

        .committee-live-note-progress {
          background: rgba(148, 163, 184, 0.16);
          border-radius: 999px;
          display: block;
          height: 7px;
          overflow: hidden;
        }

        .committee-live-note-progress span {
          animation: cas-live-progress 1.35s ease-in-out infinite;
          background: linear-gradient(90deg, var(--cas-warning), var(--cas-neutral));
          border-radius: inherit;
          display: block;
          height: 100%;
          width: 42%;
        }

        .committee-live-loading-screen {
          background: var(--cas-card-bg);
          border: 1px solid var(--cas-border-soft);
          border-left: 4px solid var(--cas-warning);
          border-radius: 10px;
          box-shadow: var(--cas-card-shadow);
          margin: 0.85rem 0 1rem 0;
          padding: 1.05rem 1.15rem;
        }

        .committee-live-loading-header {
          align-items: start;
          display: grid;
          gap: 0.82rem;
          grid-template-columns: auto minmax(0, 1fr);
        }

        .committee-live-loading-spinner {
          height: 24px;
          margin-top: 0.1rem;
          width: 24px;
        }

        .committee-live-loading-title {
          color: var(--cas-text);
          font-size: 1rem;
          font-weight: 900;
          line-height: 1.4;
          margin-bottom: 0.18rem;
          word-break: keep-all;
        }

        .committee-live-loading-body {
          color: var(--cas-muted);
          font-size: 0.92rem;
          line-height: 1.58;
          word-break: keep-all;
        }

        .committee-live-loading-steps {
          display: grid;
          gap: 0.65rem;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          margin-top: 1rem;
        }

        .committee-live-loading-steps div {
          background: var(--cas-live-step-bg);
          border: 1px solid var(--cas-border-soft);
          border-radius: 8px;
          padding: 0.72rem 0.8rem;
        }

        .committee-live-loading-steps span {
          animation: cas-live-pulse 1.35s ease-in-out infinite;
          background: rgba(148, 163, 184, 0.22);
          border-radius: 999px;
          display: block;
          height: 7px;
          margin-bottom: 0.48rem;
          width: 54%;
        }

        .committee-live-loading-steps div:nth-child(2) span {
          animation-delay: 0.18s;
          width: 68%;
        }

        .committee-live-loading-steps div:nth-child(3) span {
          animation-delay: 0.36s;
          width: 46%;
        }

        .committee-live-loading-steps p {
          color: var(--cas-text);
          font-size: 0.86rem;
          font-weight: 780;
          line-height: 1.35;
          margin: 0;
          word-break: keep-all;
        }

        @keyframes cas-live-spin {
          to {
            transform: rotate(360deg);
          }
        }

        @keyframes cas-live-progress {
          0% {
            transform: translateX(-110%);
          }
          50% {
            transform: translateX(52%);
          }
          100% {
            transform: translateX(220%);
          }
        }

        @keyframes cas-live-pulse {
          0%,
          100% {
            opacity: 0.42;
          }
          50% {
            opacity: 1;
          }
        }

        @media (max-width: 900px) {
          .committee-live-loading-steps {
            grid-template-columns: 1fr;
          }
        }

        .market-search-panel {
          background:
            linear-gradient(135deg, var(--cas-accent-surface), rgba(255, 255, 255, 0.02)),
            var(--cas-panel);
          border: 1px solid var(--cas-accent-border);
          border-left: 6px solid var(--cas-accent);
          border-radius: 12px;
          box-shadow: var(--cas-shadow);
          margin: 0.9rem 0 1rem 0;
          padding: 1.1rem 1.2rem;
        }

        .market-search-eyebrow {
          color: var(--cas-accent);
          font-size: 0.78rem;
          font-weight: 850;
          letter-spacing: 0.04em;
          margin-bottom: 0.25rem;
          text-transform: uppercase;
        }

        .market-search-panel h2 {
          border-top: 0;
          font-size: 1.24rem !important;
          line-height: 1.35;
          margin: 0 0 0.35rem 0 !important;
          padding-top: 0 !important;
        }

        .market-search-panel p {
          color: var(--cas-muted);
          font-size: 0.9rem;
          line-height: 1.55;
          margin: 0;
        }

        .market-search-chips {
          display: flex;
          flex-wrap: wrap;
          gap: 0.45rem;
          margin-top: 0.85rem;
        }

        .market-search-chip {
          background: var(--cas-accent-soft);
          border: 1px solid var(--cas-accent-border);
          border-radius: 999px;
          color: var(--cas-accent-text);
          font-size: 0.78rem;
          font-weight: 800;
          padding: 0.28rem 0.62rem;
        }

        .landing-filter-summary {
          background: var(--cas-panel);
          border: 1px solid var(--cas-border-soft);
          border-radius: 10px;
          color: var(--cas-muted);
          font-size: 0.9rem;
          line-height: 1.56;
          margin: 0.72rem 0 0.65rem 0;
          padding: 0.72rem 0.85rem;
          word-break: keep-all;
        }

        .landing-filter-summary b {
          color: inherit;
          font-weight: 900;
        }

        .landing-section-title {
          color: inherit;
          font-size: 1.02rem;
          font-weight: 850;
          margin: 1.05rem 0 0.2rem 0;
          word-break: keep-all;
        }

        .landing-section-caption {
          color: var(--cas-muted);
          font-size: 0.9rem;
          line-height: 1.55;
          margin-bottom: 0.65rem;
          word-break: keep-all;
        }

        .market-card {
          background: var(--cas-panel);
          border: 1px solid var(--cas-border);
          border-left: 5px solid var(--cas-risk);
          border-radius: 8px;
          box-shadow: var(--cas-shadow);
          min-height: 148px;
          padding: 0.9rem 1rem;
        }

        .market-card.explore {
          background: var(--cas-accent-soft);
          border-color: var(--cas-accent-border);
          border-left-color: var(--cas-accent);
        }

        .market-card-rank {
          color: var(--cas-risk);
          font-size: 0.78rem;
          font-weight: 800;
          margin-bottom: 0.3rem;
          text-transform: uppercase;
        }

        .market-card-rank.explore {
          color: var(--cas-accent);
          letter-spacing: 0.01em;
          text-transform: none;
        }

        .market-card-title {
          color: inherit;
          font-size: 1.02rem;
          font-weight: 800;
          line-height: 1.35;
          margin-bottom: 0.45rem;
          word-break: keep-all;
        }

        .market-card-meta {
          color: var(--cas-muted);
          display: flex;
          flex-wrap: wrap;
          font-size: 0.86rem;
          gap: 0.35rem;
          line-height: 1.45;
        }

        .market-card-risk {
          color: var(--cas-risk-text);
          font-size: 1.22rem;
          font-weight: 800;
          margin-top: 0.55rem;
        }

        .market-card-risk.stable {
          color: var(--cas-success);
        }

        .market-card-risk.watch {
          color: var(--cas-warning);
        }

        .market-card-risk.high {
          color: var(--cas-risk);
        }

        .market-card-risk.neutral {
          color: var(--cas-muted);
        }

        .market-card-prob-label {
          color: var(--cas-muted);
          font-size: 0.78rem;
          font-weight: 750;
          margin-top: 0.55rem;
        }

        .market-card-band {
          border: 1px solid var(--cas-neutral-border);
          border-radius: 999px;
          display: inline-block;
          font-size: 0.78rem;
          font-weight: 800;
          margin-top: 0.45rem;
          padding: 0.22rem 0.55rem;
        }

        .market-card-band.stable {
          background: var(--cas-success-soft);
          border-color: var(--cas-success-border);
          color: var(--cas-success);
        }

        .market-card-band.watch {
          background: var(--cas-warning-soft);
          border-color: var(--cas-warning-border);
          color: var(--cas-warning);
        }

        .market-card-band.high {
          background: var(--cas-risk-soft);
          border-color: var(--cas-risk-border);
          color: var(--cas-risk);
        }

        .market-card-band.neutral {
          background: var(--cas-neutral-soft);
          color: var(--cas-text);
        }

        .market-section-title {
          color: inherit;
          font-size: 1rem;
          font-weight: 800;
          margin: 0.2rem 0 0.45rem 0;
        }

        .selected-company-hero {
          align-items: stretch;
          background:
            linear-gradient(135deg, var(--cas-accent-surface), rgba(255, 255, 255, 0.02)),
            var(--cas-panel);
          border: 1px solid var(--cas-accent-border);
          border-left: 6px solid var(--cas-accent);
          border-radius: 12px;
          box-shadow: var(--cas-shadow);
          display: flex;
          flex-direction: column;
          justify-content: center;
          margin: 0.1rem 0 1rem 0;
          min-height: 12.4rem;
          overflow: hidden;
          padding: 1.22rem 1.26rem;
          position: relative;
        }

        .selected-company-hero::before {
          display: none;
        }

        .selected-company-eyebrow {
          color: var(--cas-accent);
          font-size: 0.76rem;
          font-weight: 900;
          letter-spacing: 0.02em;
          margin-bottom: 0.28rem;
        }

        .selected-company-title {
          color: inherit;
          font-size: clamp(1.45rem, 2.1vw, 1.9rem);
          font-weight: 900;
          letter-spacing: 0;
          line-height: 1.22;
        }

        .selected-company-subtitle {
          color: var(--cas-muted);
          font-size: 0.95rem;
          line-height: 1.58;
          margin-top: 0.45rem;
          max-width: 780px;
          word-break: keep-all;
        }

        .selected-company-chip-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.45rem;
          margin-top: 0.85rem;
        }

        .selected-company-chip {
          align-items: center;
          background: var(--cas-accent-soft);
          border: 1px solid var(--cas-accent-border);
          border-radius: 999px;
          color: var(--cas-accent-text);
          display: inline-flex;
          font-size: 0.8rem;
          font-weight: 800;
          min-height: 1.95rem;
          padding: 0.32rem 0.72rem;
        }

        .selected-company-action-panel {
          background:
            linear-gradient(135deg, var(--cas-accent-surface), rgba(255, 255, 255, 0.02)),
            var(--cas-panel);
          border: 1px solid var(--cas-accent-border);
          border-left: 6px solid var(--cas-accent);
          border-radius: 12px;
          box-shadow: var(--cas-shadow);
          margin: 0.1rem 0 0.62rem 0;
          min-height: 9.2rem;
          overflow: hidden;
          padding: 1rem 1.05rem;
          position: relative;
        }

        .selected-company-action-panel::before {
          display: none;
        }

        .selected-company-action-label {
          color: var(--cas-accent);
          font-size: 0.76rem;
          font-weight: 900;
          letter-spacing: 0.02em;
          line-height: 1.28;
          margin-bottom: 0.35rem;
          word-break: keep-all;
        }

        .selected-company-action-title {
          color: var(--cas-text);
          font-size: clamp(1.18rem, 1.8vw, 1.48rem);
          font-weight: 900;
          line-height: 1.2;
          margin-bottom: 0.48rem;
          word-break: keep-all;
        }

        .selected-company-action-body {
          color: var(--cas-muted);
          font-size: 0.9rem;
          line-height: 1.54;
          word-break: keep-all;
        }

        .selected-company-signal {
          background: var(--cas-panel);
          border: 1px solid var(--cas-border-soft);
          border-radius: 10px;
          box-shadow: none;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          min-height: 154px;
          padding: 0.95rem 1rem;
        }

        .selected-company-signal-label {
          color: var(--cas-muted);
          font-size: 0.82rem;
          font-weight: 850;
          margin-bottom: 0.25rem;
        }

        .selected-company-signal-value {
          color: inherit;
          font-size: clamp(1.28rem, 1.8vw, 1.58rem);
          font-weight: 900;
          letter-spacing: 0;
          line-height: 1.16;
        }

        .selected-company-signal-caption {
          color: var(--cas-muted);
          font-size: 0.84rem;
          line-height: 1.45;
          margin-top: 0.35rem;
          word-break: keep-all;
        }

        .selected-company-info-grid {
          display: grid;
          gap: 0.45rem;
          grid-template-columns: 1fr 1fr;
          margin-top: 0.82rem;
        }

        .selected-company-info-grid div {
          background: var(--cas-card-bg);
          border: 1px solid var(--cas-border-soft);
          border-radius: 8px;
          min-height: 3.05rem;
          padding: 0.54rem 0.62rem;
        }

        .selected-company-info-grid span {
          color: var(--cas-muted);
          display: block;
          font-size: 0.72rem;
          font-weight: 780;
          line-height: 1.25;
          margin-bottom: 0.16rem;
          word-break: keep-all;
        }

        .selected-company-info-grid b {
          color: var(--cas-text);
          display: block;
          font-size: 0.9rem;
          font-weight: 870;
          line-height: 1.3;
          word-break: keep-all;
        }

        .selected-company-info-band.high {
          color: var(--cas-warning);
        }

        .selected-company-info-band.watch {
          color: var(--cas-warning);
        }

        .selected-company-info-band.stable {
          color: var(--cas-success);
        }

        .selected-company-badge-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.45rem;
          margin-top: 0.85rem;
        }

        .selected-company-badge {
          border: 1px solid var(--cas-border);
          border-radius: 999px;
          font-size: 0.8rem;
          font-weight: 850;
          padding: 0.32rem 0.65rem;
        }

        .selected-company-badge.stable {
          background: var(--cas-success-soft);
          border-color: var(--cas-success-border);
          color: var(--cas-success);
        }

        .selected-company-badge.watch {
          background: var(--cas-warning-soft);
          border-color: var(--cas-warning-border);
          color: var(--cas-warning);
        }

        .selected-company-badge.high {
          background: var(--cas-warning-soft);
          border-color: var(--cas-warning-border);
          color: var(--cas-warning);
        }

        .selected-company-badge.neutral {
          background: var(--cas-neutral-soft);
          color: var(--cas-text);
        }

        @media (max-width: 900px) {
          .selected-company-hero {
            grid-template-columns: 1fr;
          }
        }

        .committee-decision-strip {
          background: var(--cas-card-bg);
          border: 1px solid var(--cas-border-soft);
          border-radius: var(--cas-card-radius);
          box-shadow: var(--cas-card-shadow), inset 0 3px 0 var(--cas-border-soft);
          margin: 0.4rem 0 0.75rem 0;
          padding: 1rem 1.05rem;
        }

        .committee-decision-topline {
          align-items: center;
          display: flex;
          flex-wrap: wrap;
          gap: 0.65rem;
          margin-bottom: 0.55rem;
        }

        .committee-decision-label {
          color: var(--cas-muted);
          font-size: 0.88rem;
          font-weight: 800;
        }

        .committee-decision-summary {
          color: inherit;
          font-size: 1rem;
          font-weight: 700;
          line-height: 1.62;
          margin: 0;
          word-break: keep-all;
        }

        .committee-review-hero {
          background:
            linear-gradient(135deg, var(--cas-accent-surface), rgba(255, 255, 255, 0.02)),
            var(--cas-panel);
          border: 1px solid var(--cas-accent-border);
          border-left: 6px solid var(--cas-accent);
          border-radius: 12px;
          box-shadow: var(--cas-shadow);
          margin: 0.35rem 0 0.85rem 0;
          overflow: hidden;
          padding: 1.18rem 1.25rem 1.2rem 1.25rem;
          position: relative;
        }

        .committee-review-hero::before {
          display: none;
        }

        .committee-review-hero.risk,
        .committee-review-hero.watch,
        .committee-review-hero.stable {
          border-left-color: var(--cas-accent);
        }

        .agent-disagreement-card {
          background:
            linear-gradient(135deg, var(--cas-accent-soft), rgba(255, 255, 255, 0.02)),
            var(--cas-panel);
          border: 1px solid var(--cas-accent-border);
          border-left: 5px solid var(--cas-accent);
          border-radius: 10px;
          box-shadow: var(--cas-shadow);
          margin: 0.1rem 0 0.95rem 0;
          padding: 0.95rem 1rem 1rem 1rem;
        }

        .agent-disagreement-card.high {
          background:
            linear-gradient(135deg, rgba(245, 158, 11, 0.13), rgba(255, 255, 255, 0.02)),
            var(--cas-panel);
          border-color: var(--cas-warning-border);
          border-left-color: var(--cas-warning);
        }

        .agent-disagreement-card.medium {
          border-left-color: var(--cas-accent);
        }

        .agent-disagreement-head {
          align-items: flex-start;
          display: grid;
          gap: 0.8rem;
          grid-template-columns: minmax(160px, 0.36fr) minmax(0, 0.64fr);
        }

        .agent-disagreement-eyebrow {
          color: var(--cas-muted);
          font-size: 0.78rem;
          font-weight: 800;
          letter-spacing: 0;
          margin-bottom: 0.35rem;
        }

        .agent-disagreement-title-row {
          align-items: center;
          display: flex;
          flex-wrap: wrap;
          gap: 0.45rem;
        }

        .agent-disagreement-level,
        .agent-disagreement-score {
          align-items: center;
          border-radius: 999px;
          display: inline-flex;
          font-size: 0.82rem;
          font-weight: 850;
          line-height: 1.1;
          padding: 0.28rem 0.62rem;
        }

        .agent-disagreement-level {
          background: var(--cas-accent-surface-strong);
          border: 1px solid var(--cas-accent-border);
          color: var(--cas-accent-text);
        }

        .agent-disagreement-level.high {
          background: var(--cas-warning-soft);
          border-color: var(--cas-warning-border);
          color: var(--cas-warning);
        }

        .agent-disagreement-score {
          background: var(--cas-neutral-soft);
          border: 1px solid var(--cas-neutral-border);
          color: var(--cas-muted);
        }

        .agent-disagreement-summary {
          color: var(--cas-text);
          font-size: 0.93rem;
          font-weight: 700;
          line-height: 1.55;
          word-break: keep-all;
        }

        .agent-disagreement-reasons {
          color: var(--cas-text);
          display: grid;
          font-size: 0.9rem;
          font-weight: 650;
          gap: 0.35rem;
          line-height: 1.52;
          margin: 0.75rem 0 0 0;
          padding-left: 1.05rem;
          word-break: keep-all;
        }

        .agent-disagreement-qa {
          border-radius: 8px;
          color: var(--cas-muted);
          font-size: 0.86rem;
          font-weight: 700;
          line-height: 1.45;
          margin-top: 0.75rem;
          padding: 0.58rem 0.72rem;
          word-break: keep-all;
        }

        .agent-disagreement-qa.qa-on {
          background: var(--cas-accent-soft);
          border: 1px solid var(--cas-accent-border);
          color: var(--cas-accent-text-strong);
        }

        .agent-disagreement-qa.qa-off {
          background: var(--cas-neutral-soft);
          border: 1px solid var(--cas-neutral-border);
        }

        .agent-disagreement-qa span {
          display: block;
          font-weight: 650;
          margin-top: 0.2rem;
        }

        @media (max-width: 760px) {
          .agent-disagreement-head {
            grid-template-columns: minmax(0, 1fr);
          }
        }

        .committee-loading-card {
          align-items: center;
          background:
            linear-gradient(135deg, var(--cas-accent-surface), rgba(255, 255, 255, 0.04)),
            var(--cas-card-bg);
          border: 1px solid var(--cas-border-soft);
          border-radius: var(--cas-card-radius);
          box-shadow: var(--cas-card-shadow);
          display: grid;
          gap: 0.9rem;
          grid-template-columns: auto minmax(0, 1fr);
          margin: 0.35rem 0 1rem 0;
          padding: 1rem 1.05rem;
        }

        .committee-loading-orb {
          align-items: center;
          background: var(--cas-accent-surface-strong);
          border: 1px solid var(--cas-accent-border);
          border-radius: 999px;
          display: inline-flex;
          height: 42px;
          justify-content: center;
          position: relative;
          width: 42px;
        }

        .committee-loading-orb::before {
          animation: committee-loading-spin 1.1s linear infinite;
          border: 3px solid var(--cas-accent-border);
          border-top-color: var(--cas-accent);
          border-radius: 999px;
          content: "";
          height: 24px;
          width: 24px;
        }

        .committee-loading-title {
          color: inherit;
          font-size: 1rem;
          font-weight: 880;
          line-height: 1.4;
          margin-bottom: 0.18rem;
          word-break: keep-all;
        }

        .committee-loading-body {
          color: var(--cas-muted);
          font-size: 0.91rem;
          line-height: 1.55;
          word-break: keep-all;
        }

        @keyframes committee-loading-spin {
          to {
            transform: rotate(360deg);
          }
        }

        .committee-review-layout {
          display: grid;
          gap: 1.35rem;
          grid-template-columns: minmax(0, 1.18fr) minmax(280px, 0.48fr);
        }

        .committee-review-eyebrow {
          color: var(--cas-accent);
          font-size: 0.76rem;
          font-weight: 900;
          letter-spacing: 0.02em;
          margin-bottom: 0.28rem;
        }

        .committee-review-title-row {
          align-items: center;
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin-bottom: 0.48rem;
        }

        .committee-review-title {
          color: inherit;
          font-size: clamp(1.68rem, 2.7vw, 2.28rem);
          font-weight: 900;
          letter-spacing: 0;
          line-height: 1.1;
          margin: 0;
          word-break: keep-all;
        }

        .committee-review-summary {
          color: var(--cas-muted);
          font-size: 1rem;
          line-height: 1.62;
          max-width: 900px;
          word-break: keep-all;
        }

        .committee-review-chip-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.45rem;
          margin-top: 0.9rem;
        }

        .committee-review-chip {
          align-items: center;
          background: var(--cas-accent-soft);
          border: 1px solid var(--cas-accent-border);
          border-radius: 999px;
          color: var(--cas-accent-text);
          display: inline-flex;
          font-size: 0.82rem;
          font-weight: 820;
          min-height: 1.9rem;
          padding: 0.3rem 0.68rem;
        }

        .committee-review-facts {
          align-self: center;
          border-left: 1px solid var(--cas-accent-border);
          display: grid;
          gap: 0.1rem;
          padding-left: 1.1rem;
        }

        .committee-review-facts-title {
          color: var(--cas-accent);
          font-size: 0.82rem;
          font-weight: 900;
          margin-bottom: 0.45rem;
          word-break: keep-all;
        }

        .committee-review-fact-row {
          align-items: center;
          border-top: 1px solid var(--cas-accent-border);
          display: flex;
          gap: 0.7rem;
          justify-content: space-between;
          min-height: 2.25rem;
          padding: 0.35rem 0;
        }

        .committee-review-fact-row:first-of-type {
          border-top: 0;
        }

        .committee-review-fact-label {
          color: var(--cas-muted);
          font-size: 0.82rem;
          font-weight: 760;
          word-break: keep-all;
        }

        .committee-review-fact-value {
          align-items: center;
          display: inline-flex;
          font-size: 0.9rem;
          font-weight: 850;
          justify-content: flex-end;
          text-align: right;
          word-break: keep-all;
        }

        @media (max-width: 900px) {
          .committee-review-layout {
            grid-template-columns: 1fr;
          }

          .committee-review-facts {
            border-left: 0;
            border-top: 1px solid var(--cas-accent-border);
            padding-left: 0;
            padding-top: 0.8rem;
          }
        }

        .committee-stage-scale {
          align-items: center;
          display: grid;
          gap: 0.7rem;
          grid-template-columns: auto minmax(0, 1fr);
          margin: -0.35rem 0 1rem 0;
        }

        .committee-stage-scale-label {
          color: var(--cas-muted);
          font-size: 0.82rem;
          font-weight: 860;
          white-space: nowrap;
        }

        .committee-stage-track {
          background: var(--cas-step-track-bg);
          border: 1px solid var(--cas-accent-border);
          border-radius: 999px;
          display: grid;
          gap: 0.28rem;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          padding: 0.28rem;
        }

        .committee-stage-step {
          align-items: center;
          border-radius: 999px;
          color: var(--cas-muted);
          display: flex;
          flex-direction: column;
          gap: 0.03rem;
          justify-content: center;
          min-height: 2.45rem;
          padding: 0.32rem 0.5rem;
          text-align: center;
        }

        .committee-stage-step span {
          font-size: 0.84rem;
          font-weight: 890;
          line-height: 1.18;
          word-break: keep-all;
        }

        .committee-stage-step small {
          font-size: 0.68rem;
          font-weight: 760;
          line-height: 1.15;
          word-break: keep-all;
        }

        .committee-stage-step.active {
          background: var(--cas-accent-soft);
          box-shadow: inset 0 0 0 1px var(--cas-accent-border);
          color: var(--cas-accent-text);
        }

        @media (max-width: 900px) {
          .committee-stage-scale {
            grid-template-columns: 1fr;
          }

          .committee-stage-track {
            border-radius: 12px;
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
        }

        .committee-highlights-heading {
          color: var(--cas-text);
          font-size: 1rem;
          font-weight: 900;
          line-height: 1.35;
          margin: 0.15rem 0 0.55rem 0;
          word-break: keep-all;
        }

        .committee-highlight-grid {
          display: grid;
          gap: 0.75rem;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          margin: 0.2rem 0 0.85rem 0;
        }

        .committee-highlight-card {
          background:
            linear-gradient(135deg, var(--cas-accent-surface), rgba(255, 255, 255, 0.02)),
            var(--cas-panel);
          border: 1px solid var(--cas-accent-border);
          border-left: 5px solid var(--cas-accent);
          border-radius: 8px;
          box-shadow: var(--cas-shadow);
          min-height: 116px;
          padding: 0.85rem 0.95rem;
        }

        .committee-highlight-card.risk,
        .committee-highlight-card.mitigate,
        .committee-highlight-card.warning {
          border-left-color: var(--cas-accent);
        }

        .committee-highlight-title {
          color: var(--cas-accent);
          font-size: 0.9rem;
          font-weight: 800;
          margin-bottom: 0.45rem;
        }

        .committee-highlight-body {
          color: var(--cas-text);
          font-size: 0.93rem;
          font-weight: 700;
          line-height: 1.55;
          word-break: keep-all;
        }

        .committee-highlight-body ul {
          margin: 0;
          padding-left: 1.05rem;
        }

        .committee-highlight-body li {
          margin-bottom: 0.35rem;
        }

        @media (max-width: 900px) {
          .committee-highlight-grid {
            grid-template-columns: 1fr;
          }
        }

        .committee-metric-guide {
          background: var(--cas-card-bg);
          border: 1px solid var(--cas-border-soft);
          border-radius: var(--cas-card-radius);
          box-shadow: var(--cas-card-shadow), inset 0 3px 0 var(--cas-accent);
          margin: 0.35rem 0 0.95rem 0;
          padding: 1rem 1.05rem;
        }

        .committee-metric-guide-title {
          color: inherit;
          font-size: 1.04rem;
          font-weight: 850;
          line-height: 1.45;
          margin-bottom: 0.35rem;
          word-break: keep-all;
        }

        .committee-metric-guide-body {
          color: var(--cas-muted);
          font-size: 0.94rem;
          line-height: 1.6;
          margin-bottom: 0.8rem;
          word-break: keep-all;
        }

        .committee-metric-grid {
          display: grid;
          gap: 0.75rem;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        }

        .committee-metric-card {
          background: var(--cas-card-bg);
          border: 1px solid var(--cas-border-soft);
          border-radius: 12px;
          box-shadow: inset 0 3px 0 var(--cas-accent);
          padding: 0.85rem 0.95rem;
        }

        .committee-metric-card.risk {
          box-shadow: inset 0 3px 0 var(--cas-risk);
        }

        .committee-metric-card.warning {
          box-shadow: inset 0 3px 0 var(--cas-warning);
        }

        .committee-metric-card.neutral {
          box-shadow: inset 0 3px 0 var(--cas-neutral);
        }

        .committee-metric-label {
          color: var(--cas-muted);
          font-size: 0.84rem;
          font-weight: 800;
          margin-bottom: 0.2rem;
        }

        .committee-metric-value {
          color: inherit;
          font-size: 1.25rem;
          font-weight: 900;
          line-height: 1.3;
          margin-bottom: 0.42rem;
        }

        .committee-metric-card-body {
          color: inherit;
          font-size: 0.9rem;
          line-height: 1.55;
          word-break: keep-all;
        }

        .committee-signal-guide {
          display: grid;
          gap: 0.75rem;
          grid-template-columns: minmax(260px, 1.15fr) repeat(3, minmax(180px, 1fr));
          margin: 0.35rem 0 0.95rem 0;
        }

        .committee-signal-card {
          background: var(--cas-card-bg);
          border: 1px solid var(--cas-border-soft);
          border-radius: var(--cas-card-radius);
          box-shadow: inset 0 3px 0 var(--cas-accent);
          padding: 0.9rem 1rem;
          position: relative;
        }

        .committee-signal-card.risk {
          box-shadow: var(--cas-card-shadow), inset 0 3px 0 var(--cas-risk);
        }

        .committee-signal-card.mitigate {
          box-shadow: var(--cas-card-shadow), inset 0 3px 0 var(--cas-success);
        }

        .committee-signal-card.neutral {
          box-shadow: var(--cas-card-shadow), inset 0 3px 0 var(--cas-neutral);
        }

        .committee-signal-card.warning {
          box-shadow: var(--cas-card-shadow), inset 0 3px 0 var(--cas-warning);
        }

        .committee-signal-card.active {
          background:
            linear-gradient(180deg, var(--cas-accent-soft), transparent 68%),
            var(--cas-panel-strong);
          border-color: var(--cas-neutral-border);
          transform: translateY(-1px);
        }

        .committee-signal-card.active.risk {
          border-color: var(--cas-risk-border);
        }

        .committee-signal-card.active.mitigate {
          border-color: var(--cas-success-border);
        }

        .committee-signal-card.active.warning {
          border-color: var(--cas-warning-border);
        }

        .committee-signal-current-badge {
          background: var(--cas-neutral-soft);
          border: 1px solid var(--cas-neutral-border);
          border-radius: 999px;
          color: var(--cas-text);
          display: inline-block;
          font-size: 0.74rem;
          font-weight: 900;
          margin-bottom: 0.5rem;
          padding: 0.22rem 0.55rem;
        }

        .committee-signal-card.active.risk .committee-signal-current-badge {
          background: var(--cas-risk-soft);
          border-color: var(--cas-risk-border);
          color: var(--cas-risk);
        }

        .committee-signal-card.active.mitigate .committee-signal-current-badge {
          background: var(--cas-success-soft);
          border-color: var(--cas-success-border);
          color: var(--cas-success);
        }

        .committee-signal-card.active.warning .committee-signal-current-badge {
          background: var(--cas-warning-soft);
          border-color: var(--cas-warning-border);
          color: var(--cas-warning);
        }

        .committee-signal-eyebrow {
          color: var(--cas-muted);
          font-size: 0.82rem;
          font-weight: 800;
          margin-bottom: 0.35rem;
        }

        .committee-signal-title {
          color: inherit;
          font-size: 1.02rem;
          font-weight: 850;
          line-height: 1.35;
          margin-bottom: 0.42rem;
          word-break: keep-all;
        }

        .committee-signal-body {
          color: inherit;
          font-size: 0.92rem;
          line-height: 1.58;
          word-break: keep-all;
        }

        .committee-signal-detail {
          color: var(--cas-text);
          font-size: 0.86rem;
          line-height: 1.5;
          margin-top: 0.45rem;
          word-break: keep-all;
        }

        .committee-signal-action {
          color: var(--cas-muted);
          font-size: 0.86rem;
          line-height: 1.48;
          margin-top: 0.5rem;
          word-break: keep-all;
        }

        .committee-detail-flow {
          background: var(--cas-card-bg);
          border: 1px solid var(--cas-border-soft);
          border-radius: var(--cas-card-radius);
          box-shadow: var(--cas-card-shadow);
          margin: 0.25rem 0 1rem 0;
          padding: 1rem 1.05rem;
        }

        .committee-detail-title {
          color: inherit;
          font-size: 1rem;
          font-weight: 800;
          margin-bottom: 0.65rem;
        }

        .committee-detail-section {
          border-top: 1px solid var(--cas-border);
          padding: 0.75rem 0 0.1rem 0;
        }

        .committee-detail-section:first-of-type {
          border-top: 0;
          padding-top: 0;
        }

        .committee-detail-heading {
          color: var(--cas-muted);
          font-size: 0.9rem;
          font-weight: 800;
          margin-bottom: 0.28rem;
        }

        .committee-detail-text,
        .committee-detail-section li {
          color: inherit;
          font-size: 0.97rem;
          line-height: 1.68;
          word-break: keep-all;
        }

        .committee-detail-section ul {
          margin: 0.1rem 0 0 1.1rem;
          padding: 0;
        }

        .committee-section-divider {
          align-items: center;
          display: flex;
          gap: 0.7rem;
          margin: 1.15rem 0 0.55rem 0;
        }

        .committee-section-divider::after {
          background: var(--cas-border-soft);
          content: "";
          flex: 1;
          height: 1px;
        }

        .committee-section-kicker {
          color: var(--cas-accent);
          font-size: 0.78rem;
          font-weight: 900;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }

        hr {
          margin: 1rem 0;
        }

        @media (max-width: 900px) {
          .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
          }

          div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
          }

          div[data-testid="stTabs"] [role="tablist"] {
            overflow-x: auto;
            white-space: nowrap;
          }

          .committee-signal-guide {
            grid-template-columns: 1fr;
          }
        }

        /* ------------------------------------------------------------------
        CAS dark-mode hard fixes for Streamlit native widgets
        ------------------------------------------------------------------ */

        section[data-testid="stSidebar"] {
          background: var(--cas-page-bg) !important;
          color: var(--cas-text) !important;
          border-right: 1px solid var(--cas-border) !important;
        }

        section[data-testid="stSidebar"] * {
          color: var(--cas-text) !important;
        }

        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] small,
        section[data-testid="stSidebar"] p {
          color: var(--cas-muted) !important;
        }

        /* Selectbox closed state */
        div[data-baseweb="select"] > div {
          background: var(--cas-panel-strong) !important;
          border-color: var(--cas-border) !important;
          color: var(--cas-text) !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {
          color: var(--cas-text) !important;
        }

        /* Selectbox opened dropdown / popover */
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"],
        div[role="listbox"] {
          background: var(--cas-panel-strong) !important;
          border: 1px solid var(--cas-border) !important;
          color: var(--cas-text) !important;
        }

        div[data-baseweb="popover"] *,
        div[data-baseweb="menu"] *,
        ul[role="listbox"] *,
        div[role="listbox"] * {
          background-color: transparent !important;
          color: var(--cas-text) !important;
        }

        li[role="option"],
        div[role="option"] {
          background: var(--cas-panel-strong) !important;
          color: var(--cas-text) !important;
        }

        li[role="option"]:hover,
        div[role="option"]:hover,
        li[aria-selected="true"],
        div[aria-selected="true"] {
          background: var(--cas-accent-soft) !important;
          color: var(--cas-accent-text-strong) !important;
        }

        /* Expander header / expanded top bar */
        details[data-testid="stExpander"],
        div[data-testid="stExpander"] {
          background: var(--cas-panel) !important;
          border: 1px solid var(--cas-border) !important;
          color: var(--cas-text) !important;
        }

        details[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"],
        div[data-testid="stExpander"] > div:first-child {
          background: var(--cas-panel-strong) !important;
          color: var(--cas-text) !important;
          border-color: var(--cas-border) !important;
        }

        div[data-testid="stExpander"] * {
          color: var(--cas-text) !important;
        }

        /* Tabs and overflow arrow */
        div[data-testid="stTabs"] {
          color: var(--cas-text) !important;
        }

        div[data-testid="stTabs"] button {
          background: var(--cas-panel-strong) !important;
          color: var(--cas-text) !important;
          border-color: var(--cas-border) !important;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
          background: var(--cas-tab-selected-bg) !important;
          color: var(--cas-accent-text-strong) !important;
          border-color: var(--cas-accent-border-strong) !important;
        }

        div[data-testid="stTabs"] button svg,
        div[data-testid="stTabs"] svg,
        button[aria-label*="scroll"] svg,
        button[title*="scroll"] svg {
          color: var(--cas-accent-text-strong) !important;
          fill: var(--cas-accent-text-strong) !important;
          stroke: var(--cas-accent-text-strong) !important;
        }

        div[data-testid="stTabs"] button:last-child,
        div[data-testid="stTabs"] [data-baseweb="button"] {
          background: var(--cas-panel-strong) !important;
          color: var(--cas-accent-text-strong) !important;
          border-color: var(--cas-border) !important;
        }

        /* Native text inputs, number inputs, textarea */
        input,
        textarea,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
          background: var(--cas-panel-strong) !important;
          border-color: var(--cas-border) !important;
          color: var(--cas-text) !important;
        }

        input::placeholder,
        textarea::placeholder {
          color: var(--cas-muted) !important;
        }

        /* Streamlit dataframe/table shell */
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
          background: var(--cas-panel) !important;
          border: 1px solid var(--cas-border) !important;
          border-radius: 8px !important;
          overflow: hidden !important;
        }

        div[data-testid="stDataFrame"] *,
        div[data-testid="stTable"] * {
          color: var(--cas-text) !important;
        }

        div[data-testid="stDataFrame"] canvas {
          background: var(--cas-panel) !important;
        }

        /* HTML tables rendered from pandas Styler */
        table,
        .dataframe,
        div[data-testid="stMarkdownContainer"] table {
          background: var(--cas-panel) !important;
          color: var(--cas-text) !important;
          border-color: var(--cas-border) !important;
        }

        table thead,
        .dataframe thead,
        table th,
        .dataframe th {
          background: var(--cas-panel-strong) !important;
          color: var(--cas-text) !important;
          border-color: var(--cas-border) !important;
        }

        table tbody,
        table tbody tr,
        table tbody td,
        .dataframe tbody,
        .dataframe tbody tr,
        .dataframe tbody td {
          background: var(--cas-panel) !important;
          color: var(--cas-text) !important;
          border-color: var(--cas-border-soft) !important;
        }

        /* Keep direction-badge cells visible even when pandas Styler injects pale colors */
        table tbody td[style*="background-color"],
        .dataframe tbody td[style*="background-color"] {
          color: var(--cas-text) !important;
        }

        </style>
        """
    
    css = css.replace("__CAS_THEME_VARS__", cas_theme_vars)
    st.markdown(css, unsafe_allow_html=True)
    
