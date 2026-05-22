"""Theme CSS for the Streamlit credit dashboard."""

from __future__ import annotations

import streamlit as st


def inject_dashboard_theme() -> None:
    """Apply dashboard styling without forcing a fixed light theme.

    Streamlit's Settings menu changes the app theme on the client side.  CSS
    inserted through ``st.markdown`` cannot reliably read that setting in every
    Streamlit release, so this stylesheet deliberately avoids hard-coded page
    backgrounds.  Custom CAS cards are rendered as subtle currentColor-based
    translucent surfaces, which makes them follow both Streamlit light and dark
    themes automatically.
    """
    st.markdown(
        """
        <style>
        :root,
        .stApp {
          color-scheme: light dark;
          --cas-blue: var(--st-primary-color, var(--primary-color, #1d4ed8));
          --cas-risk: #c85050;
          --cas-risk-text: #d14a4a;
          --cas-success: #2f9e5b;
          --cas-warning: #b7791f;
          --cas-neutral: #4f6fad;
          --cas-text: inherit;
          --cas-muted: currentColor;
          --cas-panel: rgba(128, 128, 128, 0.08);
          --cas-panel-strong: rgba(128, 128, 128, 0.12);
          --cas-border: rgba(128, 128, 128, 0.28);
          --cas-border-soft: rgba(128, 128, 128, 0.18);
          --cas-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
          --cas-card-bg: rgba(128, 128, 128, 0.06);
          --cas-card-radius: 14px;
          --cas-card-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
          --cas-risk-soft: rgba(200, 80, 80, 0.14);
          --cas-risk-border: rgba(200, 80, 80, 0.38);
          --cas-success-soft: rgba(47, 158, 91, 0.14);
          --cas-success-border: rgba(47, 158, 91, 0.38);
          --cas-warning-soft: rgba(183, 121, 31, 0.14);
          --cas-warning-border: rgba(183, 121, 31, 0.38);
          --cas-neutral-soft: rgba(128, 128, 128, 0.10);
          --cas-neutral-border: rgba(128, 128, 128, 0.28);
        }

        @supports (color: color-mix(in srgb, white, black)) {
          :root,
          .stApp {
            --cas-muted: color-mix(in srgb, currentColor 64%, transparent);
            --cas-panel: color-mix(in srgb, currentColor 5%, transparent);
            --cas-panel-strong: color-mix(in srgb, currentColor 8%, transparent);
            --cas-border: color-mix(in srgb, currentColor 18%, transparent);
            --cas-border-soft: color-mix(in srgb, currentColor 10%, transparent);
            --cas-shadow: 0 1px 2px color-mix(in srgb, currentColor 13%, transparent);
            --cas-card-bg: color-mix(in srgb, currentColor 4%, transparent);
            --cas-card-shadow: 0 8px 24px color-mix(in srgb, currentColor 9%, transparent);
            --cas-risk-soft: color-mix(in srgb, var(--cas-risk) 17%, transparent);
            --cas-risk-border: color-mix(in srgb, var(--cas-risk) 42%, transparent);
            --cas-success-soft: color-mix(in srgb, var(--cas-success) 17%, transparent);
            --cas-success-border: color-mix(in srgb, var(--cas-success) 42%, transparent);
            --cas-warning-soft: color-mix(in srgb, var(--cas-warning) 17%, transparent);
            --cas-warning-border: color-mix(in srgb, var(--cas-warning) 42%, transparent);
            --cas-neutral-soft: color-mix(in srgb, currentColor 7%, transparent);
            --cas-neutral-border: color-mix(in srgb, currentColor 18%, transparent);
          }
        }

        .stApp,
        div[data-testid="stAppViewContainer"],
        div[data-testid="stMain"],
        div[data-testid="stMainBlockContainer"],
        .main,
        .main .block-container {
          color: inherit !important;
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
          background: var(--cas-panel);
          border: 1px solid var(--cas-border-soft);
          border-radius: 999px;
          box-shadow: var(--cas-shadow);
          gap: 0.15rem;
          max-width: 100%;
          padding: 0.28rem;
          position: sticky;
          top: 0;
          z-index: 10;
        }

        button[role="tab"] {
          border: 1px solid transparent !important;
          border-radius: 999px !important;
          color: var(--cas-muted) !important;
          font-size: 0.92rem !important;
          font-weight: 700 !important;
          min-height: 2.28rem;
          padding: 0.42rem 0.8rem !important;
          transition: background 0.18s ease, color 0.18s ease, border-color 0.18s ease;
        }

        button[role="tab"][aria-selected="true"] {
          background: rgba(224, 242, 254, 0.72) !important;
          border-color: rgba(2, 132, 199, 0.24) !important;
          box-shadow: none;
          color: #0284c7 !important;
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

        .market-search-panel {
          background:
            linear-gradient(135deg, rgba(224, 242, 254, 0.48), rgba(255, 255, 255, 0.02)),
            var(--cas-panel);
          border: 1px solid rgba(2, 132, 199, 0.24);
          border-left: 6px solid #0284c7;
          border-radius: 12px;
          box-shadow: var(--cas-shadow);
          margin: 0.9rem 0 1rem 0;
          padding: 1.1rem 1.2rem;
        }

        .market-search-eyebrow {
          color: #0284c7;
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
          background: rgba(2, 132, 199, 0.10);
          border: 1px solid rgba(2, 132, 199, 0.20);
          border-radius: 999px;
          color: #0369a1;
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
          background: rgba(224, 242, 254, 0.38);
          border-color: rgba(2, 132, 199, 0.24);
          border-left-color: #0284c7;
        }

        .market-card-rank {
          color: var(--cas-risk);
          font-size: 0.78rem;
          font-weight: 800;
          margin-bottom: 0.3rem;
          text-transform: uppercase;
        }

        .market-card-rank.explore {
          color: #0284c7;
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
            radial-gradient(circle at 8% 8%, rgba(14, 165, 233, 0.16), transparent 30%),
            linear-gradient(135deg, rgba(224, 242, 254, 0.42), rgba(255, 255, 255, 0.02)),
            var(--cas-panel);
          border: 1px solid rgba(2, 132, 199, 0.22);
          border-radius: 16px;
          box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
          display: grid;
          gap: 1rem;
          grid-template-columns: minmax(0, 1fr) minmax(280px, 0.42fr);
          margin: 0 0 1rem 0;
          overflow: hidden;
          padding: 1.15rem 1.2rem;
          position: relative;
        }

        .selected-company-hero::before {
          background: linear-gradient(180deg, #0284c7, #38bdf8);
          content: "";
          inset: 0 auto 0 0;
          position: absolute;
          width: 6px;
        }

        .selected-company-eyebrow {
          color: #0284c7;
          font-size: 0.76rem;
          font-weight: 900;
          letter-spacing: 0.08em;
          margin-bottom: 0.28rem;
          text-transform: uppercase;
        }

        .selected-company-title {
          color: inherit;
          font-size: clamp(1.35rem, 2vw, 1.78rem);
          font-weight: 900;
          letter-spacing: -0.02em;
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
          background: rgba(255, 255, 255, 0.46);
          border: 1px solid rgba(2, 132, 199, 0.18);
          border-radius: 999px;
          color: var(--cas-text);
          display: inline-flex;
          font-size: 0.84rem;
          font-weight: 780;
          min-height: 1.95rem;
          padding: 0.32rem 0.72rem;
        }

        .selected-company-signal {
          background: rgba(255, 255, 255, 0.62);
          border: 1px solid rgba(2, 132, 199, 0.18);
          border-radius: 14px;
          box-shadow: var(--cas-shadow);
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          min-height: 168px;
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
          font-size: clamp(1.45rem, 2.3vw, 2rem);
          font-weight: 900;
          letter-spacing: -0.03em;
          line-height: 1.1;
        }

        .selected-company-signal-caption {
          color: var(--cas-muted);
          font-size: 0.84rem;
          line-height: 1.45;
          margin-top: 0.35rem;
          word-break: keep-all;
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
          background: var(--cas-risk-soft);
          border-color: var(--cas-risk-border);
          color: var(--cas-risk);
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
            linear-gradient(180deg, rgba(224, 242, 254, 0.16), transparent 58%),
            var(--cas-card-bg);
          border: 1px solid var(--cas-border-soft);
          border-radius: var(--cas-card-radius);
          box-shadow: var(--cas-card-shadow);
          margin: 0.35rem 0 1rem 0;
          overflow: hidden;
          padding: 1.15rem 1.2rem;
          position: relative;
        }

        .committee-review-hero::before {
          background: linear-gradient(180deg, #0284c7, #38bdf8);
          content: "";
          height: 4px;
          inset: 0 0 auto 0;
          position: absolute;
          width: auto;
        }

        .committee-review-hero.risk::before {
          background: var(--cas-risk);
        }

        .committee-review-hero.watch::before {
          background: var(--cas-warning);
        }

        .committee-review-hero.stable::before {
          background: var(--cas-success);
        }

        .committee-loading-card {
          align-items: center;
          background:
            linear-gradient(135deg, rgba(224, 242, 254, 0.42), rgba(255, 255, 255, 0.04)),
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
          background: rgba(224, 242, 254, 0.74);
          border: 1px solid rgba(2, 132, 199, 0.24);
          border-radius: 999px;
          display: inline-flex;
          height: 42px;
          justify-content: center;
          position: relative;
          width: 42px;
        }

        .committee-loading-orb::before {
          animation: committee-loading-spin 1.1s linear infinite;
          border: 3px solid rgba(2, 132, 199, 0.18);
          border-top-color: #0284c7;
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
          gap: 1rem;
          grid-template-columns: minmax(0, 1fr) minmax(260px, 0.36fr);
        }

        .committee-review-eyebrow {
          color: var(--cas-muted);
          font-size: 0.76rem;
          font-weight: 900;
          letter-spacing: 0.08em;
          margin-bottom: 0.28rem;
          text-transform: uppercase;
        }

        .committee-review-title {
          color: inherit;
          font-size: clamp(1.28rem, 2vw, 1.7rem);
          font-weight: 900;
          letter-spacing: -0.02em;
          line-height: 1.28;
          margin-bottom: 0.55rem;
          word-break: keep-all;
        }

        .committee-review-summary {
          color: var(--cas-muted);
          font-size: 0.96rem;
          line-height: 1.65;
          max-width: 880px;
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
          background: var(--cas-neutral-soft);
          border: 1px solid var(--cas-neutral-border);
          border-radius: 999px;
          color: var(--cas-text);
          display: inline-flex;
          font-size: 0.82rem;
          font-weight: 820;
          min-height: 1.9rem;
          padding: 0.3rem 0.68rem;
        }

        .committee-review-score {
          background: var(--cas-card-bg);
          border: 1px solid var(--cas-border-soft);
          border-radius: 12px;
          box-shadow: none;
          padding: 0.9rem 0.95rem;
        }

        .committee-review-score-label {
          color: var(--cas-muted);
          font-size: 0.82rem;
          font-weight: 850;
          margin-bottom: 0.25rem;
        }

        .committee-review-score-value {
          color: inherit;
          font-size: clamp(1.42rem, 2.2vw, 1.95rem);
          font-weight: 900;
          letter-spacing: -0.03em;
          line-height: 1.12;
          margin-bottom: 0.45rem;
        }

        .committee-review-score-caption {
          color: var(--cas-muted);
          font-size: 0.84rem;
          line-height: 1.5;
          word-break: keep-all;
        }

        @media (max-width: 900px) {
          .committee-review-layout {
            grid-template-columns: 1fr;
          }
        }

        .committee-highlight-grid {
          display: grid;
          gap: 0.75rem;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          margin: 0.2rem 0 0.95rem 0;
        }

        .committee-highlight-card {
          background: var(--cas-card-bg);
          border: 1px solid var(--cas-border-soft);
          border-radius: var(--cas-card-radius);
          box-shadow: var(--cas-card-shadow), inset 0 3px 0 var(--cas-blue);
          min-height: 132px;
          padding: 0.9rem 1rem;
        }

        .committee-highlight-card.risk {
          box-shadow: var(--cas-card-shadow), inset 0 3px 0 var(--cas-risk);
        }

        .committee-highlight-card.mitigate {
          box-shadow: var(--cas-card-shadow), inset 0 3px 0 var(--cas-success);
        }

        .committee-highlight-card.warning {
          box-shadow: var(--cas-card-shadow), inset 0 3px 0 var(--cas-warning);
        }

        .committee-highlight-title {
          color: var(--cas-muted);
          font-size: 0.9rem;
          font-weight: 800;
          margin-bottom: 0.45rem;
        }

        .committee-highlight-body {
          color: inherit;
          font-size: 0.96rem;
          font-weight: 700;
          line-height: 1.6;
          word-break: keep-all;
        }

        .committee-highlight-body ul {
          margin: 0;
          padding-left: 1.05rem;
        }

        .committee-highlight-body li {
          margin-bottom: 0.35rem;
        }

        .committee-metric-guide {
          background: var(--cas-card-bg);
          border: 1px solid var(--cas-border-soft);
          border-radius: var(--cas-card-radius);
          box-shadow: var(--cas-card-shadow), inset 0 3px 0 var(--cas-blue);
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
          box-shadow: inset 0 3px 0 var(--cas-blue);
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
          box-shadow: var(--cas-card-shadow), inset 0 3px 0 var(--cas-blue);
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
            linear-gradient(180deg, rgba(224, 242, 254, 0.22), transparent 68%),
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
          color: #0284c7;
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
        </style>
        """,
        unsafe_allow_html=True,
    )
