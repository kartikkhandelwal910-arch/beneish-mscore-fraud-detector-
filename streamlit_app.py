"""
streamlit_app.py
-----------------
Interactive Streamlit front-end for the Beneish M-Score Forensic Earnings
Screen. Wraps the existing data_collection / mscore_calculator /
visualization modules with a UI: type in tickers, hit Run, get the same
dashboard the notebook produces, live in the browser.

This file makes NO changes to the pipeline logic itself — it only calls
the already-tested functions from the three existing modules. The only
additions in this version are presentational: a "ledger" visual theme
(see .streamlit/config.toml) and a set of small HTML/CSS render helpers
so results read like a case file rather than a bare dataframe.
"""

import matplotlib
matplotlib.use("Agg")  # headless backend — required before pyplot is imported anywhere

import streamlit as st
import pandas as pd

import data_collection
import mscore_calculator
import visualization

st.set_page_config(
    page_title="Beneish M-Score — Forensic Earnings Screen",
    page_icon="🖋️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Ledger theme — CSS injection
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; }

/* headings use the ledger serif */
h1, h2, h3 { font-family: 'Source Serif 4', serif !important; letter-spacing: -0.01em; }

/* eyebrow label, used above section headers */
.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #c9a54a;
    margin-bottom: 6px;
}
.eyebrow::before { content: "— "; }

/* metrics + numbers get tabular mono, like a ledger */
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8d9284 !important;
}

/* a reusable "verdict stamp" for flagged vs clean companies */
.mscore-stamp {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    letter-spacing: 0.05em;
    padding: 8px 14px;
    border-radius: 3px;
    border: 1px solid;
    margin: 3px 6px 3px 0;
    white-space: nowrap;
}
.mscore-stamp .ticker { font-weight: 600; font-size: 13.5px; }
.mscore-stamp.flagged { color: #d97a6a; border-color: #a3382c; background: rgba(163, 56, 44, 0.12); }
.mscore-stamp.clean   { color: #8fc48f; border-color: #3d6b45; background: rgba(61, 107, 69, 0.12); }
.mscore-stamp .star   { color: #c9a54a; }

/* dataframes / tables get a ledger-card look */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(236,228,209,0.14);
    border-radius: 3px;
}

/* tabs styled like ledger dividers */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid rgba(236,228,209,0.14); }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #8d9284;
}
.stTabs [aria-selected="true"] { color: #c9a54a !important; }

/* case-card, used for the known-fraud grid */
.case-card {
    border: 1px solid rgba(236,228,209,0.14);
    background: #181d16;
    padding: 18px 20px;
    border-radius: 3px;
    height: 100%;
}
.case-card .m {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    color: #d97a6a;
}
.case-card h5 { font-family: 'Source Serif 4', serif; font-size: 16px; margin: 8px 0 6px; color: #ece4d1; }
.case-card p { font-size: 13px; color: #8d9284; margin: 0; }

/* hero banner */
.hero-banner {
    border-bottom: 1px solid rgba(236,228,209,0.14);
    padding-bottom: 22px;
    margin-bottom: 26px;
}
</style>
""", unsafe_allow_html=True)


def render_verdict(ticker: str, m_score: float, flagged: bool, is_known_case: bool = False) -> str:
    """Return HTML for a single stamp badge — used in the scorecard strip."""
    css_class = "flagged" if flagged else "clean"
    label = "FLAGGED" if flagged else "CLEAN"
    star = '<span class="star">\u2605</span> ' if is_known_case else ""
    return (
        f'<span class="mscore-stamp {css_class}">{star}'
        f'<span class="ticker">{ticker}</span> M = {m_score:.2f} · {label}</span>'
    )


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-banner">', unsafe_allow_html=True)
st.markdown('<p class="eyebrow">Forensic Accounting · Live Data</p>', unsafe_allow_html=True)
st.title("Beneish M-Score — Forensic Earnings Screen")
st.caption(
    "Pulls live SEC EDGAR 10-K data, computes the 8-ratio Beneish M-Score, "
    "and flags likely earnings manipulators against both the fixed academic "
    "threshold and a sector-relative z-score."
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="eyebrow">Configuration</p>', unsafe_allow_html=True)

    user_agent = st.text_input(
        "Your name + email (required by SEC)",
        placeholder="Jane Doe (jane.doe@example.com)",
        help="SEC EDGAR requires every request to identify a real contact. "
             "This is sent as the User-Agent header on each request; it is "
             "not stored anywhere by this app.",
    )

    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, AMZN, NVDA, UAA",
    )

    include_sector = st.checkbox("Include sector-relative flagging", value=True)

    run_clicked = st.button("Run Screen", type="primary", use_container_width=True)

    st.divider()
    st.markdown('<p class="eyebrow">Validation Set</p>', unsafe_allow_html=True)
    st.caption(
        "Known historical fraud cases used as a validation sanity check: "
        + ", ".join(sorted(mscore_calculator.KNOWN_FRAUD_CASES.keys()))
    )


# ---------------------------------------------------------------------------
# Cached data pull — re-runs only when tickers/user_agent/include_sector change
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _run_pipeline(tickers: tuple, user_agent: str, include_sector: bool):
    data_collection.USER_AGENT = user_agent
    raw = data_collection.collect_dataset(list(tickers), include_sector=include_sector)
    scored = mscore_calculator.compute_mscore_components(raw)
    return raw, scored


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if not run_clicked and "raw_financials" not in st.session_state:
    st.info("Enter your details in the sidebar and click **Run Screen** to start.")
    st.stop()

if run_clicked:
    tickers = tuple(t.strip().upper() for t in tickers_input.split(",") if t.strip())

    if not user_agent.strip() or "@" not in user_agent:
        st.error("SEC requires a real name and email in the User-Agent field before it will "
                 "accept requests. Please fill that in on the left.")
        st.stop()

    if not tickers:
        st.error("Enter at least one ticker.")
        st.stop()

    with st.spinner(f"Pulling SEC EDGAR data for {len(tickers)} ticker(s)... this can take a "
                     f"minute due to SEC's rate limit."):
        try:
            raw_financials, mscore_df = _run_pipeline(tickers, user_agent.strip(), include_sector)
        except RuntimeError as e:
            st.error(f"Pipeline error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error while collecting or scoring data: {e}")
            st.stop()

    if raw_financials.empty:
        st.error("No data was collected for any of the requested tickers. Check the ticker "
                 "symbols and try again.")
        st.stop()

    st.session_state["raw_financials"] = raw_financials
    st.session_state["mscore_df"] = mscore_df

raw_financials = st.session_state["raw_financials"]
mscore_df = st.session_state["mscore_df"]

n_tickers = mscore_df["ticker"].nunique()
n_requested = raw_financials["ticker"].nunique()

# --- top-line metrics row ---------------------------------------------------
summary_for_metrics = visualization.summary_table(mscore_df)
n_flagged = int(summary_for_metrics["flagged"].sum()) if "flagged" in summary_for_metrics.columns else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Scored", f"{n_tickers} / {n_requested}")
m2.metric("Fiscal Years", mscore_df["fiscal_year"].nunique())
m3.metric("Flagged (fixed)", n_flagged)
m4.metric("Known Cases in Basket", summary_for_metrics["known_fraud_case"].astype(bool).sum()
          if "known_fraud_case" in summary_for_metrics.columns else 0)

# --- verdict strip: one stamp per ticker, most recent fiscal year ----------
st.markdown('<p class="eyebrow">Case Verdicts — Latest Fiscal Year</p>', unsafe_allow_html=True)
stamp_html = "".join(
    render_verdict(
        row["ticker"],
        row["MScore"],
        bool(row["flagged"]),
        is_known_case=bool(row.get("known_fraud_case", "")),
    )
    for _, row in summary_for_metrics.iterrows()
)
st.markdown(f'<div style="margin-bottom: 22px;">{stamp_html}</div>', unsafe_allow_html=True)

tab_dashboard, tab_summary, tab_data, tab_validation = st.tabs(
    ["Dashboard", "Summary Table", "Raw Data & Warnings", "Known Fraud Cases"]
)

# --- Dashboard tab ---------------------------------------------------------
with tab_dashboard:
    summary = visualization.summary_table(mscore_df)
    top_ticker = summary.iloc[0]["ticker"] if not summary.empty else None

    if top_ticker is not None:
        fig = visualization.plot_dashboard(mscore_df, top_ticker=top_ticker)
        st.pyplot(fig)
    else:
        st.warning("Not enough multi-year data to build the dashboard for any ticker.")

# --- Summary table tab ------------------------------------------------------
with tab_summary:
    st.markdown('<p class="eyebrow">Latest Fiscal Year — Summary</p>', unsafe_allow_html=True)
    summary = visualization.summary_table(mscore_df)
    st.dataframe(summary, use_container_width=True)

    flagged = mscore_calculator.flagged_companies(mscore_df)
    st.markdown(
        f'<p class="eyebrow">Flagged on the Academic Threshold '
        f'({mscore_calculator.MANIPULATION_THRESHOLD})</p>',
        unsafe_allow_html=True,
    )
    if flagged.empty:
        st.write("No companies in this basket are flagged on the fixed academic threshold.")
    else:
        st.dataframe(flagged, use_container_width=True)

    if "sector_flagged" in mscore_df.columns:
        latest = mscore_df.sort_values("fiscal_year").groupby("ticker").tail(1)
        sector_flagged = latest[latest["sector_flagged"] == True]  # noqa: E712
        st.markdown('<p class="eyebrow">Sector-Relative Outliers</p>', unsafe_allow_html=True)
        if sector_flagged.empty:
            st.write("No companies are sector-relative outliers in this basket.")
        else:
            st.dataframe(
                sector_flagged[["ticker", "fiscal_year", "MScore", "sector_group", "sector_zscore"]],
                use_container_width=True,
            )

# --- Raw data tab ------------------------------------------------------------
with tab_data:
    st.markdown('<p class="eyebrow">Full M-Score Component Table</p>', unsafe_allow_html=True)
    st.dataframe(mscore_df, use_container_width=True)

    st.markdown('<p class="eyebrow">Data-Quality Warnings</p>', unsafe_allow_html=True)
    data_issues = raw_financials[raw_financials["data_warnings"] != ""]
    st.write(f"{len(data_issues)} of {len(raw_financials)} ticker/fiscal-year rows flagged by validation.")
    if not data_issues.empty:
        st.dataframe(
            data_issues[["ticker", "fiscal_year", "data_warnings"]],
            use_container_width=True,
        )

    st.download_button(
        "Download full M-Score table as CSV",
        data=mscore_df.to_csv(index=False).encode("utf-8"),
        file_name="mscore_results.csv",
        mime="text/csv",
    )

# --- Known fraud case validation tab -----------------------------------------
with tab_validation:
    st.markdown('<p class="eyebrow">Cross-Check Against Known Historical Fraud Cases</p>', unsafe_allow_html=True)
    validation = mscore_calculator.validate_against_known_cases(mscore_df)
    if validation.empty:
        st.write(
            "None of the tickers in this basket match a known historical fraud case "
            f"({', '.join(sorted(mscore_calculator.KNOWN_FRAUD_CASES.keys()))}). "
            "This is expected unless you've included one of those tickers."
        )
    else:
        st.dataframe(validation, use_container_width=True)

    # reference grid of every case the model knows about, regardless of basket
    st.markdown('<p class="eyebrow">Reference — Cases the Model Knows</p>', unsafe_allow_html=True)
    case_items = list(mscore_calculator.KNOWN_FRAUD_CASES.items())
    cols = st.columns(len(case_items)) if case_items else []
    for col, (ticker, description) in zip(cols, case_items):
        with col:
            st.markdown(
                f'<div class="case-card"><div class="m">\u2605 {ticker}</div>'
                f'<h5>{description.split(" — ")[0] if " — " in description else ticker}</h5>'
                f'<p>{description}</p></div>',
                unsafe_allow_html=True,
            )
