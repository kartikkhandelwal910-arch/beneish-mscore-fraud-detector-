"""
visualization.py
-----------------
Dashboard layer for the Beneish M-Score screen: a multi-year trend chart,
a cross-sectional comparison bar chart, a per-company radar chart, and a
summary table — all styled consistently and all fraud-case-aware (known
historical fraud tickers get annotated automatically, no manual step).

Styling note: every chart here now uses the same "ledger" palette as the
Streamlit theme (see .streamlit/config.toml) and landing page — dark ink
background, red ink for flagged, muted green for clean, brass/gold for
known historical fraud cases — so a screenshot of any chart matches the
rest of the product instead of Matplotlib's default light theme.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from mscore_calculator import MANIPULATION_THRESHOLD, KNOWN_FRAUD_CASES
except ImportError:
    MANIPULATION_THRESHOLD = -1.78
    KNOWN_FRAUD_CASES = {}

COMPONENTS = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"]

# ---------------------------------------------------------------------------
# Ledger palette — consistent across every chart, the Streamlit theme, and
# the landing page. Change these six values and every chart updates.
# ---------------------------------------------------------------------------
INK = "#12160f"
INK_2 = "#181d16"
PAPER = "#ece4d1"
SLATE = "#8d9284"
RULE = "#2a2f24"

COLOR_FLAGGED = "#c24a3a"      # red ink
COLOR_CLEAN = "#7fae7f"        # muted ledger green
COLOR_THRESHOLD = "#8d9284"    # slate
COLOR_KNOWN_CASE = "#c9a54a"   # brass / gold
FIGSIZE_WIDE = (10, 5.5)
FIGSIZE_SQUARE = (6.5, 6.5)

plt.rcParams.update({
    "figure.facecolor": INK,
    "axes.facecolor": INK_2,
    "savefig.facecolor": INK,
    "axes.edgecolor": SLATE,
    "axes.labelcolor": PAPER,
    "text.color": PAPER,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "grid.color": RULE,
    "font.family": "monospace",
    "legend.facecolor": INK_2,
    "legend.edgecolor": RULE,
    "legend.labelcolor": PAPER,
})


def _is_known_case(ticker: str) -> bool:
    return ticker in KNOWN_FRAUD_CASES


def summary_table(mscore_df: pd.DataFrame) -> pd.DataFrame:
    """
    Latest fiscal year per ticker, one row each, folding in:
      - flag status (fixed academic threshold)
      - sector-relative flag + z-score, if sector data is present
      - known historical fraud-case note, if the ticker matches one
      - any data_warnings surfaced during collection
    Sorted by MScore descending (most suspicious first) — this is the
    single table you'd hand someone as the "results" artifact.
    """
    latest = mscore_df.sort_values("fiscal_year").groupby("ticker").tail(1).copy()
    latest = latest.sort_values("MScore", ascending=False)

    latest["known_fraud_case"] = latest["ticker"].map(KNOWN_FRAUD_CASES).fillna("")

    cols = ["ticker", "fiscal_year"] + COMPONENTS + ["MScore", "flagged"]
    if "data_source" in latest.columns:
        cols += ["data_source"]
    if "sector_zscore" in latest.columns:
        cols += ["sector_group", "sector_zscore", "sector_flagged"]
    if "data_warnings" in latest.columns:
        cols += ["data_warnings"]
    cols += ["known_fraud_case"]
    cols = [c for c in cols if c in latest.columns]

    out = latest[cols].reset_index(drop=True)
    round_cols = COMPONENTS + ["MScore"] + (["sector_zscore"] if "sector_zscore" in out.columns else [])
    out[round_cols] = out[round_cols].round(3)
    return out


def plot_comparison_bar(mscore_df: pd.DataFrame, ax=None):
    """
    Cross-sectional bar chart: latest MScore per ticker, red if flagged /
    green if clean, dashed line at the manipulation threshold. Known
    historical fraud-case tickers get a gold star marker next to their
    label — automatic, no manual annotation step needed.
    """
    latest = mscore_df.sort_values("fiscal_year").groupby("ticker").tail(1).sort_values("MScore")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    colors = [COLOR_FLAGGED if f else COLOR_CLEAN for f in latest["flagged"]]
    bars = ax.barh(latest["ticker"], latest["MScore"], color=colors, edgecolor=INK, height=0.6)

    ax.axvline(MANIPULATION_THRESHOLD, color=COLOR_THRESHOLD, linestyle="--", linewidth=1.2,
               label=f"Manipulation threshold ({MANIPULATION_THRESHOLD})")

    # Star-annotate known fraud cases directly on the y-axis labels
    labels = []
    for t in latest["ticker"]:
        labels.append(f"\u2605 {t}" if _is_known_case(t) else t)
    ax.set_yticks(range(len(latest)))
    ax.set_yticklabels(labels)

    for bar, val in zip(bars, latest["MScore"]):
        ax.text(val + (0.05 if val >= 0 else -0.05), bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left" if val >= 0 else "right",
                fontsize=9, color=PAPER)

    ax.set_xlabel("Beneish M-Score (most recent fiscal year)")
    ax.set_title("M-Score by Company — Cross-Sectional Comparison", color=PAPER)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    if own_fig:
        fig.tight_layout()
        return fig
    return ax


def plot_mscore_trend(mscore_df: pd.DataFrame, tickers=None, ax=None):
    """
    Multi-year M-Score trend line, one line per ticker, instead of only
    the latest-year snapshot — a company can cross the threshold
    gradually, and the trend shows *when* it started drifting, not just
    where it landed.

    Known historical fraud-case tickers are auto-annotated with a star at
    their flagged points; every ticker gets its own color from a fixed
    qualitative palette so the same company is identifiable across charts.
    """
    df = mscore_df.copy()
    if tickers:
        df = df[df["ticker"].isin(tickers)]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # a qualitative palette tuned to read clearly on the dark ledger background
    palette = plt.get_cmap("Set2")
    for i, (ticker, g) in enumerate(df.sort_values("fiscal_year").groupby("ticker")):
        color = palette(i % 8)
        ax.plot(g["fiscal_year"], g["MScore"], marker="o", markersize=4,
                linewidth=1.8, color=color, label=ticker)

        flagged_points = g[g["flagged"]]
        if not flagged_points.empty:
            ax.scatter(flagged_points["fiscal_year"], flagged_points["MScore"],
                       s=70, facecolors="none", edgecolors=COLOR_FLAGGED, linewidths=1.8, zorder=5)

        if _is_known_case(ticker):
            last = g.iloc[-1]
            ax.annotate(f"\u2605 {ticker}", (last["fiscal_year"], last["MScore"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=9,
                        fontweight="bold", color=COLOR_KNOWN_CASE)

    ax.axhline(MANIPULATION_THRESHOLD, color=COLOR_THRESHOLD, linestyle="--", linewidth=1.2,
               label=f"Manipulation threshold ({MANIPULATION_THRESHOLD})")

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Fiscal Year")
    ax.set_ylabel("Beneish M-Score")
    ax.set_title("M-Score Trend Over Time", color=PAPER)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)

    if own_fig:
        fig.tight_layout()
        return fig
    return ax


def plot_radar(mscore_df: pd.DataFrame, ticker: str, fiscal_year=None, ax=None):
    """
    Single-company radar of the 8 M-Score components for one fiscal year
    (defaults to the most recent). Lets you see *which* ratio is driving
    a score instead of just the composite number.
    """
    g = mscore_df[mscore_df["ticker"] == ticker].sort_values("fiscal_year")
    if g.empty:
        raise ValueError(f"No M-Score rows found for ticker '{ticker}'")
    row = g.iloc[-1] if fiscal_year is None else g[g["fiscal_year"] == fiscal_year].iloc[0]

    values = [row[c] for c in COMPONENTS]
    angles = np.linspace(0, 2 * np.pi, len(COMPONENTS), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE, subplot_kw=dict(polar=True))

    color = COLOR_FLAGGED if row["flagged"] else COLOR_CLEAN
    ax.plot(angles, values, color=color, linewidth=2)
    ax.fill(angles, values, color=color, alpha=0.28)
    ax.axhline(1.0, color=COLOR_THRESHOLD, linestyle=":", linewidth=1)  # 1.0 = no YoY change, per-ratio baseline

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(COMPONENTS)
    ax.tick_params(colors=SLATE)

    title = f"{ticker} — FY{int(row['fiscal_year'])} (MScore {row['MScore']:.2f}"
    title += ", FLAGGED)" if row["flagged"] else ")"
    if _is_known_case(ticker):
        title = f"\u2605 {title}"
    ax.set_title(title, pad=20, color=PAPER)

    if own_fig:
        fig.tight_layout()
        return fig
    return ax


def plot_dashboard(mscore_df: pd.DataFrame, top_ticker: str = None):
    """
    One-call combined dashboard figure — trend line (top), comparison bar
    (bottom-left), radar for the top-flagged company (bottom-right).
    One figure you can screenshot or save straight into a resume
    portfolio / README, styled to match the rest of the product.
    """
    latest = mscore_df.sort_values("fiscal_year").groupby("ticker").tail(1).sort_values("MScore", ascending=False)
    if top_ticker is None:
        top_ticker = latest.iloc[0]["ticker"]

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])

    ax_trend = fig.add_subplot(gs[0, :])
    plot_mscore_trend(mscore_df, ax=ax_trend)

    ax_bar = fig.add_subplot(gs[1, 0])
    plot_comparison_bar(mscore_df, ax=ax_bar)

    ax_radar = fig.add_subplot(gs[1, 1], polar=True)
    plot_radar(mscore_df, top_ticker, ax=ax_radar)

    fig.suptitle("Beneish M-Score — Forensic Earnings Screen", fontsize=14, fontweight="bold", color=PAPER)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
