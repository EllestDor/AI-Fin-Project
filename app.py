"""
app.py — Streamlit entry point for the Global Macro Asset Allocation Dashboard.

Layout
------
Tab 1  Live Market Pulse      – treemap of recent 5-day sector performance
Tab 2  Correlation Heatmap    – dynamic correlation matrix
Tab 3  Portfolio Sandbox      – AI-augmented Monte Carlo optimiser
Tab 4  Walk-Forward Backtest  – out-of-sample walk-forward analysis
Tab 5  Risk/Return Database   – sortable metrics table
"""

import os
import subprocess

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import (
    MARKET_TICKERS,
    compute_sentiment_alpha,
    fetch_and_calculate_live_data,
    load_nlp_model,
)
from optimizer import run_monte_carlo, walk_forward_backtest

# ============================================================
# 1. Page config  (must be the first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="Global Macro Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. Sidebar – navigation & theme
# ============================================================
st.sidebar.title("Navigation Menu")
st.sidebar.markdown("Welcome to the **Global Macro Asset Allocation** Engine.")
st.sidebar.divider()

theme_choice = st.sidebar.radio("UI Theme:", ["Dark Mode 🌙", "Light Mode ☀️"])

if theme_choice == "Dark Mode 🌙":
    # ── Dark palette ────────────────────────────────────────
    bg_color       = "#0D1117"
    sidebar_bg     = "#161B22"
    card_bg        = "#161B22"
    border_color   = "#30363D"
    text_color     = "#E6EDF3"
    sub_text_color = "#8B949E"
    input_bg       = "#21262D"
    tab_hover      = "rgba(255,255,255,0.07)"
    tab_selected   = "rgba(212,175,55,0.18)"
    accent         = "#D4AF37"
    # chart-specific — gold is legible on dark; near-black makes a good
    # neutral midpoint in the treemap; faint white for reference lines
    chart_accent   = "#D4AF37"
    chart_neutral  = "#1C2128"   # near-bg, soft midpoint for treemap zero band
    chart_hline    = "rgba(255,255,255,0.20)"
else:
    # ── Light palette ───────────────────────────────────────
    bg_color       = "#FFFFFF"
    sidebar_bg     = "#F6F6F4"
    card_bg        = "#F9F9F7"
    border_color   = "#E5E5E3"
    text_color     = "#1A1A1A"
    sub_text_color = "#4A4A4A"   # darkened from #555 → richer contrast on white
    input_bg       = "#F0F0EE"
    tab_hover      = "rgba(0,0,0,0.05)"
    tab_selected   = "rgba(180,138,30,0.12)"
    accent         = "#B8920A"
    # chart-specific — #9B6F00 gives ~4.7:1 on white (WCAG AA); mid-gray for
    # treemap zero band; faint dark line for reference baselines
    chart_accent   = "#9B6F00"
    chart_neutral  = "#C8C8C8"   # light gray midpoint, readable on white
    chart_hline    = "rgba(0,0,0,0.18)"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Vollkorn:ital,wght@0,400;0,600;0,700;1,400&display=swap');

    /* ── Base ─────────────────────────────────────────────── */
    .stApp {{ background-color: {bg_color} !important; }}

    /* Exclude [data-testid="stIconMaterial"] at selector level so Streamlit's
       DynamicIcon spans (expander arrows, sidebar collapse, tab overflow arrows,
       dropdown chevrons — all confirmed to use this single testid) are never
       matched by this rule and their "Material Symbols Rounded" font is preserved. */
    html, body, [class*="css"]:not([data-testid="stIconMaterial"]),
    p, div, li, td, th, label, input, textarea, select {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: {text_color} !important;
    }}
    /* Belt-and-suspenders: explicitly restore the confirmed font name for icon spans
       so that any path that still matches them gets the right font. */
    [data-testid="stIconMaterial"] {{
        font-family: "Material Symbols Rounded" !important;
        font-feature-settings: "liga" !important;
        -webkit-font-feature-settings: "liga" !important;
    }}

    /* ── Page title — responsive, clear of sidebar arrow ──── */
    h1 {{
        font-family: 'Vollkorn', Georgia, serif !important;
        font-size: clamp(1.25rem, 3vw, 1.9rem) !important;
        line-height: 1.25 !important;
        letter-spacing: -0.01em !important;
        color: {text_color} !important;
        /* right gap keeps text away from the sidebar collapse arrow */
        padding-right: 3rem !important;
        margin-bottom: 1.25rem !important;
        word-spacing: 0.06em !important;
    }}

    /* ── Section headings (### with emoji) ────────────────── */
    h2, h3, h4, h5, h6 {{
        font-family: 'Vollkorn', Georgia, serif !important;
        font-size: clamp(0.95rem, 2vw, 1.15rem) !important;
        line-height: 1.4 !important;
        letter-spacing: -0.005em !important;
        color: {text_color} !important;
        /* word-spacing pushes emoji glyphs away from adjacent text */
        word-spacing: 0.12em !important;
        margin-top: 0.25rem !important;
        margin-bottom: 0.6rem !important;
    }}

    /* ── Sidebar ──────────────────────────────────────────── */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color} !important;
    }}
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {{
        color: {sub_text_color} !important;
        font-size: 0.85rem !important;
    }}
    [data-testid="stSidebar"] h1 {{
        color: {text_color} !important;
        font-family: 'Vollkorn', Georgia, serif !important;
        /* Match the h3 section-heading size used by "Global Market Pulse" */
        font-size: clamp(0.95rem, 2vw, 1.15rem) !important;
        font-weight: 700 !important;
        line-height: 1.15 !important;
        letter-spacing: -0.01em !important;
        word-spacing: normal !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        max-width: 100% !important;
        padding-right: 0 !important;
        margin-bottom: 0.5rem !important;
    }}
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: {text_color} !important;
        font-size: clamp(1rem, 2vw, 1.2rem) !important;
        word-spacing: normal !important;
        padding-right: 0 !important;
    }}

    /* ── Caption / helper text ────────────────────────────── */
    [data-testid="stCaptionContainer"] p,
    .stCaption, small {{
        color: {sub_text_color} !important;
        font-size: 0.78rem !important;
        line-height: 1.5 !important;
    }}

    /* ── Metric cards ─────────────────────────────────────── */
    [data-testid="stMetric"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 10px !important;
        padding: 0.85rem 1rem !important;
    }}
    [data-testid="stMetricLabel"] p {{
        color: {sub_text_color} !important;
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        white-space: nowrap !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {text_color} !important;
        font-size: clamp(1.1rem, 3vw, 1.5rem) !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
    }}

    /* ── Tab bar — scrollable on narrow screens ───────────── */
    div[data-testid="stTabs"] > div[role="tablist"] {{
        display: flex !important;
        width: 100% !important;
        /* natural width per tab; scrollable rather than squeezed */
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
        gap: 0.15rem !important;
        border-bottom: 1px solid {border_color} !important;
        padding-bottom: 0 !important;
        margin-bottom: 1.5rem !important;
    }}
    div[data-testid="stTabs"] > div[role="tablist"]::-webkit-scrollbar {{
        display: none !important;
    }}
    button[data-baseweb="tab"] {{
        /* don't stretch to fill — keeps labels from crowding */
        flex: 0 0 auto !important;
        min-width: 0 !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 0.45rem 0.7rem !important;
        background-color: transparent !important;
        border-radius: 6px 6px 0 0 !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.18s ease !important;
        white-space: nowrap !important;
    }}
    button[data-baseweb="tab"] > div {{
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: {sub_text_color} !important;
        font-family: 'Inter', sans-serif !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        max-width: 18ch !important;
    }}
    button[data-baseweb="tab"]:hover {{
        background-color: {tab_hover} !important;
    }}
    button[data-baseweb="tab"]:hover > div {{ color: {text_color} !important; }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        background-color: transparent !important;
        border-bottom: 2px solid {accent} !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] > div {{
        color: {text_color} !important;
        font-weight: 700 !important;
    }}

    /* ── Inputs & selects ─────────────────────────────────── */
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stMultiSelect"] label p,
    div[data-testid="stSlider"] label p {{
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: {sub_text_color} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 0.25rem !important;
    }}
    div[data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="select"] span {{ color: {text_color} !important; }}

    /* ── Expander ─────────────────────────────────────────── */
    [data-testid="stExpander"] {{
        border: 1px solid {border_color} !important;
        border-radius: 10px !important;
        background-color: {card_bg} !important;
    }}
    [data-testid="stExpander"] summary p {{
        color: {text_color} !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }}

    /* ── Info / warning / success banners ─────────────────── */
    [data-testid="stAlert"] p {{
        color: {text_color} !important;
        font-size: 0.85rem !important;
        line-height: 1.55 !important;
    }}

    /* ── Dataframe ────────────────────────────────────────── */
    [data-testid="stDataFrame"] {{
        border-radius: 10px !important;
        overflow: hidden !important;
    }}

    /* ── Mobile breakpoint (≤ 640 px) ────────────────────── */
    @media (max-width: 640px) {{
        h1 {{ font-size: 1.15rem !important; padding-right: 2.5rem !important; }}
        h2, h3 {{ font-size: 0.95rem !important; }}
        button[data-baseweb="tab"] {{ padding: 0.4rem 0.5rem !important; }}
        button[data-baseweb="tab"] > div {{
            font-size: 0.7rem !important;
            max-width: 12ch !important;
        }}
        [data-testid="stMetric"] {{ padding: 0.65rem 0.75rem !important; }}
        [data-testid="stMetricValue"] {{ font-size: 1rem !important; }}
    }}

    </style>
""", unsafe_allow_html=True)

# ============================================================
# 3. Data loading
# ============================================================
st.title("Global Macro Asset Allocation Dashboard")

PLOT_BG = "rgba(0,0,0,0)"


def get_deployed_version() -> tuple[str, str]:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir, stderr=subprocess.DEVNULL,
        ).decode().strip()
        commit_date = subprocess.check_output(
            ["git", "log", "-1", "--format=%cd", "--date=format:%m/%d %H:%M"],
            cwd=repo_dir, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit_hash, commit_date = "unknown", "unknown"
    return commit_hash, commit_date


with st.spinner("Initializing live market data engine..."):
    try:
        live_returns_df, live_summary_df, last_update_str = fetch_and_calculate_live_data()

        st.sidebar.success(f"🟢 Live Data Active\n\nLast Updated: \n{last_update_str}")
        if st.sidebar.button("🔄 Refresh Live Data", use_container_width=True):
            fetch_and_calculate_live_data.clear()
            st.rerun()

        commit_hash, commit_date = get_deployed_version()
        st.sidebar.caption(f"Build `{commit_hash}` · committed {commit_date}")
        st.sidebar.divider()

    except Exception as e:
        st.error(f"Failed to fetch live data from Yahoo Finance. Error: {e}")
        st.stop()

sia = load_nlp_model()

# ============================================================
# 4. Tab layout
# ============================================================
tab_pulse, tab_heatmap, tab_opt, tab_wf, tab_database = st.tabs([
    "Live Market Pulse",
    "Correlation Heatmap",
    "Portfolio Sandbox",
    "Walk-Forward Backtest",
    "Risk/Return Database",
])

# --------------------------------------------------------
# Tab 1 – Live Market Pulse
# --------------------------------------------------------
with tab_pulse:
    st.markdown("### 🌐 Global Market Pulse")

    recent_5d = (1 + live_returns_df.tail(5)).prod() - 1
    perf_df = pd.DataFrame({
        'Sector':      list(MARKET_TICKERS.keys()),
        'Ticker':      list(MARKET_TICKERS.values()),
        'Performance': recent_5d.reindex(MARKET_TICKERS.values()).values,
    })
    perf_df['Weight'] = 1
    perf_df['Label'] = (
        perf_df['Sector'] + "<br>" +
        perf_df['Performance'].apply(lambda x: f"{x*100:.2f}%")
    )

    best   = perf_df.loc[perf_df['Performance'].idxmax()]
    worst  = perf_df.loc[perf_df['Performance'].idxmin()]
    st.info(
        f"**💡 Quant Insight (Trailing 5D):** "
        f"**{best['Sector']}** is leading ({best['Performance']*100:.2f}%), "
        f"while **{worst['Sector']}** is lagging."
    )

    fig_tree = px.treemap(
        perf_df,
        path=[px.Constant("Global Macro Universe"), 'Label'],
        values='Weight',
        color='Performance',
        color_continuous_scale=['#FF4B4B', chart_neutral, '#00C853'],
        color_continuous_midpoint=0,
    )
    fig_tree.update_layout(
        margin=dict(t=20, l=0, r=0, b=0),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        coloraxis_showscale=False,
    )
    fig_tree.update_traces(
        textfont=dict(family="Vollkorn", size=18, color="white"),
        textinfo="label",
    )
    st.plotly_chart(fig_tree, use_container_width=True)


# --------------------------------------------------------
# Tab 2 – Correlation Heatmap
# --------------------------------------------------------
with tab_heatmap:
    st.markdown("### 🔗 Real-time Correlation Matrix")
    st.markdown("Calculated dynamically using the latest 1-year daily returns.")

    selected_tickers = st.multiselect(
        "Select assets to compare:",
        list(MARKET_TICKERS.values()),
        default=['XLK', 'XLV', 'XLF', 'XLE', 'TLT', 'GLD'],
    )

    if len(selected_tickers) > 1:
        corr_matrix = live_returns_df[selected_tickers].corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdYlBu_r", zmin=-0.5, zmax=1,
        )
        fig_corr.update_layout(
            height=600, margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=text_color),
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Please select at least 2 assets.")


# --------------------------------------------------------
# Tab 3 – Portfolio Sandbox
# --------------------------------------------------------
with tab_opt:
    st.markdown("### 🧪 AI-Augmented Portfolio Sandbox")
    st.markdown(
        "Simulate optimal portfolios using **Live Historical Data** "
        "overlayed with NLP Sentiment Alpha."
    )

    col_input1, col_input2 = st.columns([2, 1])
    with col_input1:
        opt_assets = st.multiselect(
            "Select Assets for Optimization:",
            list(MARKET_TICKERS.values()),
            default=['XLK', 'TLT', 'GLD', 'XLE'],
            key='opt_select',
        )
    with col_input2:
        alpha_weight = st.slider("NLP Sentiment Alpha Weight", 0.0, 0.1, 0.05, 0.01)

    if len(opt_assets) >= 2:
        with st.spinner("Running Monte Carlo Simulation..."):
            try:
                finnhub_key = st.secrets.get("FINNHUB_API_KEY", "")
            except Exception:
                finnhub_key = ""

            sentiments, headlines_map = compute_sentiment_alpha(opt_assets, sia, finnhub_key)

            if not finnhub_key:
                st.caption(
                    "Headlines sourced from Yahoo Finance. "
                    "Add a FINNHUB_API_KEY to `.streamlit/secrets.toml` for a broader news feed."
                )

            with st.expander("📰 Headlines driving the sentiment alpha"):
                for ticker in opt_assets:
                    st.markdown(f"**{ticker}** — avg sentiment: `{sentiments[ticker]:+.3f}`")
                    for h in headlines_map[ticker]:
                        st.caption(f"• {h}")
                    if not headlines_map[ticker]:
                        st.caption("• No recent headlines found")

            result = run_monte_carlo(
                live_returns_df, opt_assets,
                sentiment_alpha=sentiments,
                alpha_weight=alpha_weight,
            )

            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(
                x=result.sim_std_devs, y=result.sim_returns,
                mode='markers',
                marker=dict(size=4, color=result.sim_sharpes, colorscale='Viridis', showscale=True),
                name='Simulated Portfolios', hoverinfo='none',
            ))
            fig_opt.add_trace(go.Scatter(
                x=[result.sim_std_devs[result.max_sharpe_idx]],
                y=[result.sim_returns[result.max_sharpe_idx]],
                mode='markers+text',
                marker=dict(color=chart_accent, size=16, symbol='star'),
                name='Max Sharpe', text=['Max Sharpe'], textposition="top center",
            ))
            fig_opt.update_layout(
                xaxis_title="Predicted Volatility (Risk)",
                yaxis_title="AI-Adjusted Expected Return",
                height=400, margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                font=dict(color=text_color),
            )

            best_w = result.weights
            fig_weights = go.Figure(go.Bar(
                x=best_w.values * 100, y=best_w.index, orientation='h',
                marker=dict(color=chart_accent),
                text=[f"{w*100:.1f}%" for w in best_w.values],
                textposition='outside',
            ))
            fig_weights.update_xaxes(range=[0, best_w.values.max() * 100 * 1.25])
            fig_weights.update_layout(
                title="Max Sharpe Allocation", xaxis_title="Weight (%)",
                height=400, margin=dict(l=0, r=20, t=40, b=0),
                plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                font=dict(color=text_color),
            )

            col_frontier, col_weights = st.columns([2, 1])
            with col_frontier:
                st.plotly_chart(fig_opt, use_container_width=True)
            with col_weights:
                st.plotly_chart(fig_weights, use_container_width=True)


# --------------------------------------------------------
# Tab 4 – Walk-Forward Backtest
# --------------------------------------------------------
with tab_wf:
    st.markdown("### 🔄 Walk-Forward Backtest")
    st.markdown(
        "Each window optimises weights on a **training period**, then evaluates "
        "performance on the immediately following **out-of-sample test period**. "
        "No look-ahead bias — the optimiser never sees the test data."
    )

    col_wf1, col_wf2, col_wf3 = st.columns(3)
    with col_wf1:
        wf_assets = st.multiselect(
            "Assets",
            list(MARKET_TICKERS.values()),
            default=['XLK', 'TLT', 'GLD', 'XLE', 'XLV'],
            key='wf_assets',
        )
    with col_wf2:
        train_months = st.selectbox("Training window", [3, 6, 9, 12], index=1)
        train_days   = int(train_months * 21)
    with col_wf3:
        test_months = st.selectbox("Test window", [1, 2, 3], index=0)
        test_days   = int(test_months * 21)

    if len(wf_assets) < 2:
        st.warning("Please select at least 2 assets.")
    else:
        if st.button("▶ Run Walk-Forward Backtest", type="primary"):
            with st.spinner("Running walk-forward optimisation…"):
                try:
                    wf = walk_forward_backtest(
                        live_returns_df, wf_assets,
                        train_window=train_days,
                        test_window=test_days,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()

            # ── Summary metrics ──────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("OOS CAGR",        f"{wf.cagr*100:.2f}%")
            m2.metric("Ann. Volatility", f"{wf.ann_volatility*100:.2f}%")
            m3.metric("Sharpe Ratio",    f"{wf.sharpe_ratio:.2f}")
            m4.metric("Max Drawdown",    f"{wf.max_drawdown*100:.2f}%")

            # ── Cumulative growth chart ──────────────────────────────────
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=wf.cumulative_curve.index,
                y=wf.cumulative_curve.values,
                mode='lines', line=dict(color=chart_accent, width=2),
                name='Walk-Forward Portfolio',
                hovertemplate="%{x|%Y-%m-%d}<br>Growth: %{y:.3f}<extra></extra>",
            ))
            fig_cum.add_hline(y=1.0, line_dash="dot", line_color=chart_hline, opacity=1.0)
            fig_cum.update_layout(
                title="Cumulative OOS Growth (starting at 1.0)",
                xaxis_title="Date", yaxis_title="Portfolio Value",
                height=350, margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                font=dict(color=text_color),
            )
            st.plotly_chart(fig_cum, use_container_width=True)

            # ── Per-window OOS returns bar chart ────────────────────────
            ws = wf.window_summary.reset_index()
            fig_bar = go.Figure(go.Bar(
                x=ws["window"].astype(str),
                y=ws["oos_return"] * 100,
                marker_color=np.where(ws["oos_return"] >= 0, '#00C853', '#FF4B4B'),
                hovertemplate="Window %{x}<br>OOS Return: %{y:.2f}%<extra></extra>",
            ))
            fig_bar.update_layout(
                title="OOS Return per Window",
                xaxis_title="Window #", yaxis_title="Return (%)",
                height=280, margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                font=dict(color=text_color),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Per-window allocation heatmap ────────────────────────────
            weight_cols = [c for c in wf.window_summary.columns if c in wf_assets]
            weight_df   = wf.window_summary[weight_cols].T

            fig_hw = px.imshow(
                weight_df,
                text_auto=".0%",
                color_continuous_scale="YlOrBr",
                aspect="auto",
                labels=dict(x="Window #", y="Asset", color="Weight"),
            )
            fig_hw.update_layout(
                title="Optimal Weights per Window",
                height=max(200, len(wf_assets) * 45),
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                font=dict(color=text_color),
            )
            st.plotly_chart(fig_hw, use_container_width=True)

            # ── Full window table ────────────────────────────────────────
            with st.expander("📋 Window-by-window detail"):
                display_ws = wf.window_summary.copy()
                display_ws["oos_return"] = display_ws["oos_return"].map("{:.2%}".format)
                display_ws["oos_sharpe"] = display_ws["oos_sharpe"].map(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "—"
                )
                for col in weight_cols:
                    display_ws[col] = display_ws[col].map("{:.1%}".format)
                st.dataframe(display_ws, use_container_width=True)


# --------------------------------------------------------
# Tab 5 – Risk/Return Database
# --------------------------------------------------------
with tab_database:
    st.markdown("### 📊 Live Asset Risk & Return Summary")
    st.markdown("Metrics calculated dynamically based on the trailing 1-year daily close prices.")

    display_data = (
        live_summary_df
        .sort_values('Sharpe_Ratio', ascending=False)
        .set_index('Sector')
        .copy()
    )
    display_data['Ann_Return'] *= 100
    display_data['Volatility'] *= 100

    st.dataframe(
        display_data,
        use_container_width=True,
        height=500,
        column_config={
            "Ann_Return":   st.column_config.NumberColumn("Ann. Return",  format="%.2f%%"),
            "Volatility":   st.column_config.NumberColumn("Volatility",   format="%.2f%%"),
            "Sharpe_Ratio": st.column_config.NumberColumn("Sharpe Ratio", format="%.2f"),
        },
    )
