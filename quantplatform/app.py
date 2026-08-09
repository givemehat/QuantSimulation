"""
QuantTerminal — AI-Driven Quantitative Trading Platform
Production-grade Streamlit dashboard.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Platform imports ──────────────────────────────────────────────────────────
from data.ingestion import fetch_ohlcv, DataValidationError
from features.engineering import compute_features
from strategies.rule_based import STRATEGY_REGISTRY, MovingAverageCrossover
from strategies.ml_strategy import MLStrategy
from backtesting.engine import run_backtest
from analytics.charts import (
    plot_equity_curve,
    plot_price_with_signals,
    plot_returns_distribution,
    plot_monthly_returns_heatmap,
    plot_feature_importance,
    plot_rolling_sharpe,
    plot_ml_probability,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantTerminal",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #21262d;
    --accent: #00d4aa;
    --accent2: #ff6b6b;
    --accent3: #ffd93d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --border: #30363d;
    --font-mono: 'IBM Plex Mono', monospace;
    --font-sans: 'IBM Plex Sans', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-sans) !important;
}

.stApp { background-color: var(--bg) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Header brand */
.brand-header {
    font-family: var(--font-mono);
    font-size: 22px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.08em;
    padding: 8px 0 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 18px;
}
.brand-sub {
    font-size: 10px;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: -14px;
    margin-bottom: 20px;
}

/* KPI cards */
.kpi-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    flex: 1;
    min-width: 130px;
    position: relative;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 8px 8px 0 0;
    background: var(--accent);
}
.kpi-card.negative::before { background: var(--accent2); }
.kpi-card.neutral::before { background: var(--accent3); }
.kpi-label {
    font-family: var(--font-mono);
    font-size: 9px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: var(--font-mono);
    font-size: 22px;
    font-weight: 600;
    color: var(--text);
    line-height: 1;
}
.kpi-value.green { color: var(--accent); }
.kpi-value.red { color: var(--accent2); }
.kpi-value.yellow { color: var(--accent3); }

/* Section headers */
.section-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin: 20px 0 12px;
}

/* Status badge */
.status-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 10px;
    padding: 3px 10px;
    border-radius: 20px;
    background: rgba(0,212,170,0.12);
    color: var(--accent);
    border: 1px solid rgba(0,212,170,0.3);
    letter-spacing: 0.1em;
    margin-left: 8px;
}
.status-badge.red {
    background: rgba(255,107,107,0.12);
    color: var(--accent2);
    border-color: rgba(255,107,107,0.3);
}

/* Metric table */
.metric-table {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    font-family: var(--font-mono);
    font-size: 12px;
}
.metric-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid var(--surface2);
}
.metric-row:last-child { border-bottom: none; }
.metric-key { color: var(--text-muted); }
.metric-val { color: var(--text); font-weight: 500; }

/* Streamlit widget overrides */
.stSelectbox > div, .stSlider, .stCheckbox { color: var(--text) !important; }
div[data-baseweb="select"] { background: var(--surface2) !important; border-color: var(--border) !important; }
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Tabs */
div[data-baseweb="tab-list"] { border-bottom: 1px solid var(--border) !important; background: transparent !important; }
button[data-baseweb="tab"] {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.1em !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Expander */
details { border: 1px solid var(--border) !important; border-radius: 6px !important; }
summary { font-family: var(--font-mono) !important; font-size: 11px !important; }

/* Dataframe */
.dataframe { font-family: var(--font-mono) !important; font-size: 11px !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO MODE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _check_live_data() -> bool:
    try:
        import yfinance as yf

        t = yf.download("SPY", period="5d", progress=False)
        return not t.empty
    except Exception:
        return False


LIVE_DATA = _check_live_data()
if not LIVE_DATA:
    st.markdown(
        """
    <div style="background:rgba(255,211,61,0.08);border:1px solid rgba(255,211,61,0.3);
    border-radius:8px;padding:10px 16px;margin-bottom:12px;
    font-family:'IBM Plex Mono';font-size:11px;color:#ffd93d;letter-spacing:0.04em;">
    ⚡ DEMO MODE — Synthetic GBM price data active.
    All strategies, risk controls and analytics fully functional.
    </div>""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="brand-header">⬡ QUANT TERMINAL</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="brand-sub">Strategy Research Platform</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">📡 Data Configuration</div>', unsafe_allow_html=True
    )
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper().strip()
    benchmark_ticker = st.text_input("Benchmark", value="SPY").upper().strip()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=pd.to_datetime("2020-01-01"))
    with col2:
        end_date = st.date_input("End", value=pd.to_datetime("2024-12-31"))

    st.markdown(
        '<div class="section-label">⚡ Strategy Engine</div>', unsafe_allow_html=True
    )
    strategy_mode = st.selectbox(
        "Strategy Type",
        ["Rule-Based", "ML Strategy"],
    )

    if strategy_mode == "Rule-Based":
        strategy_name = st.selectbox("Strategy", list(STRATEGY_REGISTRY.keys()))

        with st.expander("Strategy Parameters"):
            if strategy_name == "MA Crossover":
                fast_ma = st.slider("Fast MA Period", 5, 50, 20)
                slow_ma = st.slider("Slow MA Period", 20, 200, 50)
            elif strategy_name == "Momentum":
                lookback = st.slider("Lookback Period", 5, 60, 20)
                vol_filter = st.checkbox("Volatility Regime Filter", value=True)
            elif strategy_name == "Mean Reversion":
                bb_window = st.slider("BB Window", 10, 50, 20)
                rsi_os = st.slider("RSI Oversold", 20, 45, 35)
                rsi_ob = st.slider("RSI Overbought", 55, 80, 65)
            elif strategy_name == "Breakout":
                break_window = st.slider("Breakout Window", 10, 60, 20)
    else:
        ml_model = st.selectbox("Model", ["random_forest", "xgboost", "logistic"])
        ml_threshold = st.slider("Signal Threshold", 0.50, 0.80, 0.55, step=0.01)
        ml_horizon = st.slider("Prediction Horizon (days)", 1, 20, 5)

    st.markdown(
        '<div class="section-label">🛡 Risk Controls</div>', unsafe_allow_html=True
    )
    initial_capital = st.number_input(
        "Initial Capital ($)", 10_000, 10_000_000, 100_000, step=10_000
    )
    max_pos_pct = st.slider("Max Position Size (%)", 5, 50, 20) / 100
    stop_loss = st.slider("Stop Loss (%)", 1, 20, 5) / 100
    take_profit = st.slider("Take Profit (%)", 5, 50, 15) / 100
    max_dd = st.slider("Max Drawdown Limit (%)", 10, 50, 25) / 100
    vol_scaling = st.checkbox("Volatility Position Scaling", value=True)

    st.markdown("---")
    run_btn = st.button("▶  RUN BACKTEST", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div style="display:flex; align-items:center; margin-bottom:4px;">
    <span style="font-family:'IBM Plex Mono';font-size:26px;font-weight:600;color:#e6edf3;">{ticker}</span>
    <span class="status-badge">LIVE SIM</span>
</div>
<div style="font-family:'IBM Plex Mono';font-size:11px;color:#8b949e;margin-bottom:20px;">
    {strategy_mode} &nbsp;·&nbsp; {start_date} → {end_date} &nbsp;·&nbsp; Capital: ${initial_capital:,}
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "df" not in st.session_state:
    st.session_state.df = None
if "benchmark_df" not in st.session_state:
    st.session_state.benchmark_df = None
if "ml_model_obj" not in st.session_state:
    st.session_state.ml_model_obj = None
if "feature_df" not in st.session_state:
    st.session_state.feature_df = None

# ─────────────────────────────────────────────────────────────────────────────
# RUN BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Fetching market data..."):
        try:
            df = fetch_ohlcv(ticker, str(start_date), str(end_date))
            st.session_state.df = df
        except DataValidationError as e:
            st.error(f"❌ Data error: {e}")
            st.stop()

        try:
            bench_df = fetch_ohlcv(benchmark_ticker, str(start_date), str(end_date))
            st.session_state.benchmark_df = bench_df
        except Exception:
            st.session_state.benchmark_df = None

    with st.spinner("Building strategy & running backtest..."):
        try:
            df = st.session_state.df

            if strategy_mode == "Rule-Based":
                if strategy_name == "MA Crossover":
                    strategy = MovingAverageCrossover(fast=fast_ma, slow=slow_ma)
                elif strategy_name == "Momentum":
                    from strategies.rule_based import MomentumStrategy

                    strategy = MomentumStrategy(
                        lookback=lookback, vol_filter=vol_filter
                    )
                elif strategy_name == "Mean Reversion":
                    from strategies.rule_based import MeanReversionStrategy

                    strategy = MeanReversionStrategy(
                        bb_window=bb_window, rsi_oversold=rsi_os, rsi_overbought=rsi_ob
                    )
                elif strategy_name == "Breakout":
                    from strategies.rule_based import BreakoutStrategy

                    strategy = BreakoutStrategy(window=break_window)

                signals = strategy.generate_signals(df)
                ml_obj = None
                feature_df = None

            else:
                feature_df = compute_features(df, horizon=ml_horizon)
                ml_obj = MLStrategy(
                    model_type=ml_model,
                    horizon=ml_horizon,
                    threshold=ml_threshold,
                )
                metrics = ml_obj.fit(feature_df)
                strategy = ml_obj
                signals = ml_obj.generate_signals(feature_df)
                # Align signals to df index
                signals = signals.reindex(df.index).fillna(0)
                st.session_state.ml_model_obj = ml_obj
                st.session_state.feature_df = feature_df

            result = run_backtest(
                df=df,
                strategy=strategy,
                signals=signals,
                initial_capital=initial_capital,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                max_drawdown_pct=max_dd,
                max_position_pct=max_pos_pct,
                volatility_scaling=vol_scaling,
                ticker=ticker,
            )
            st.session_state.result = result
            st.success("✅ Backtest complete")

        except Exception as e:
            st.error(f"❌ Backtest error: {e}")
            import traceback

            st.code(traceback.format_exc())
            st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────────────────────
result = st.session_state.result
df = st.session_state.df

if result is not None and df is not None:
    m = result.metrics

    # ── KPI Cards ──────────────────────────────────────────────────────────
    def kpi_card(label, value, cls=""):
        color_cls = ""
        if "%" in str(value):
            try:
                v = float(str(value).replace("%", "").replace(",", ""))
                if v > 0:
                    color_cls = "green"
                elif v < 0:
                    color_cls = "red"
            except:
                pass
        return f"""
        <div class="kpi-card {cls}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {color_cls}">{value}</div>
        </div>"""

    st.markdown(
        '<div class="section-label">📊 Performance Summary</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
    <div class="kpi-grid">
        {kpi_card("Total Return", m.get("Total Return", "—"))}
        {kpi_card("CAGR", m.get("CAGR", "—"))}
        {kpi_card("Sharpe Ratio", m.get("Sharpe Ratio", "—"), "neutral")}
        {kpi_card("Sortino Ratio", m.get("Sortino Ratio", "—"), "neutral")}
        {kpi_card("Max Drawdown", m.get("Max Drawdown", "—"), "negative")}
        {kpi_card("Win Rate", m.get("Win Rate", "—"))}
        {kpi_card("Profit Factor", m.get("Profit Factor", "—"), "neutral")}
        {kpi_card("Num Trades", m.get("Num Trades", "—"), "neutral")}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── TABS ───────────────────────────────────────────────────────────────
    tabs = st.tabs(
        [
            "📈 Equity Curve",
            "🕯 Price & Signals",
            "📉 Drawdown & Risk",
            "📅 Monthly Returns",
            "🔄 Trade Ledger",
            "🤖 ML Insights",
        ]
    )

    equity_df = result.equity_df
    signals = result.signals
    bench_df = st.session_state.benchmark_df

    with tabs[0]:
        st.markdown(
            '<div class="section-label">Equity Curve vs Benchmark</div>',
            unsafe_allow_html=True,
        )
        fig_eq = plot_equity_curve(equity_df, bench_df, initial_capital)
        st.plotly_chart(fig_eq, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '<div class="section-label">Return Distribution</div>',
                unsafe_allow_html=True,
            )
            fig_dist = plot_returns_distribution(equity_df)
            st.plotly_chart(fig_dist, use_container_width=True)
        with col2:
            st.markdown(
                '<div class="section-label">Rolling Sharpe Ratio</div>',
                unsafe_allow_html=True,
            )
            fig_sharpe = plot_rolling_sharpe(equity_df)
            st.plotly_chart(fig_sharpe, use_container_width=True)

    with tabs[1]:
        st.markdown(
            '<div class="section-label">Price Chart with Entry/Exit Signals</div>',
            unsafe_allow_html=True,
        )
        # Show last 252 bars for readability
        display_df = df.tail(252)
        display_sig = signals.reindex(display_df.index).fillna(0)
        fig_price = plot_price_with_signals(display_df, display_sig, ticker)
        st.plotly_chart(fig_price, use_container_width=True)

    with tabs[2]:
        st.markdown(
            '<div class="section-label">Drawdown Analysis</div>', unsafe_allow_html=True
        )
        dd = equity_df["drawdown"] * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=dd,
                fill="tozeroy",
                fillcolor="rgba(255,107,107,0.15)",
                line=dict(color="#ff6b6b", width=1.5),
                name="Drawdown %",
            )
        )
        fig_dd.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3", family="IBM Plex Mono"),
            yaxis=dict(title="Drawdown (%)", gridcolor="#21262d"),
            xaxis=dict(gridcolor="#21262d"),
            margin=dict(l=50, r=20, t=20, b=40),
            height=300,
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # Risk metrics detail
        st.markdown(
            '<div class="section-label">Full Risk Metrics</div>', unsafe_allow_html=True
        )
        display_metrics = {k: v for k, v in m.items() if not k.startswith("_")}
        metric_rows = "".join(
            [
                f'<div class="metric-row"><span class="metric-key">{k}</span><span class="metric-val">{v}</span></div>'
                for k, v in display_metrics.items()
            ]
        )
        st.markdown(
            f'<div class="metric-table">{metric_rows}</div>', unsafe_allow_html=True
        )

    with tabs[3]:
        st.markdown(
            '<div class="section-label">Monthly Returns Heatmap</div>',
            unsafe_allow_html=True,
        )
        fig_heat = plot_monthly_returns_heatmap(equity_df)
        st.plotly_chart(fig_heat, use_container_width=True)

    with tabs[4]:
        trades_df = result.trades_df
        st.markdown(
            '<div class="section-label">Trade Ledger</div>', unsafe_allow_html=True
        )
        if not trades_df.empty:
            st.markdown(
                f"**{len(trades_df)} total orders** | "
                f"**{len(trades_df[trades_df['Side']=='BUY'])} buys** | "
                f"**{len(trades_df[trades_df['Side']=='SELL'])} sells**"
            )
            sells = trades_df[trades_df["Side"] == "SELL"].copy()
            if not sells.empty:
                sells["PnL_Color"] = sells["PnL"].apply(
                    lambda x: "🟢" if x > 0 else "🔴"
                )
                sells["PnL"] = sells["PnL"].map("${:,.2f}".format)
                sells["Price"] = sells["Price"].map("${:,.2f}".format)
                sells["Commission"] = sells["Commission"].map("${:,.2f}".format)
                st.dataframe(
                    sells[
                        [
                            "Date",
                            "Ticker",
                            "Side",
                            "Quantity",
                            "Price",
                            "Commission",
                            "PnL",
                            "PnL_Color",
                        ]
                    ].rename(columns={"PnL_Color": ""}),
                    use_container_width=True,
                    height=400,
                )
        else:
            st.info("No trades executed in this backtest period.")

    with tabs[5]:
        ml_obj = st.session_state.ml_model_obj
        feature_df = st.session_state.feature_df

        if ml_obj is not None and feature_df is not None:
            st.markdown(
                '<div class="section-label">ML Model Performance</div>',
                unsafe_allow_html=True,
            )

            tm = ml_obj.train_metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Walk-Forward Accuracy",
                    f"{tm.get('walk_forward_acc', 0):.1%}",
                    f"±{tm.get('walk_forward_std', 0):.1%}",
                )
            with col2:
                st.metric("Out-of-Sample Accuracy", f"{tm.get('oos_accuracy', 0):.1%}")
            with col3:
                st.metric("Training Samples", f"{tm.get('n_train', 0):,}")

            # Fold accuracy chart
            fold_accs = tm.get("fold_accuracies", [])
            if fold_accs:
                fig_folds = go.Figure(
                    go.Bar(
                        x=[f"Fold {i+1}" for i in range(len(fold_accs))],
                        y=[a * 100 for a in fold_accs],
                        marker_color="#00d4aa",
                        opacity=0.8,
                    )
                )
                fig_folds.add_hline(
                    y=50,
                    line_dash="dot",
                    line_color="#555",
                    annotation_text="Random Baseline",
                    annotation_font_color="#8b949e",
                )
                fig_folds.update_layout(
                    paper_bgcolor="#0d1117",
                    plot_bgcolor="#0d1117",
                    font=dict(color="#e6edf3", family="IBM Plex Mono"),
                    yaxis=dict(
                        title="Accuracy (%)", gridcolor="#21262d", range=[40, 80]
                    ),
                    xaxis=dict(gridcolor="#21262d"),
                    margin=dict(l=50, r=20, t=20, b=40),
                    height=250,
                    title=dict(
                        text="Walk-Forward Fold Accuracies",
                        font=dict(color="#e6edf3", size=12),
                    ),
                )
                st.plotly_chart(fig_folds, use_container_width=True)

            # Feature importance
            st.markdown(
                '<div class="section-label">Feature Importance</div>',
                unsafe_allow_html=True,
            )
            fig_fi = plot_feature_importance(ml_obj.feature_importances_)
            st.plotly_chart(fig_fi, use_container_width=True)

            # ML probability
            st.markdown(
                '<div class="section-label">Model Prediction Probabilities</div>',
                unsafe_allow_html=True,
            )
            pred_df = ml_obj.get_prediction_df(feature_df)
            fig_prob = plot_ml_probability(pred_df)
            st.plotly_chart(fig_prob, use_container_width=True)

        else:
            st.info("🤖 Run a **ML Strategy** backtest to see model insights here.")

else:
    # Welcome / empty state
    st.markdown(
        """
    <div style="
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 60px 40px;
        text-align: center;
        background: #161b22;
        margin-top: 30px;
    ">
        <div style="font-size: 48px; margin-bottom: 16px;">⬡</div>
        <div style="font-family:'IBM Plex Mono';font-size:20px;font-weight:600;color:#e6edf3;margin-bottom:8px;">
            QuantTerminal Ready
        </div>
        <div style="font-family:'IBM Plex Sans';font-size:14px;color:#8b949e;max-width:400px;margin:0 auto 24px;">
            Configure your strategy in the sidebar and click
            <strong style="color:#00d4aa;">▶ RUN BACKTEST</strong> to begin simulation.
        </div>
        <div style="display:flex;justify-content:center;gap:32px;flex-wrap:wrap;">
            <div style="font-family:'IBM Plex Mono';font-size:11px;color:#8b949e;">
                📊 Rule-Based Strategies<br>
                <span style="color:#00d4aa;">MA Crossover · Momentum</span><br>
                <span style="color:#00d4aa;">Mean Reversion · Breakout</span>
            </div>
            <div style="font-family:'IBM Plex Mono';font-size:11px;color:#8b949e;">
                🤖 ML Strategies<br>
                <span style="color:#00d4aa;">Random Forest · XGBoost</span><br>
                <span style="color:#00d4aa;">Logistic Regression</span>
            </div>
            <div style="font-family:'IBM Plex Mono';font-size:11px;color:#8b949e;">
                🛡 Risk Controls<br>
                <span style="color:#00d4aa;">Stop-Loss · Take-Profit</span><br>
                <span style="color:#00d4aa;">Drawdown Limits · Vol Scaling</span>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
