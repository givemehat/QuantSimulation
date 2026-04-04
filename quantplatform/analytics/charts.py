"""
Analytics module: compute and visualize institutional-grade performance analytics.
Returns Plotly figures for embedding in the Streamlit dashboard.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

DARK_BG = "#0d1117"
CARD_BG = "#161b22"
ACCENT = "#00d4aa"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#ffd93d"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#e6edf3"
FONT = "IBM Plex Mono"


def _base_layout(title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=14, family=FONT)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR, family=FONT, size=11),
        xaxis=dict(gridcolor=GRID_COLOR, showline=False, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, showline=False, zeroline=False),
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )


def plot_equity_curve(
    equity_df: pd.DataFrame,
    benchmark_df: pd.DataFrame = None,
    initial_capital: float = 100_000.0,
) -> go.Figure:
    """Interactive equity curve with optional benchmark comparison."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    # Normalize to starting value
    equity_norm = equity_df["equity"] / equity_df["equity"].iloc[0] * 100

    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_norm,
        name="Strategy", line=dict(color=ACCENT, width=2),
        fill="tozeroy", fillcolor=f"rgba(0,212,170,0.07)",
    ), row=1, col=1)

    if benchmark_df is not None and not benchmark_df.empty:
        bench_norm = benchmark_df["Close"] / benchmark_df["Close"].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=benchmark_df.index, y=bench_norm,
            name="Benchmark", line=dict(color="#888", width=1.5, dash="dot"),
        ), row=1, col=1)

    # Drawdown
    drawdown_pct = equity_df["drawdown"] * 100
    fig.add_trace(go.Scatter(
        x=equity_df.index, y=drawdown_pct,
        name="Drawdown", line=dict(color=ACCENT2, width=1.5),
        fill="tozeroy", fillcolor="rgba(255,107,107,0.12)",
    ), row=2, col=1)

    layout = _base_layout("")
    layout["yaxis"] = dict(title="NAV (indexed to 100)", gridcolor=GRID_COLOR,
                           showline=False, zeroline=False, tickfont=dict(size=10))
    layout["yaxis2"] = dict(title="Drawdown %", gridcolor=GRID_COLOR,
                            showline=False, zeroline=False, tickfont=dict(size=10))
    layout["height"] = 480
    layout["showlegend"] = True
    fig.update_layout(**layout)
    return fig


def plot_price_with_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    ticker: str = "ASSET",
) -> go.Figure:
    """Candlestick chart with buy/sell signal markers and SMA overlays."""
    fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25],
                        shared_xaxes=True, vertical_spacing=0.04)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name=ticker,
        increasing_line_color=ACCENT,
        decreasing_line_color=ACCENT2,
        increasing_fillcolor=f"rgba(0,212,170,0.4)",
        decreasing_fillcolor=f"rgba(255,107,107,0.4)",
    ), row=1, col=1)

    # SMA overlays
    for w, col in [(20, "#ffd93d"), (50, "#a78bfa")]:
        sma = df["Close"].rolling(w).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=sma, name=f"SMA{w}",
            line=dict(color=col, width=1, dash="dot"),
        ), row=1, col=1)

    # Buy signals
    buy_dates = signals[signals == 1].index
    buy_prices = df.loc[buy_dates, "Low"] * 0.99 if len(buy_dates) > 0 else []
    if len(buy_dates) > 0:
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_prices,
            mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", color=ACCENT, size=9),
        ), row=1, col=1)

    # Sell signals
    sell_dates = signals[signals == -1].index
    sell_prices = df.loc[sell_dates, "High"] * 1.01 if len(sell_dates) > 0 else []
    if len(sell_dates) > 0:
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_prices,
            mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", color=ACCENT2, size=9),
        ), row=1, col=1)

    # Volume
    colors = [ACCENT if c >= o else ACCENT2
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, opacity=0.6,
    ), row=2, col=1)

    layout = _base_layout(f"{ticker} — Price & Signals")
    layout["height"] = 520
    layout["xaxis2"] = dict(gridcolor=GRID_COLOR)
    layout["yaxis2"] = dict(title="Volume", gridcolor=GRID_COLOR)
    layout["xaxis_rangeslider_visible"] = False
    fig.update_layout(**layout)
    return fig


def plot_returns_distribution(equity_df: pd.DataFrame) -> go.Figure:
    """Histogram of daily returns with normal distribution overlay."""
    returns = equity_df["equity"].pct_change().dropna() * 100

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns, nbinsx=60, name="Daily Returns",
        marker_color=ACCENT, opacity=0.75,
        histnorm="probability density",
    ))

    # Normal distribution overlay
    mu, sigma = returns.mean(), returns.std()
    x_range = np.linspace(returns.min(), returns.max(), 200)
    normal_curve = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_curve, name="Normal Dist.",
        line=dict(color=ACCENT3, width=2),
    ))

    layout = _base_layout("Return Distribution")
    layout["xaxis"]["title"] = "Daily Return (%)"
    layout["yaxis"]["title"] = "Density"
    fig.update_layout(**layout)
    return fig


def plot_monthly_returns_heatmap(equity_df: pd.DataFrame) -> go.Figure:
    """Monthly returns heatmap."""
    if equity_df.empty or len(equity_df) < 20:
        return go.Figure()

    daily_ret = equity_df["equity"].pct_change().dropna()
    monthly = daily_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    df_m = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values * 100,
    })

    pivot = df_m.pivot_table(values="Return", index="Year", columns="Month", aggfunc="sum")
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    cols = [month_labels[m-1] for m in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=cols,
        y=pivot.index.astype(str),
        colorscale=[[0, ACCENT2], [0.5, "#1a1f2e"], [1, ACCENT]],
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=9),
        showscale=True,
        colorbar=dict(ticksuffix="%"),
    ))

    layout = _base_layout("Monthly Returns Heatmap")
    layout["height"] = 300
    fig.update_layout(**layout)
    return fig


def plot_feature_importance(feature_importances: pd.Series, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of ML feature importances."""
    if feature_importances is None or len(feature_importances) == 0:
        return go.Figure()

    top = feature_importances.head(top_n).sort_values()

    fig = go.Figure(go.Bar(
        x=top.values, y=top.index,
        orientation="h",
        marker=dict(
            color=top.values,
            colorscale=[[0, "#1a1f2e"], [1, ACCENT]],
            showscale=False,
        ),
    ))

    layout = _base_layout(f"Top {top_n} Feature Importances")
    layout["height"] = 420
    layout["xaxis"]["title"] = "Importance"
    layout["margin"]["l"] = 150
    fig.update_layout(**layout)
    return fig


def plot_rolling_sharpe(equity_df: pd.DataFrame, window: int = 60) -> go.Figure:
    """Rolling Sharpe ratio over time."""
    ret = equity_df["equity"].pct_change().dropna()
    rolling_sharpe = (ret.rolling(window).mean() /
                      (ret.rolling(window).std() + 1e-9)) * np.sqrt(252)

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1)
    fig.add_hline(y=1, line_dash="dot", line_color=ACCENT, line_width=1, opacity=0.4)
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index, y=rolling_sharpe.values,
        name=f"{window}d Rolling Sharpe",
        line=dict(color=ACCENT, width=2),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
    ))

    layout = _base_layout(f"{window}-Day Rolling Sharpe Ratio")
    layout["yaxis"]["title"] = "Sharpe Ratio"
    fig.update_layout(**layout)
    return fig


def plot_ml_probability(pred_df: pd.DataFrame) -> go.Figure:
    """ML model long probability over time."""
    if pred_df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_hline(y=0.5, line_dash="dot", line_color="#555", line_width=1)
    fig.add_trace(go.Scatter(
        x=pred_df.index, y=pred_df["long_prob"],
        name="Long Probability", line=dict(color=ACCENT, width=1.5),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=pred_df.index, y=pred_df["short_prob"],
        name="Short Probability", line=dict(color=ACCENT2, width=1.5),
    ))

    layout = _base_layout("ML Model Prediction Probabilities")
    layout["yaxis"]["title"] = "Probability"
    layout["yaxis"]["range"] = [0, 1]
    fig.update_layout(**layout)
    return fig
