"""
Backtesting engine: event-driven simulation with realistic trading assumptions.
Integrates strategy signals → risk checks → order sizing → portfolio execution.
"""
import logging
from typing import Optional, Union
import numpy as np
import pandas as pd

from portfolio.portfolio import Portfolio
from risk.risk_manager import RiskManager
from strategies.rule_based import BaseStrategy
from strategies.ml_strategy import MLStrategy

logger = logging.getLogger(__name__)


class BacktestResult:
    def __init__(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame,
                 signals: pd.Series, metrics: dict):
        self.equity_df = equity_df
        self.trades_df = trades_df
        self.signals = signals
        self.metrics = metrics


def run_backtest(
    df: pd.DataFrame,
    strategy: Union[BaseStrategy, MLStrategy],
    signals: Optional[pd.Series] = None,
    initial_capital: float = 100_000.0,
    commission_pct: float = 0.001,
    slippage_bps: float = 5.0,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.15,
    max_drawdown_pct: float = 0.25,
    max_position_pct: float = 0.20,
    volatility_scaling: bool = True,
    ticker: str = "ASSET",
) -> BacktestResult:
    """
    Run a full vectorized + event-driven backtest.

    Parameters
    ----------
    df : OHLCV DataFrame
    strategy : strategy object (or pre-computed signals via `signals`)
    signals : Optional pre-computed signal Series (1/-1/0)
    """
    # Generate signals if not provided
    if signals is None:
        signals = strategy.generate_signals(df)

    # Align signals to df index
    signals = signals.reindex(df.index).fillna(0)

    # Compute realized volatility for position sizing
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    rolling_vol = log_ret.rolling(20).std() * np.sqrt(252)

    # Initialize portfolio and risk manager
    portfolio = Portfolio(
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        slippage_bps=slippage_bps,
    )
    risk = RiskManager(
        initial_capital=initial_capital,
        max_position_pct=max_position_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_drawdown_pct=max_drawdown_pct,
        volatility_scaling=volatility_scaling,
    )

    current_signal = 0  # track current position direction
    prev_date = None

    for date, row in df.iterrows():
        price = row["Close"]
        sig = signals.loc[date] if date in signals.index else 0
        vol = rolling_vol.loc[date] if date in rolling_vol.index else None

        # Update equity and check risk controls
        equity = portfolio.update_equity(date, {ticker: price})
        risk.check_drawdown(equity)

        if prev_date is None or date.date() != prev_date:
            risk.reset_daily(equity)
        prev_date = date.date()

        if risk.is_halted():
            # Close all positions on halt
            if portfolio.positions:
                portfolio.close_all(date, {ticker: price})
                risk.clear_entry(ticker)
                current_signal = 0
            continue

        # Check stop-loss / take-profit for existing position
        if current_signal != 0:
            if (risk.check_stop_loss(ticker, price) or
                    risk.check_take_profit(ticker, price)):
                portfolio.close_all(date, {ticker: price})
                risk.clear_entry(ticker)
                current_signal = 0
                continue

        # Process signal changes
        if sig != current_signal:
            # Close existing position first
            if current_signal != 0 and portfolio.positions:
                portfolio.close_all(date, {ticker: price})
                risk.clear_entry(ticker)

            # Open new position
            if sig == 1:  # Long
                if risk.check_daily_loss(equity, date):
                    qty = risk.size_position(equity, price, vol if vol and not np.isnan(vol) else None)
                    if qty > 0:
                        portfolio.buy(date, ticker, qty, price)
                        risk.record_entry(ticker, price)

            elif sig == -1:  # Short (implemented as staying flat for long-only)
                # For simplicity: short = sell existing and go flat
                # A proper short would require margin accounting
                pass

            current_signal = sig

    # Close all remaining positions at end
    if portfolio.positions:
        last_date = df.index[-1]
        last_price = df["Close"].iloc[-1]
        portfolio.close_all(last_date, {ticker: last_price})

    equity_df = portfolio.get_equity_df()
    trades_df = portfolio.get_trades_df()

    metrics = compute_metrics(equity_df, trades_df, initial_capital)

    logger.info(f"Backtest complete | Trades: {len(trades_df)} | Final equity: {equity_df['equity'].iloc[-1] if len(equity_df) > 0 else initial_capital:.2f}")
    return BacktestResult(equity_df, trades_df, signals, metrics)


def compute_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame,
                    initial_capital: float) -> dict:
    """Compute institutional-grade performance metrics."""
    if equity_df.empty:
        return {}

    equity = equity_df["equity"]
    returns = equity.pct_change().dropna()

    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    n_years = len(equity) / 252
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / max(n_years, 0.01)) - 1

    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
    downside = returns[returns < 0].std() + 1e-9
    sortino = (returns.mean() / downside) * np.sqrt(252)

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / (rolling_max + 1e-9)
    max_dd = drawdown.min()

    volatility = returns.std() * np.sqrt(252)

    # Trade-level stats
    if not trades_df.empty:
        sells = trades_df[trades_df["Side"] == "SELL"]
        winning = sells[sells["PnL"] > 0]
        losing = sells[sells["PnL"] <= 0]
        win_rate = len(winning) / max(len(sells), 1)
        gross_profit = winning["PnL"].sum()
        gross_loss = abs(losing["PnL"].sum())
        profit_factor = gross_profit / max(gross_loss, 1e-9)
        avg_trade_ret = sells["PnL"].mean() if len(sells) > 0 else 0
        n_trades = len(sells)
    else:
        win_rate = profit_factor = avg_trade_ret = 0
        n_trades = 0

    return {
        "Total Return": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Volatility (Ann.)": f"{volatility:.2%}",
        "Win Rate": f"{win_rate:.2%}",
        "Profit Factor": f"{profit_factor:.2f}",
        "Avg Trade PnL": f"${avg_trade_ret:,.2f}",
        "Num Trades": n_trades,
        "Final Equity": f"${equity.iloc[-1]:,.2f}",
        "Initial Capital": f"${initial_capital:,.2f}",
        # raw for charts
        "_total_return": total_return,
        "_cagr": cagr,
        "_sharpe": sharpe,
        "_sortino": sortino,
        "_max_dd": max_dd,
        "_volatility": volatility,
        "_win_rate": win_rate,
        "_profit_factor": profit_factor,
    }
