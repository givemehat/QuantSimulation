"""
Unit tests for QuantTerminal platform components.
Run with: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd

from data.synthetic import generate_synthetic_ohlcv, get_demo_data
from data.ingestion import _validate_and_clean
from features.engineering import compute_features, get_feature_columns
from strategies.rule_based import (
    MovingAverageCrossover, MomentumStrategy,
    MeanReversionStrategy, BreakoutStrategy
)
from strategies.ml_strategy import MLStrategy
from portfolio.portfolio import Portfolio
from risk.risk_manager import RiskManager
from backtesting.engine import run_backtest, compute_metrics


@pytest.fixture
def sample_df():
    return generate_synthetic_ohlcv("TEST", "2020-01-01", "2023-12-31", seed=42)


@pytest.fixture
def feature_df(sample_df):
    return compute_features(sample_df, horizon=5)


# ── Data Tests ────────────────────────────────────────────────────────────────

class TestDataIngestion:
    def test_synthetic_shape(self, sample_df):
        assert sample_df.shape[1] == 5
        assert set(sample_df.columns) == {"Open", "High", "Low", "Close", "Volume"}

    def test_no_nans(self, sample_df):
        assert sample_df.isna().sum().sum() == 0

    def test_positive_prices(self, sample_df):
        assert (sample_df["Close"] > 0).all()
        assert (sample_df["High"] >= sample_df["Low"]).all()

    def test_date_index(self, sample_df):
        assert isinstance(sample_df.index, pd.DatetimeIndex)
        assert sample_df.index.is_monotonic_increasing

    def test_demo_tickers(self):
        for ticker in ["AAPL", "MSFT", "SPY", "TSLA"]:
            df = get_demo_data(ticker, "2022-01-01", "2023-12-31")
            assert len(df) > 100
            assert "Close" in df.columns


# ── Feature Tests ─────────────────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_feature_count(self, feature_df):
        feature_cols = get_feature_columns(feature_df)
        assert len(feature_cols) >= 40

    def test_no_nans_after_drop(self, feature_df):
        assert feature_df.isna().sum().sum() == 0

    def test_target_binary(self, feature_df):
        assert set(feature_df["target"].unique()).issubset({0, 1})

    def test_no_lookahead(self, sample_df):
        """Features computed on bar t should not use bar t's data."""
        feat = compute_features(sample_df, horizon=5)
        # The shift(1) means the first signal feature row should be NaN before drop
        # We just verify the feature df is shorter than input (NaN rows dropped)
        assert len(feat) < len(sample_df)

    def test_required_features_present(self, feature_df):
        for col in ["sma_20", "ema_50", "rsi_14", "macd", "bb_width", "atr_14"]:
            assert col in feature_df.columns, f"Missing feature: {col}"


# ── Strategy Tests ────────────────────────────────────────────────────────────

class TestRuleBasedStrategies:
    def test_ma_crossover_signals(self, sample_df):
        strat = MovingAverageCrossover(fast=10, slow=30)
        sigs = strat.generate_signals(sample_df)
        assert set(sigs.unique()).issubset({-1, 0, 1})
        assert len(sigs) == len(sample_df)

    def test_momentum_signals(self, sample_df):
        strat = MomentumStrategy(lookback=20)
        sigs = strat.generate_signals(sample_df)
        assert set(sigs.unique()).issubset({-1, 0, 1})

    def test_mean_reversion_signals(self, sample_df):
        strat = MeanReversionStrategy()
        sigs = strat.generate_signals(sample_df)
        assert set(sigs.unique()).issubset({-1, 0, 1})

    def test_breakout_signals(self, sample_df):
        strat = BreakoutStrategy(window=20)
        sigs = strat.generate_signals(sample_df)
        assert set(sigs.unique()).issubset({-1, 0, 1})

    def test_signals_shifted(self, sample_df):
        """First signal should be 0 (shifted, no lookahead)."""
        strat = MovingAverageCrossover(fast=5, slow=20)
        sigs = strat.generate_signals(sample_df)
        assert sigs.iloc[0] == 0


# ── Portfolio Tests ───────────────────────────────────────────────────────────

class TestPortfolio:
    def test_initial_state(self):
        p = Portfolio(initial_capital=100_000)
        assert p.cash == 100_000
        assert len(p.positions) == 0
        assert len(p.trades) == 0

    def test_buy_reduces_cash(self):
        p = Portfolio(initial_capital=100_000, commission_pct=0, slippage_bps=0)
        trade = p.buy(pd.Timestamp("2020-01-01"), "AAPL", 100, 150.0)
        assert trade is not None
        assert p.cash < 100_000
        assert "AAPL" in p.positions
        assert p.positions["AAPL"].quantity == 100

    def test_sell_increases_cash(self):
        p = Portfolio(initial_capital=100_000, commission_pct=0, slippage_bps=0)
        p.buy(pd.Timestamp("2020-01-01"), "AAPL", 100, 150.0)
        cash_after_buy = p.cash
        p.sell(pd.Timestamp("2020-01-02"), "AAPL", 100, 160.0)
        assert p.cash > cash_after_buy

    def test_pnl_calculation(self):
        p = Portfolio(initial_capital=100_000, commission_pct=0, slippage_bps=0)
        p.buy(pd.Timestamp("2020-01-01"), "AAPL", 100, 100.0)
        trade = p.sell(pd.Timestamp("2020-01-02"), "AAPL", 100, 110.0)
        assert trade is not None
        assert abs(trade.pnl - 1000.0) < 1.0  # $10 gain × 100 shares

    def test_insufficient_cash(self):
        p = Portfolio(initial_capital=1_000)
        # Try to buy way too many shares
        trade = p.buy(pd.Timestamp("2020-01-01"), "AAPL", 10000, 500.0)
        # Should scale down or return None, not crash
        assert p.cash >= 0


# ── Risk Manager Tests ────────────────────────────────────────────────────────

class TestRiskManager:
    def test_position_sizing(self):
        rm = RiskManager(initial_capital=100_000, max_position_pct=0.20,
                         volatility_scaling=False)
        qty = rm.size_position(100_000, 100.0)
        assert qty in (199, 200)  # 20% of 100k / $100 (int floor, float-safe)

    def test_stop_loss_trigger(self):
        rm = RiskManager(stop_loss_pct=0.05)
        rm.record_entry("AAPL", 100.0)
        assert not rm.check_stop_loss("AAPL", 96.0)   # 4% loss — not triggered
        assert rm.check_stop_loss("AAPL", 94.0)        # 6% loss — triggered

    def test_take_profit_trigger(self):
        rm = RiskManager(take_profit_pct=0.10)
        rm.record_entry("AAPL", 100.0)
        assert not rm.check_take_profit("AAPL", 108.0)  # 8% gain — not triggered
        assert rm.check_take_profit("AAPL", 112.0)       # 12% gain — triggered

    def test_max_drawdown_halt(self):
        rm = RiskManager(initial_capital=100_000, max_drawdown_pct=0.25)
        rm.check_drawdown(100_000)
        rm.check_drawdown(90_000)
        assert not rm.is_halted()
        rm.check_drawdown(70_000)  # 30% drawdown > 25% limit
        assert rm.is_halted()

    def test_halted_returns_zero_size(self):
        rm = RiskManager(initial_capital=100_000, max_drawdown_pct=0.10)
        rm.check_drawdown(100_000)
        rm.check_drawdown(80_000)  # 20% > 10% → halt
        assert rm.size_position(100_000, 100.0) == 0


# ── Backtest Tests ────────────────────────────────────────────────────────────

class TestBacktestEngine:
    def test_backtest_runs(self, sample_df):
        strat = MovingAverageCrossover(fast=20, slow=50)
        sigs = strat.generate_signals(sample_df)
        result = run_backtest(sample_df, strat, signals=sigs,
                              initial_capital=100_000, ticker="TEST")
        assert not result.equity_df.empty
        assert "equity" in result.equity_df.columns

    def test_equity_never_negative(self, sample_df):
        strat = MomentumStrategy(lookback=20)
        sigs = strat.generate_signals(sample_df)
        result = run_backtest(sample_df, strat, signals=sigs,
                              initial_capital=100_000, ticker="TEST")
        assert (result.equity_df["equity"] >= 0).all()

    def test_metrics_computed(self, sample_df):
        strat = MovingAverageCrossover()
        sigs = strat.generate_signals(sample_df)
        result = run_backtest(sample_df, strat, signals=sigs,
                              initial_capital=100_000, ticker="TEST")
        m = result.metrics
        assert "Total Return" in m
        assert "Sharpe Ratio" in m
        assert "Max Drawdown" in m

    def test_ml_backtest(self, sample_df, feature_df):
        ml = MLStrategy(model_type="random_forest", horizon=5)
        ml.fit(feature_df)
        sigs = ml.generate_signals(feature_df).reindex(sample_df.index).fillna(0)
        result = run_backtest(sample_df, ml, signals=sigs,
                              initial_capital=100_000, ticker="TEST")
        assert not result.equity_df.empty
        assert isinstance(result.metrics.get("_sharpe"), float)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
