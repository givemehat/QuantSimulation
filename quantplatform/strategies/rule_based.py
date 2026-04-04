"""
Rule-based strategy implementations.
Each strategy exposes a generate_signals(df) method returning a Series of
{1 = long, -1 = short, 0 = flat} signals.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BaseStrategy:
    name: str = "base"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class MovingAverageCrossover(BaseStrategy):
    """
    Go long when fast SMA > slow SMA.
    Exit (flat) when fast SMA < slow SMA.
    """
    name = "MA Crossover"

    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()
        raw = pd.Series(0, index=df.index)
        raw[fast_ma > slow_ma] = 1
        raw[fast_ma < slow_ma] = -1
        # Shift by 1: trade on next open after signal fires
        signals = raw.shift(1).fillna(0)
        logger.info(f"[{self.name}] Generated {(signals != 0).sum()} non-zero signals")
        return signals


class MomentumStrategy(BaseStrategy):
    """
    Buy top-performing assets (positive momentum over lookback).
    Sell assets with negative momentum.
    Optional volatility regime filter.
    """
    name = "Momentum"

    def __init__(self, lookback: int = 20, vol_filter: bool = True, vol_window: int = 60):
        self.lookback = lookback
        self.vol_filter = vol_filter
        self.vol_window = vol_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        momentum = close.pct_change(self.lookback)
        signals = pd.Series(0, index=df.index)
        signals[momentum > 0.0] = 1
        signals[momentum < 0.0] = -1

        if self.vol_filter:
            # Only trade in low-volatility regime
            realized_vol = close.pct_change().rolling(self.vol_window).std() * np.sqrt(252)
            long_term_vol = realized_vol.rolling(self.vol_window * 2).mean()
            high_vol_regime = realized_vol > long_term_vol * 1.3
            signals[high_vol_regime] = 0

        signals = signals.shift(1).fillna(0)
        logger.info(f"[{self.name}] Generated {(signals != 0).sum()} non-zero signals")
        return signals


class MeanReversionStrategy(BaseStrategy):
    """
    Fade extreme moves: buy oversold, sell overbought.
    Uses Bollinger Bands + RSI confirmation.
    """
    name = "Mean Reversion"

    def __init__(self, bb_window: int = 20, bb_std: float = 2.0,
                 rsi_period: int = 14, rsi_oversold: float = 35,
                 rsi_overbought: float = 65):
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]

        bb_mid = close.rolling(self.bb_window).mean()
        bb_std = close.rolling(self.bb_window).std()
        bb_upper = bb_mid + self.bb_std * bb_std
        bb_lower = bb_mid - self.bb_std * bb_std

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rsi = 100 - 100 / (1 + gain / (loss + 1e-9))

        signals = pd.Series(0, index=df.index)
        # Buy: price below lower band AND RSI oversold
        buy_cond = (close < bb_lower) & (rsi < self.rsi_oversold)
        # Sell: price above upper band AND RSI overbought
        sell_cond = (close > bb_upper) & (rsi > self.rsi_overbought)

        signals[buy_cond] = 1
        signals[sell_cond] = -1
        signals = signals.shift(1).fillna(0)
        logger.info(f"[{self.name}] Generated {(signals != 0).sum()} non-zero signals")
        return signals


class BreakoutStrategy(BaseStrategy):
    """
    Buy on N-day high breakout; short on N-day low breakdown.
    """
    name = "Breakout"

    def __init__(self, window: int = 20, atr_multiplier: float = 1.5):
        self.window = window
        self.atr_multiplier = atr_multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        rolling_high = high.rolling(self.window).max()
        rolling_low = low.rolling(self.window).min()

        # ATR filter for noise reduction
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        signals = pd.Series(0, index=df.index)
        # Breakout: current close > previous window high + ATR buffer
        signals[close > rolling_high.shift(1) + self.atr_multiplier * atr] = 1
        signals[close < rolling_low.shift(1) - self.atr_multiplier * atr] = -1
        signals = signals.shift(1).fillna(0)
        logger.info(f"[{self.name}] Generated {(signals != 0).sum()} non-zero signals")
        return signals


# Strategy registry for UI selection
STRATEGY_REGISTRY = {
    "MA Crossover": MovingAverageCrossover,
    "Momentum": MomentumStrategy,
    "Mean Reversion": MeanReversionStrategy,
    "Breakout": BreakoutStrategy,
}
