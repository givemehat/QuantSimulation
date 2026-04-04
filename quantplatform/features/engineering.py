"""
Feature engineering pipeline.
All features are shifted by 1 bar to prevent look-ahead bias.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Compute full feature set from raw OHLCV.
    Returns DataFrame with features + target, NaN rows dropped.

    ANTI-LEAKAGE: All signal features are shifted by +1 so they
    are only available AFTER the bar closes. Target is forward return.
    """
    feat = df.copy()
    close = feat["Close"]
    high = feat["High"]
    low = feat["Low"]
    volume = feat["Volume"]

    # ── Returns ──────────────────────────────────────────────
    feat["ret_1d"] = close.pct_change(1)
    feat["ret_5d"] = close.pct_change(5)
    feat["log_ret_1d"] = np.log(close / close.shift(1))

    # ── Moving Averages ───────────────────────────────────────
    for w in [5, 10, 20, 50, 200]:
        feat[f"sma_{w}"] = close.rolling(w).mean()
        feat[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()

    feat["sma_cross_5_20"] = feat["sma_5"] - feat["sma_20"]
    feat["sma_cross_20_50"] = feat["sma_20"] - feat["sma_50"]

    # ── MACD ──────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    feat["macd"] = ema12 - ema26
    feat["macd_signal"] = feat["macd"].ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = feat["macd"] - feat["macd_signal"]

    # ── RSI ───────────────────────────────────────────────────
    feat["rsi_14"] = _rsi(close, 14)
    feat["rsi_28"] = _rsi(close, 28)

    # ── Bollinger Bands ───────────────────────────────────────
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat["bb_upper"] = bb_mid + 2 * bb_std
    feat["bb_lower"] = bb_mid - 2 * bb_std
    feat["bb_width"] = (feat["bb_upper"] - feat["bb_lower"]) / bb_mid
    feat["bb_pct"] = (close - feat["bb_lower"]) / (feat["bb_upper"] - feat["bb_lower"] + 1e-9)

    # ── Volatility ────────────────────────────────────────────
    for w in [5, 10, 20, 60]:
        feat[f"vol_{w}d"] = feat["log_ret_1d"].rolling(w).std() * np.sqrt(252)

    # ── Momentum ──────────────────────────────────────────────
    for w in [5, 10, 20, 60]:
        feat[f"mom_{w}d"] = close.pct_change(w)

    # ── Rolling z-score ───────────────────────────────────────
    feat["zscore_20"] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-9)

    # ── Volume features ───────────────────────────────────────
    feat["vol_ma_20"] = volume.rolling(20).mean()
    feat["vol_ratio"] = volume / (feat["vol_ma_20"] + 1e-9)
    feat["log_volume"] = np.log(volume + 1)

    # ── ATR (Average True Range) ──────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean()
    feat["atr_pct"] = feat["atr_14"] / (close + 1e-9)

    # ── Price position in recent range ───────────────────────
    feat["high_20d"] = high.rolling(20).max()
    feat["low_20d"] = low.rolling(20).min()
    feat["range_pct"] = (close - feat["low_20d"]) / (feat["high_20d"] - feat["low_20d"] + 1e-9)

    # ── Lagged features ───────────────────────────────────────
    for lag in [1, 2, 3, 5]:
        feat[f"ret_lag_{lag}"] = feat["ret_1d"].shift(lag)

    # ── TARGET: forward return classification ─────────────────
    # 1 if price is higher in `horizon` days, 0 otherwise
    fwd_return = close.shift(-horizon) / close - 1
    feat["target"] = (fwd_return > 0).astype(int)
    feat["fwd_return"] = fwd_return  # continuous target for analysis

    # ── Shift all signal features by 1 bar (anti-leakage) ────
    signal_cols = [c for c in feat.columns if c not in
                   ["Open", "High", "Low", "Close", "Volume", "target", "fwd_return"]]
    feat[signal_cols] = feat[signal_cols].shift(1)

    # Drop NaN rows from rolling windows and horizon
    feat = feat.dropna()
    logger.info(f"Feature matrix: {len(feat)} rows × {len(feat.columns)} cols")
    return feat


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI without look-ahead."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of usable ML feature column names."""
    exclude = {"Open", "High", "Low", "Close", "Volume", "target", "fwd_return"}
    return [c for c in df.columns if c not in exclude]
