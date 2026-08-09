"""
Market data ingestion, validation, preprocessing, and caching.
Designed as a swappable data service abstraction.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    pass


def _cache_path(ticker: str, start: str, end: str, interval: str) -> Path:
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{ticker}_{start}_{end}_{interval}.parquet".replace(":", "-")
    return cache_dir / fname


def fetch_ohlcv(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance with optional parquet caching.
    Returns a clean, validated DataFrame indexed by date.
    """
    cache_file = _cache_path(ticker, start, end, interval)

    if use_cache and cache_file.exists():
        logger.info(f"Loading cached data for {ticker}")
        df = pd.read_parquet(cache_file)
        return df

    logger.info(f"Fetching {ticker} from {start} to {end} [{interval}]")
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        logger.warning(f"yfinance failed ({e}), falling back to synthetic data")
        from data.synthetic import get_demo_data

        return get_demo_data(ticker, start, end)

    if raw.empty:
        logger.warning(f"Empty response for {ticker}, falling back to synthetic data")
        from data.synthetic import get_demo_data

        return get_demo_data(ticker, start, end)

    df = _validate_and_clean(raw, ticker)

    if use_cache:
        df.to_parquet(cache_file)
        logger.info(f"Cached {ticker} data → {cache_file}")

    return df


def _validate_and_clean(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate schema, handle missing values, normalize columns."""
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing columns {missing} for {ticker}")

    # Remove rows where Close is NaN
    before = len(df)
    df = df.dropna(subset=["Close"])
    after = len(df)
    if before != after:
        logger.warning(f"Dropped {before - after} rows with NaN Close for {ticker}")

    # Fill remaining NaN with forward fill then back fill
    df = df.ffill().bfill()

    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    # Basic sanity: no negative prices
    if (df["Close"] <= 0).any():
        raise DataValidationError(f"Non-positive prices found for {ticker}")

    # Ensure float dtype
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype(float)
    df["Volume"] = df["Volume"].astype(float)

    logger.info(
        f"Validated {ticker}: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}"
    )
    return df


def resample_ohlcv(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Resample OHLCV data to a lower frequency (W=weekly, M=monthly)."""
    resampled = (
        df.resample(freq)
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )
    return resampled


def simulate_stream(df: pd.DataFrame, warmup: int = 50):
    """
    Generator that yields bars one-by-one (for paper-trading style replay).
    Yields (timestamp, bar_dict) tuples.
    """
    for i in range(warmup, len(df)):
        bar = df.iloc[i]
        yield df.index[i], {
            "open": bar["Open"],
            "high": bar["High"],
            "low": bar["Low"],
            "close": bar["Close"],
            "volume": bar["Volume"],
        }


def fetch_multiple(tickers: list, start: str, end: str, interval: str = "1d") -> dict:
    """Fetch OHLCV for a list of tickers. Returns dict of DataFrames."""
    result = {}
    for t in tickers:
        try:
            result[t] = fetch_ohlcv(t, start, end, interval)
        except DataValidationError as e:
            logger.error(f"Skipping {t}: {e}")
    return result
