"""
Synthetic OHLCV generator for demo mode and testing.
Uses Geometric Brownian Motion (GBM) with realistic price dynamics.
"""

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    ticker: str = "SYNTHETIC",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    initial_price: float = 150.0,
    annual_drift: float = 0.10,  # 10% annual drift
    annual_vol: float = 0.25,  # 25% annual volatility
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data using GBM with intrabar spread simulation.
    """
    np.random.seed(seed)
    dates = pd.bdate_range(start=start, end=end)  # business days only
    n = len(dates)

    dt = 1 / 252
    drift = (annual_drift - 0.5 * annual_vol**2) * dt
    vol = annual_vol * np.sqrt(dt)

    # Generate log returns
    log_returns = np.random.normal(drift, vol, n)

    # Add occasional fat-tail shocks (realistic market behavior)
    shock_mask = np.random.random(n) < 0.02  # 2% chance of shock
    log_returns[shock_mask] += np.random.normal(0, vol * 3, shock_mask.sum())

    # Build close prices
    close = initial_price * np.exp(np.cumsum(log_returns))

    # Build OHLCV with realistic spread
    intraday_vol = np.abs(np.random.normal(0, 0.008, n))  # ~0.8% intraday move
    high = close * (1 + intraday_vol)
    low = close * (1 - intraday_vol)
    open_prices = close * np.exp(np.random.normal(0, vol * 0.3, n))

    # Ensure OHLC consistency
    open_prices = np.clip(open_prices, low, high)

    # Volume: mean-reverting log-normal
    vol_base = 5_000_000
    volume = np.abs(np.random.lognormal(np.log(vol_base), 0.5, n))
    # Volume spikes on high-vol days
    volume *= 1 + 3 * np.abs(log_returns) / vol

    df = pd.DataFrame(
        {
            "Open": open_prices,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume.astype(float),
        },
        index=dates,
    )

    df.index.name = "Date"
    return df


# Pre-build a few demo tickers
DEMO_TICKERS = {
    "AAPL": dict(initial_price=150, annual_drift=0.18, annual_vol=0.28, seed=42),
    "MSFT": dict(initial_price=220, annual_drift=0.22, annual_vol=0.25, seed=43),
    "GOOGL": dict(initial_price=1200, annual_drift=0.15, annual_vol=0.30, seed=44),
    "TSLA": dict(initial_price=200, annual_drift=0.30, annual_vol=0.60, seed=45),
    "SPY": dict(initial_price=300, annual_drift=0.12, annual_vol=0.18, seed=46),
    "QQQ": dict(initial_price=280, annual_drift=0.16, annual_vol=0.22, seed=47),
    "NVDA": dict(initial_price=250, annual_drift=0.40, annual_vol=0.55, seed=48),
    "AMZN": dict(initial_price=140, annual_drift=0.20, annual_vol=0.32, seed=49),
}


def get_demo_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Return synthetic demo data for a ticker."""
    params = DEMO_TICKERS.get(
        ticker.upper(),
        {"initial_price": 100, "annual_drift": 0.12, "annual_vol": 0.25, "seed": 99},
    )
    return generate_synthetic_ohlcv(ticker=ticker, start=start, end=end, **params)
