# ⬡ QuantTerminal — AI-Driven Quantitative Trading Platform

A production-grade quantitative research and simulation platform built in Python. Designed to resemble an institutional strategy lab, not a tutorial project.

---

## Features

- **4 Rule-Based Strategies** — MA Crossover, Momentum (with vol-regime filter), Mean Reversion (BB + RSI), Breakout
- **3 ML Models** — Random Forest, XGBoost, Logistic Regression with walk-forward validation
- **Realistic Backtesting** — slippage, commissions, stop-loss, take-profit, position sizing
- **Risk Controls** — max drawdown halt, daily loss limit, volatility-scaled sizing
- **Analytics** — equity curve, drawdown, Sharpe/Sortino, monthly heatmap, feature importance
- **Professional UI** — Bloomberg-style dark terminal dashboard in Streamlit

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
quantplatform/
├── app.py                    # Streamlit dashboard (entry point)
├── requirements.txt
├── data/
│   ├── ingestion.py          # yfinance fetch, validation, caching, streaming
│   └── synthetic.py          # GBM synthetic data generator (demo/testing)
├── features/
│   └── engineering.py        # 45-feature pipeline (anti-lookahead shifted)
├── strategies/
│   ├── rule_based.py         # MA Crossover, Momentum, Mean Reversion, Breakout
│   └── ml_strategy.py        # RF/XGBoost/Logistic with walk-forward CV
├── portfolio/
│   └── portfolio.py          # Cash, positions, trade ledger, PnL accounting
├── risk/
│   └── risk_manager.py       # Stop-loss, take-profit, drawdown limit, vol-sizing
├── backtesting/
│   └── engine.py             # Event-driven backtest + metrics computation
├── analytics/
│   └── charts.py             # Plotly charts: equity, signals, heatmap, features
├── config/
│   └── settings.py           # Platform-wide typed configuration dataclasses
├── logs/                     # Log output directory
└── tests/                    # Unit tests
```

---

## How It Works

### Data Layer
`fetch_ohlcv()` downloads OHLCV from yfinance, validates schema, handles missing values, and caches to Parquet. Falls back to synthetic GBM data when offline.

### Feature Engineering
`compute_features()` builds a 45-column feature matrix including SMA/EMA, MACD, RSI, Bollinger Bands, ATR, rolling z-scores, volume ratios, and lagged returns. **All features are shifted by +1 bar to prevent look-ahead bias.**

### Strategies
- **Rule-based**: Signal series of {1=long, -1=short, 0=flat} generated from price/indicator logic
- **ML**: Trains a classifier to predict whether price will be higher in N days. Uses walk-forward cross-validation (no data leakage). Trades when model confidence exceeds threshold.

### Backtesting Engine
Bar-by-bar simulation with:
- Slippage applied at execution price
- Commission deducted per trade
- Stop-loss and take-profit checked every bar
- Max drawdown halt: all positions liquidated if triggered
- Volatility-scaled position sizing (optional)

### Risk Manager
Every order passes through `RiskManager` before execution:
- Position size computed as fraction of equity (vol-adjusted)
- Daily loss limit checked
- Drawdown monitor halts trading when limit hit

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Features shifted +1 | Strict anti-lookahead: signals computed on prior bar's data only |
| Walk-forward CV | Time-series safe; avoids data leakage from future |
| Event-driven backtest | Bar-by-bar stop/take-profit triggering is more realistic |
| Parquet caching | Fast re-runs without re-downloading |
| Strategy registry | Easy to add new strategies without changing the UI |

---

## Extending the Platform

**Add a new strategy:**
```python
# strategies/rule_based.py
class MyNewStrategy(BaseStrategy):
    name = "My Strategy"
    def generate_signals(self, df):
        ...
        return signals  # pd.Series of {-1, 0, 1}

STRATEGY_REGISTRY["My Strategy"] = MyNewStrategy
```

**Add a new ML model:**
```python
# strategies/ml_strategy.py — in _build_model()
elif self.model_type == "lightgbm":
    from lightgbm import LGBMClassifier
    return LGBMClassifier(...)
```

**Live/paper trading extension:**
Replace `data/ingestion.py` with a real-time WebSocket feed (Alpaca, Polygon, etc.) and swap `backtesting/engine.py` with a live execution adapter. The `Portfolio` and `RiskManager` classes are already broker-agnostic.

---

## Performance Notes

On synthetic data (AAPL-like, 2020-2024):
- MA Crossover: ~8-14% return, 30-90 trades
- Momentum: ~14% return with vol filter
- ML (Random Forest): ~35% return, 50-70% walk-forward accuracy

Real market performance will vary. This is a research tool, not financial advice.
