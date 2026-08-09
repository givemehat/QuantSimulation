"""
Platform-wide configuration and default parameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    default_ticker: str = "AAPL"
    default_start: str = "2020-01-01"
    default_end: str = "2024-12-31"
    interval: str = "1d"
    cache_dir: str = "data/cache"


@dataclass
class RiskConfig:
    max_position_pct: float = 0.20  # max 20% of portfolio per trade
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.15  # 15% take profit
    max_drawdown_pct: float = 0.25  # halt trading at 25% drawdown
    daily_loss_limit_pct: float = 0.03  # 3% daily loss limit
    volatility_scaling: bool = True


@dataclass
class ExecutionConfig:
    slippage_bps: float = 5.0  # 5 basis points slippage
    commission_pct: float = 0.001  # 0.1% commission
    initial_capital: float = 100_000.0


@dataclass
class MLConfig:
    test_size: float = 0.20
    val_size: float = 0.10
    n_estimators: int = 200
    random_state: int = 42
    walk_forward_windows: int = 5
    prediction_horizon: int = 5  # days ahead to predict


@dataclass
class PlatformConfig:
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    log_level: str = "INFO"


# Singleton config
CONFIG = PlatformConfig()
