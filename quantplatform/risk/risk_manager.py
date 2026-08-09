"""
Risk management layer.
Handles position sizing, drawdown limits, and trade rejection logic.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Gate every order through risk controls before it reaches execution.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        max_position_pct: float = 0.20,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15,
        max_drawdown_pct: float = 0.25,
        daily_loss_limit_pct: float = 0.03,
        volatility_scaling: bool = True,
        vol_target: float = 0.15,  # annualized vol target
    ):
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.volatility_scaling = volatility_scaling
        self.vol_target = vol_target

        self._peak_equity = initial_capital
        self._daily_start_equity: Optional[float] = None
        self._trading_halted = False
        self._halt_reason = ""

        # Track entry prices for stop/take-profit
        self._entry_prices: dict = {}

    def check_drawdown(self, current_equity: float) -> bool:
        """Returns False if max drawdown exceeded (halt trading)."""
        self._peak_equity = max(self._peak_equity, current_equity)
        drawdown = (self._peak_equity - current_equity) / (self._peak_equity + 1e-9)

        if drawdown >= self.max_drawdown_pct:
            self._trading_halted = True
            self._halt_reason = f"Max drawdown {drawdown:.1%} exceeded limit {self.max_drawdown_pct:.1%}"
            logger.warning(f"RISK HALT: {self._halt_reason}")
            return False
        return True

    def check_daily_loss(self, current_equity: float, date) -> bool:
        """Check daily loss limit. Reset at start of new day."""
        if self._daily_start_equity is None:
            self._daily_start_equity = current_equity
            return True

        daily_loss = (self._daily_start_equity - current_equity) / (
            self._daily_start_equity + 1e-9
        )
        if daily_loss >= self.daily_loss_limit_pct:
            logger.warning(f"Daily loss limit {daily_loss:.1%} hit on {date}")
            return False
        return True

    def reset_daily(self, equity: float):
        """Call at the start of each new trading day."""
        self._daily_start_equity = equity

    def size_position(
        self,
        equity: float,
        price: float,
        volatility: Optional[float] = None,
    ) -> int:
        """
        Compute number of shares to buy.
        Uses fixed-fractional sizing with optional volatility scaling.
        """
        if self._trading_halted:
            return 0

        base_allocation = equity * self.max_position_pct

        if self.volatility_scaling and volatility is not None and volatility > 0:
            # Scale position size inversely with realized volatility
            scale = min(self.vol_target / (volatility + 1e-9), 2.0)
            base_allocation *= scale

        shares = int(base_allocation / (price + 1e-9))
        return max(shares, 0)

    def check_stop_loss(self, ticker: str, current_price: float) -> bool:
        """Returns True if stop-loss triggered (should exit)."""
        entry = self._entry_prices.get(ticker)
        if entry is None:
            return False
        loss = (entry - current_price) / (entry + 1e-9)
        if loss >= self.stop_loss_pct:
            logger.info(
                f"Stop-loss triggered: {ticker} entry={entry:.2f} current={current_price:.2f}"
            )
            return True
        return False

    def check_take_profit(self, ticker: str, current_price: float) -> bool:
        """Returns True if take-profit triggered (should exit)."""
        entry = self._entry_prices.get(ticker)
        if entry is None:
            return False
        gain = (current_price - entry) / (entry + 1e-9)
        if gain >= self.take_profit_pct:
            logger.info(
                f"Take-profit triggered: {ticker} entry={entry:.2f} current={current_price:.2f}"
            )
            return True
        return False

    def record_entry(self, ticker: str, price: float):
        self._entry_prices[ticker] = price

    def clear_entry(self, ticker: str):
        self._entry_prices.pop(ticker, None)

    def is_halted(self) -> bool:
        return self._trading_halted

    def get_halt_reason(self) -> str:
        return self._halt_reason

    def reset(self):
        self._peak_equity = self.initial_capital
        self._daily_start_equity = None
        self._trading_halted = False
        self._halt_reason = ""
        self._entry_prices = {}
