"""
Portfolio accounting: tracks cash, positions, PnL, and trade ledger.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Position:
    ticker: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    entry_date: Optional[pd.Timestamp] = None

    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_cost

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0


@dataclass
class Trade:
    date: pd.Timestamp
    ticker: str
    side: str          # "BUY" or "SELL"
    quantity: float
    price: float
    commission: float
    slippage: float
    pnl: float = 0.0

    @property
    def gross_value(self) -> float:
        return self.quantity * self.price

    @property
    def net_cost(self) -> float:
        return self.gross_value + self.commission + self.slippage


class Portfolio:
    """
    Tracks cash, positions, trade ledger, and equity curve.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.001,
        slippage_bps: float = 5.0,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_pct = commission_pct
        self.slippage_bps = slippage_bps / 10_000.0  # convert bps to decimal

        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[dict] = []
        self._peak_equity = initial_capital

    @property
    def total_equity(self) -> float:
        """Cash + market value of all open positions (at avg cost)."""
        pos_value = sum(p.quantity * p.avg_cost for p in self.positions.values())
        return self.cash + pos_value

    def update_equity(self, date: pd.Timestamp, prices: Dict[str, float]) -> float:
        """Mark positions to market and record equity snapshot."""
        pos_value = sum(
            self.positions[t].quantity * prices.get(t, self.positions[t].avg_cost)
            for t in self.positions
        )
        equity = self.cash + pos_value
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (equity - self._peak_equity) / (self._peak_equity + 1e-9)

        self.equity_curve.append({
            "date": date,
            "equity": equity,
            "cash": self.cash,
            "positions_value": pos_value,
            "drawdown": drawdown,
        })
        return equity

    def buy(
        self,
        date: pd.Timestamp,
        ticker: str,
        quantity: float,
        price: float,
    ) -> Optional[Trade]:
        """Execute a buy order with slippage and commission."""
        if quantity <= 0:
            return None

        exec_price = price * (1 + self.slippage_bps)
        gross_cost = quantity * exec_price
        commission = gross_cost * self.commission_pct
        total_cost = gross_cost + commission

        if total_cost > self.cash:
            # Scale down quantity to fit available cash
            affordable = self.cash / (exec_price * (1 + self.commission_pct))
            if affordable < 1:
                logger.warning(f"Insufficient cash to buy {ticker}")
                return None
            quantity = np.floor(affordable)
            gross_cost = quantity * exec_price
            commission = gross_cost * self.commission_pct
            total_cost = gross_cost + commission

        self.cash -= total_cost

        pos = self.positions.get(ticker, Position(ticker))
        new_qty = pos.quantity + quantity
        pos.avg_cost = (pos.quantity * pos.avg_cost + quantity * exec_price) / new_qty
        pos.quantity = new_qty
        pos.entry_date = date
        self.positions[ticker] = pos

        trade = Trade(
            date=date, ticker=ticker, side="BUY",
            quantity=quantity, price=exec_price,
            commission=commission,
            slippage=quantity * price * self.slippage_bps,
        )
        self.trades.append(trade)
        logger.debug(f"BUY {quantity:.0f} {ticker} @ {exec_price:.2f} | cash={self.cash:.0f}")
        return trade

    def sell(
        self,
        date: pd.Timestamp,
        ticker: str,
        quantity: float,
        price: float,
    ) -> Optional[Trade]:
        """Execute a sell/close order."""
        pos = self.positions.get(ticker)
        if pos is None or pos.quantity <= 0:
            return None

        quantity = min(quantity, pos.quantity)
        exec_price = price * (1 - self.slippage_bps)
        gross_proceeds = quantity * exec_price
        commission = gross_proceeds * self.commission_pct
        net_proceeds = gross_proceeds - commission

        pnl = (exec_price - pos.avg_cost) * quantity - commission
        self.cash += net_proceeds
        pos.quantity -= quantity

        if pos.quantity <= 0.001:
            del self.positions[ticker]

        trade = Trade(
            date=date, ticker=ticker, side="SELL",
            quantity=quantity, price=exec_price,
            commission=commission,
            slippage=quantity * price * self.slippage_bps,
            pnl=pnl,
        )
        self.trades.append(trade)
        logger.debug(f"SELL {quantity:.0f} {ticker} @ {exec_price:.2f} | PnL={pnl:.2f} | cash={self.cash:.0f}")
        return trade

    def close_all(self, date: pd.Timestamp, prices: Dict[str, float]):
        """Liquidate all positions at given prices."""
        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            if pos.quantity > 0:
                self.sell(date, ticker, pos.quantity, prices.get(ticker, pos.avg_cost))

    def get_equity_df(self) -> pd.DataFrame:
        if not self.equity_curve:
            return pd.DataFrame()
        return pd.DataFrame(self.equity_curve).set_index("date")

    def get_trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        rows = [{
            "Date": t.date,
            "Ticker": t.ticker,
            "Side": t.side,
            "Quantity": t.quantity,
            "Price": t.price,
            "Commission": t.commission,
            "Slippage": t.slippage,
            "PnL": t.pnl,
        } for t in self.trades]
        return pd.DataFrame(rows)

    def reset(self):
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self._peak_equity = self.initial_capital
