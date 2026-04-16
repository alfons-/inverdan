"""Tracker de portfolio y posiciones abiertas en tiempo real."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Position:
    symbol: str
    side: str          # "long" | "short"
    qty: int
    entry_price: float
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.qty
        return (self.entry_price - self.current_price) * self.qty

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return self.unrealized_pnl / (self.entry_price * self.qty)

    @property
    def market_value(self) -> float:
        return self.current_price * self.qty


@dataclass
class PortfolioSnapshot:
    equity: float = 0.0
    buying_power: float = 0.0
    daily_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    positions: List[Position] = field(default_factory=list)
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0


class PortfolioTracker:
    """Mantiene el estado del portfolio en tiempo real. Thread-safe."""

    def __init__(self):
        self._positions: Dict[str, Position] = {}
        self._equity: float = 0.0
        self._buying_power: float = 0.0
        self._daily_pnl: float = 0.0
        self._trades_today: int = 0
        self._wins_today: int = 0
        self._losses_today: int = 0
        self._lock = threading.RLock()

    def sync_from_broker(self, broker) -> None:
        """Sincroniza con el estado real de Alpaca."""
        try:
            account = broker.get_account()
            positions = broker.get_positions()

            with self._lock:
                self._equity = float(account.equity)
                self._buying_power = float(account.buying_power)

                self._positions.clear()
                for pos in positions:
                    p = Position(
                        symbol=pos.symbol,
                        side="long" if float(pos.qty) > 0 else "short",
                        qty=abs(int(float(pos.qty))),
                        entry_price=float(pos.avg_entry_price),
                        current_price=float(pos.current_price),
                    )
                    self._positions[pos.symbol] = p
        except Exception:
            pass

    def update_price(self, symbol: str, price: float) -> None:
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].current_price = price

    def add_position(self, pos: Position) -> None:
        with self._lock:
            self._positions[pos.symbol] = pos

    def remove_position(self, symbol: str, pnl: float) -> None:
        with self._lock:
            self._positions.pop(symbol, None)
            self._daily_pnl += pnl
            self._trades_today += 1
            if pnl >= 0:
                self._wins_today += 1
            else:
                self._losses_today += 1

    def get_snapshot(self) -> PortfolioSnapshot:
        with self._lock:
            positions = list(self._positions.values())
            total_upnl = sum(p.unrealized_pnl for p in positions)
            return PortfolioSnapshot(
                equity=self._equity,
                buying_power=self._buying_power,
                daily_pnl=self._daily_pnl,
                total_unrealized_pnl=total_upnl,
                positions=positions,
                trades_today=self._trades_today,
                wins_today=self._wins_today,
                losses_today=self._losses_today,
            )

    def reset_daily(self) -> None:
        with self._lock:
            self._daily_pnl = 0.0
            self._trades_today = 0
            self._wins_today = 0
            self._losses_today = 0
