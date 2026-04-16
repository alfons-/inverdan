"""Estado compartido thread-safe para el dashboard."""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, List, Optional

from ..execution.portfolio import PortfolioSnapshot
from ..signals.signal_types import Signal


@dataclass
class SignalEntry:
    timestamp: datetime
    symbol: str
    action: str
    confidence: float
    price: float
    reasoning: str


class DashboardState:
    """Estado global del dashboard. Thread-safe."""

    def __init__(self, max_signals: int = 20, max_logs: int = 50):
        self._lock = threading.RLock()
        self._portfolio: Optional[PortfolioSnapshot] = None
        self._signals: Deque[SignalEntry] = deque(maxlen=max_signals)
        self._logs: Deque[str] = deque(maxlen=max_logs)
        self._last_prices: dict = {}
        self._started_at: datetime = datetime.utcnow()
        self._auto_trade: bool = False

    def update_portfolio(self, snapshot: PortfolioSnapshot) -> None:
        with self._lock:
            self._portfolio = snapshot

    def add_signal(self, signal: Signal) -> None:
        with self._lock:
            self._signals.appendleft(SignalEntry(
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                action=signal.action,
                confidence=signal.confidence,
                price=signal.price,
                reasoning=signal.reasoning,
            ))

    def add_log(self, msg: str) -> None:
        with self._lock:
            ts = datetime.utcnow().strftime("%H:%M:%S")
            self._logs.appendleft(f"[{ts}] {msg}")

    def update_price(self, symbol: str, price: float) -> None:
        with self._lock:
            self._last_prices[symbol] = price

    def get_snapshot(self) -> dict:
        with self._lock:
            return {
                "portfolio": self._portfolio,
                "signals": list(self._signals),
                "logs": list(self._logs),
                "prices": dict(self._last_prices),
                "started_at": self._started_at,
                "auto_trade": self._auto_trade,
            }

    def toggle_auto_trade(self) -> bool:
        with self._lock:
            self._auto_trade = not self._auto_trade
            return self._auto_trade

    @property
    def auto_trade(self) -> bool:
        return self._auto_trade
