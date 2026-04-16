"""Bus de eventos thread-safe para comunicación entre componentes."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class BarReadyEvent:
    symbol: str
    timestamp: datetime
    close: float
    volume: int


@dataclass
class SignalEvent:
    symbol: str
    action: str          # BUY / SELL / HOLD
    confidence: float
    price: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    indicators: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderFilledEvent:
    symbol: str
    side: str
    shares: int
    fill_price: float
    order_id: str
    stop_price: float
    take_profit_price: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OrderRejectedEvent:
    symbol: str
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


_AnyEvent = Any


class EventBus:
    """Bus central de eventos. Thread-safe."""

    def __init__(self, maxsize: int = 1000):
        self._queue: queue.Queue[_AnyEvent] = queue.Queue(maxsize=maxsize)
        self._subscribers: Dict[type, list] = {}
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()

    def post(self, event: _AnyEvent) -> None:
        if self._running.is_set():
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                pass  # Descartar evento si el bus está saturado

    def subscribe(self, event_type: type, callback) -> None:
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def dispatch_loop(self) -> None:
        """Bucle principal de despacho. Llamar en un hilo dedicado."""
        while self._running.is_set():
            try:
                event = self._queue.get(timeout=0.5)
                self._dispatch(event)
                self._queue.task_done()
            except queue.Empty:
                continue

    def _dispatch(self, event: _AnyEvent) -> None:
        with self._lock:
            callbacks = list(self._subscribers.get(type(event), []))
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                pass

    def shutdown(self) -> None:
        self._running.clear()
