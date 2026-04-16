"""Buffer circular por símbolo para barras OHLCV en tiempo real."""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import pandas as pd


@dataclass
class OHLCVBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None


class SymbolBuffer:
    """Mantiene las últimas N barras de un símbolo. Thread-safe."""

    def __init__(self, symbol: str, maxlen: int = 500):
        self.symbol = symbol
        self._deque: deque[OHLCVBar] = deque(maxlen=maxlen)
        self._lock = threading.RLock()
        # Acumuladores VWAP intraday
        self._vwap_price_vol = 0.0
        self._vwap_vol = 0.0
        self._vwap_date: Optional[str] = None

    def update(self, bar: OHLCVBar) -> None:
        with self._lock:
            # Resetear VWAP al inicio de cada día
            bar_date = bar.timestamp.strftime("%Y-%m-%d")
            if bar_date != self._vwap_date:
                self._vwap_price_vol = 0.0
                self._vwap_vol = 0.0
                self._vwap_date = bar_date

            typical = (bar.high + bar.low + bar.close) / 3
            self._vwap_price_vol += typical * bar.volume
            self._vwap_vol += bar.volume
            bar.vwap = self._vwap_price_vol / self._vwap_vol if self._vwap_vol else bar.close

            self._deque.append(bar)

    def get_dataframe(self) -> pd.DataFrame:
        with self._lock:
            if not self._deque:
                return pd.DataFrame()
            rows = [
                {
                    "timestamp": b.timestamp,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                    "vwap": b.vwap,
                }
                for b in self._deque
            ]
        df = pd.DataFrame(rows).set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)

    @property
    def last_close(self) -> Optional[float]:
        with self._lock:
            return self._deque[-1].close if self._deque else None


class BufferRegistry:
    """Registro central de buffers por símbolo."""

    def __init__(self, maxlen: int = 500):
        self._buffers: Dict[str, SymbolBuffer] = {}
        self._maxlen = maxlen

    def get_or_create(self, symbol: str) -> SymbolBuffer:
        if symbol not in self._buffers:
            self._buffers[symbol] = SymbolBuffer(symbol, self._maxlen)
        return self._buffers[symbol]

    def symbols(self) -> list:
        return list(self._buffers.keys())
