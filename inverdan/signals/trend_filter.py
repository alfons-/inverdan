"""Proveedor de la tendencia mayor en un timeframe superior (p. ej. diario).

Para cada símbolo calcula la posición del precio respecto a una SMA del timeframe
configurado (por defecto SMA50 en velas diarias). El filtro de reglas usa este
valor para vetar operar contra la tendencia mayor, en lugar del SMA200 intradía
(~3,3 h sobre velas de 1 min), que cambiaba de sesgo demasiado rápido.

Las barras diarias cambian una vez al día, así que basta refrescar cada pocas
horas; `HistoricalDataClient` ya cachea en disco (1 h), por lo que los reinicios
seguidos no vuelven a descargar.
"""
from __future__ import annotations

import threading
from typing import List, Optional

from ..config.settings import Settings
from ..data.historical import HistoricalDataClient
from ..utils.logger import get_logger

logger = get_logger("signals.trend_filter")


class DailyTrendProvider:
    """Mantiene, por símbolo, `precio / SMA(timeframe superior) - 1`. Thread-safe."""

    def __init__(self, settings: Settings, hist_client: Optional[HistoricalDataClient] = None):
        self._cfg = settings
        self._hist = hist_client or HistoricalDataClient(settings)
        self._tf = settings.risk.trend_timeframe
        self._period = settings.risk.trend_sma_period
        self._trend: dict[str, float] = {}
        self._lock = threading.Lock()

    def refresh(self, symbols: List[str]) -> None:
        """Recalcula la tendencia mayor de cada símbolo (precio vs SMA del timeframe)."""
        # Días naturales suficientes para `period` velas del timeframe. Para velas
        # diarias se necesitan ~1,5 días naturales por vela de mercado.
        days = max(int(self._period * 1.6) + 40, 90) if self._tf == "1Day" else self._period
        updated: dict[str, float] = {}
        for sym in symbols:
            try:
                df = self._hist.fetch_bars(sym, days=days, timeframe=self._tf, cache=True)
                if df is None or df.empty:
                    continue
                close = df["close"].dropna()
                if len(close) < self._period:
                    continue
                sma = close.rolling(self._period).mean().iloc[-1]
                last = close.iloc[-1]
                if sma and sma > 0:
                    updated[sym] = float(last / sma - 1)
            except Exception as e:
                logger.warning(f"refresh de tendencia {sym}: {e}")
        if updated:
            with self._lock:
                self._trend.update(updated)
            logger.info(
                f"Tendencia mayor ({self._tf} SMA{self._period}) actualizada — "
                + ", ".join(f"{s} {v * 100:+.1f}%" for s, v in sorted(updated.items()))
            )

    def get(self, symbol: str) -> Optional[float]:
        """Devuelve precio/SMA-1 del símbolo, o None si aún no se ha calculado."""
        with self._lock:
            return self._trend.get(symbol)
