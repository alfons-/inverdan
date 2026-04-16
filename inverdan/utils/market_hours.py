"""Utilidades de horario de mercado."""
from __future__ import annotations

from datetime import datetime, time
import pytz

_ET = pytz.timezone("America/New_York")
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)


def now_et() -> datetime:
    return datetime.now(_ET)


def is_market_open(dt: datetime | None = None) -> bool:
    """Retorna True si el mercado USA está abierto en este momento."""
    dt = (dt or datetime.now()).astimezone(_ET)
    if dt.weekday() >= 5:  # Sábado o domingo
        return False
    t = dt.time()
    return _MARKET_OPEN <= t < _MARKET_CLOSE


def market_hours_et() -> tuple[time, time]:
    return _MARKET_OPEN, _MARKET_CLOSE


def minutes_to_close() -> int:
    """Minutos que faltan para el cierre del mercado."""
    dt = now_et()
    close_dt = dt.replace(hour=16, minute=0, second=0, microsecond=0)
    delta = (close_dt - dt).total_seconds()
    return max(0, int(delta / 60))
