"""Token bucket rate limiter thread-safe."""
from __future__ import annotations

import threading
import time


class RateLimiter:
    """Limita el número de acciones por minuto usando token bucket."""

    def __init__(self, max_per_minute: int):
        self._max = max_per_minute
        self._tokens = float(max_per_minute)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max, self._tokens + elapsed * (self._max / 60.0))
        self._last_refill = now

    def acquire(self) -> bool:
        """Retorna True si se puede proceder, False si se ha excedido el límite."""
        with self._lock:
            self._refill()
            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False

    def wait_and_acquire(self) -> None:
        """Bloquea hasta que haya un token disponible."""
        while not self.acquire():
            time.sleep(0.1)
