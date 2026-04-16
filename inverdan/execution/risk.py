"""Gestión de riesgo: circuit breakers, sizing de posición, stops."""
from __future__ import annotations

import threading
from typing import Dict, Optional

from ..config.settings import Settings
from ..signals.signal_types import Signal
from ..utils.logger import get_logger

logger = get_logger("execution.risk")


class RiskManager:
    """Controla el riesgo antes de ejecutar cualquier operación."""

    def __init__(self, settings: Settings):
        self._cfg = settings.risk
        self._lock = threading.Lock()

        # Estado dinámico
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._orders_this_minute: int = 0
        self._last_minute: int = 0
        self._open_positions: Dict[str, float] = {}  # symbol -> entry_price
        self._total_exposure: float = 0.0
        self._circuit_open: bool = False

    def approve(self, signal: Signal, portfolio_value: float) -> tuple[bool, str]:
        """
        Verifica si la señal puede ejecutarse.
        Retorna (aprobado, motivo_rechazo).
        """
        with self._lock:
            cfg = self._cfg

            # Circuit breaker global
            if self._circuit_open:
                return False, "Circuit breaker activo"

            # Solo BUY/SELL
            if signal.action == "HOLD":
                return False, "Señal HOLD"

            # Precio mínimo
            if signal.price < cfg.min_stock_price:
                return False, f"Precio ${signal.price:.2f} < mínimo ${cfg.min_stock_price}"

            # Confianza mínima
            if signal.confidence < 0.5:
                return False, f"Confianza baja ({signal.confidence:.2f})"

            # Pérdida diaria máxima
            daily_loss_pct = abs(self._daily_pnl) / max(portfolio_value, 1)
            if self._daily_pnl < 0 and daily_loss_pct > cfg.max_daily_loss_pct:
                self._circuit_open = True
                return False, f"Pérdida diaria límite alcanzada ({daily_loss_pct:.1%})"

            # Pérdidas consecutivas
            if self._consecutive_losses >= cfg.max_consecutive_losses:
                self._circuit_open = True
                return False, f"Demasiadas pérdidas consecutivas ({self._consecutive_losses})"

            # No duplicar posición existente
            if signal.action == "BUY" and signal.symbol in self._open_positions:
                return False, f"Ya hay posición abierta en {signal.symbol}"

            # Exposición máxima total
            if self._total_exposure >= cfg.max_total_exposure * portfolio_value:
                return False, f"Exposición máxima alcanzada ({self._total_exposure/portfolio_value:.1%})"

            return True, ""

    def size_position(self, signal: Signal, portfolio_value: float, atr: float) -> int:
        """
        Calcula el número de acciones usando ATR-based position sizing.
        Arriesga max_position_pct del portfolio dividido entre 2*ATR por acción.
        """
        if atr <= 0 or signal.price <= 0:
            return 0

        max_risk = portfolio_value * self._cfg.max_position_pct
        risk_per_share = 2.0 * atr  # Stop-loss a 2*ATR
        shares = int(max_risk / risk_per_share)

        # Límite adicional: no gastar más del max_position_pct en una posición
        max_shares_by_value = int((portfolio_value * self._cfg.max_position_pct) / signal.price)
        shares = min(shares, max_shares_by_value)

        return max(1, shares)

    def compute_stops(
        self, entry_price: float, atr: float, action: str
    ) -> tuple[float, float]:
        """Calcula stop-loss y take-profit basados en ATR."""
        sl_dist = atr * self._cfg.stop_loss_atr_multiplier
        tp_dist = atr * self._cfg.take_profit_atr_multiplier

        if action == "BUY":
            stop_loss = entry_price - sl_dist
            take_profit = entry_price + tp_dist
        else:  # SELL (short)
            stop_loss = entry_price + sl_dist
            take_profit = entry_price - tp_dist

        return round(stop_loss, 2), round(take_profit, 2)

    def record_fill(self, symbol: str, side: str, price: float, shares: int) -> None:
        with self._lock:
            if side == "buy":
                self._open_positions[symbol] = price
                self._total_exposure += price * shares
            else:
                entry = self._open_positions.pop(symbol, price)
                pnl = (price - entry) * shares
                self._daily_pnl += pnl
                self._total_exposure -= price * shares

                if pnl < 0:
                    self._consecutive_losses += 1
                else:
                    self._consecutive_losses = 0

                logger.info(
                    f"Trade cerrado {symbol}: PnL=${pnl:+.2f} | "
                    f"PnL día: ${self._daily_pnl:+.2f} | "
                    f"Pérdidas consecutivas: {self._consecutive_losses}"
                )

    def reset_daily(self) -> None:
        """Llamar al inicio de cada jornada."""
        with self._lock:
            self._daily_pnl = 0.0
            self._consecutive_losses = 0
            self._circuit_open = False
            logger.info("Contadores diarios reseteados")

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def circuit_open(self) -> bool:
        return self._circuit_open

    @property
    def open_positions_count(self) -> int:
        return len(self._open_positions)
