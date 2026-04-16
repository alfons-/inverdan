"""Ejecutor de operaciones: señal → validación → orden → confirmación."""
from __future__ import annotations

import threading
from typing import Optional

from ..config.settings import Settings
from ..events.bus import EventBus, SignalEvent, OrderFilledEvent, OrderRejectedEvent
from ..execution.broker import AlpacaBroker
from ..execution.portfolio import PortfolioTracker, Position
from ..execution.risk import RiskManager
from ..signals.signal_types import Signal
from ..utils.logger import get_logger, TradeLogger
from ..utils.rate_limiter import RateLimiter

logger = get_logger("execution.executor")


class TradeExecutor:
    """Consume SignalEvents del bus y ejecuta operaciones via Alpaca."""

    def __init__(
        self,
        settings: Settings,
        broker: AlpacaBroker,
        risk: RiskManager,
        portfolio: PortfolioTracker,
        event_bus: EventBus,
    ):
        self._cfg = settings
        self._broker = broker
        self._risk = risk
        self._portfolio = portfolio
        self._bus = event_bus
        self._rate_limiter = RateLimiter(settings.risk.max_orders_per_minute)
        self._trade_logger = TradeLogger(settings.logs_path)
        self._enabled = threading.Event()
        self._enabled.set()

        # Suscribirse al bus de señales
        self._bus.subscribe(SignalEvent, self._on_signal_event)

    def _on_signal_event(self, event: SignalEvent) -> None:
        signal = Signal(
            symbol=event.symbol,
            action=event.action,
            confidence=event.confidence,
            price=event.price,
            reasoning=event.reasoning,
            timestamp=event.timestamp,
            indicators=event.indicators,
        )
        self._process_signal(signal)

    def _process_signal(self, signal: Signal) -> None:
        if not self._enabled.is_set():
            return

        if signal.action == "HOLD":
            return

        # Verificar rate limit
        if not self._rate_limiter.acquire():
            logger.warning(f"Rate limit alcanzado, descartando señal {signal.symbol}")
            return

        # Obtener valor del portfolio
        portfolio_value = self._broker.get_portfolio_value()
        if portfolio_value <= 0:
            return

        # Verificar riesgo
        approved, reason = self._risk.approve(signal, portfolio_value)
        if not approved:
            logger.debug(f"Señal rechazada {signal.symbol}: {reason}")
            self._bus.post(OrderRejectedEvent(symbol=signal.symbol, reason=reason))
            return

        # Calcular tamaño de posición
        atr = signal.indicators.get("atr", signal.price * 0.01)
        qty = self._risk.size_position(signal, portfolio_value, atr)
        if qty <= 0:
            self._bus.post(OrderRejectedEvent(symbol=signal.symbol, reason="Tamaño de posición = 0"))
            return

        # Calcular stops
        stop_loss, take_profit = self._risk.compute_stops(
            signal.price, atr, signal.action
        )

        # Validación final de stops
        if signal.action == "BUY" and stop_loss >= signal.price:
            self._bus.post(OrderRejectedEvent(symbol=signal.symbol, reason="Stop-loss inválido"))
            return
        if signal.action == "SELL" and stop_loss <= signal.price:
            self._bus.post(OrderRejectedEvent(symbol=signal.symbol, reason="Stop-loss inválido (short)"))
            return

        side = "buy" if signal.action == "BUY" else "sell"

        logger.info(
            f"Ejecutando {signal.action} {qty} {signal.symbol} @ ~{signal.price:.2f} "
            f"| SL={stop_loss:.2f} TP={take_profit:.2f} | conf={signal.confidence:.2f}"
        )

        # Enviar orden bracket
        order = self._broker.submit_bracket_order(
            symbol=signal.symbol,
            side=side,
            qty=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        if order:
            fill_price = signal.price  # Aproximación (orden de mercado)
            self._risk.record_fill(signal.symbol, side, fill_price, qty)

            pos = Position(
                symbol=signal.symbol,
                side="long" if side == "buy" else "short",
                qty=qty,
                entry_price=fill_price,
                current_price=fill_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            self._portfolio.add_position(pos)

            filled_event = OrderFilledEvent(
                symbol=signal.symbol,
                side=side,
                shares=qty,
                fill_price=fill_price,
                order_id=str(order.id),
                stop_price=stop_loss,
                take_profit_price=take_profit,
            )
            self._bus.post(filled_event)

            # Audit trail
            self._trade_logger.log_trade({
                **signal.to_dict(),
                "qty": qty,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "order_id": str(order.id),
            })
        else:
            self._bus.post(OrderRejectedEvent(symbol=signal.symbol, reason="Orden rechazada por broker"))

    def pause(self) -> None:
        self._enabled.clear()
        logger.info("Executor pausado")

    def resume(self) -> None:
        self._enabled.set()
        logger.info("Executor reanudado")
