"""Listener de trade updates de Alpaca para registrar cierres automáticos (SL/TP)."""
from __future__ import annotations

import threading
import time
from typing import Optional

from alpaca.trading.stream import TradingStream

from ..config.settings import Settings
from ..events.bus import EventBus, OrderFilledEvent
from ..execution.portfolio import PortfolioTracker
from ..execution.risk import RiskManager
from ..utils.logger import get_logger, TradeLogger

logger = get_logger("execution.trade_stream")

# Tipos de orden de CIERRE que registramos: legs de bracket (SL/TP) y trailing
# stops del protector. Sin "trailing_stop", los cierres por trailing se ignoraban
# (no se registraban en trades.log ni actualizaban el portfolio).
_CLOSE_ORDER_TYPES = {"stop", "limit", "stop_limit", "trailing_stop"}


def _str_enum(value) -> str:
    """Normaliza un valor que puede ser un str-enum de alpaca-py a string limpio."""
    if value is None:
        return ""
    s = str(value).lower()
    # alpaca-py str-enums pueden venir como "OrderType.stop_limit" → extraer la parte final
    return s.split(".")[-1]


class AlpacaTradeStream:
    """
    Escucha trade updates de Alpaca via WebSocket.

    Detecta fills de órdenes bracket child (stop-loss y take-profit) y
    registra el PnL real en trades.log sin necesitar que el bot haga un
    SELL explícito.
    """

    def __init__(
        self,
        settings: Settings,
        risk: RiskManager,
        trade_logger: TradeLogger,
        portfolio: PortfolioTracker,
        event_bus: Optional[EventBus] = None,
    ):
        self._cfg = settings
        self._risk = risk
        self._trade_logger = trade_logger
        self._portfolio = portfolio
        self._event_bus = event_bus
        self._stream: Optional[TradingStream] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = threading.Event()
        self._reconnect_delay = 1.0

    def _create_stream(self) -> TradingStream:
        return TradingStream(
            api_key=self._cfg.alpaca.api_key,
            secret_key=self._cfg.alpaca.api_secret,
            paper=self._cfg.alpaca.paper_trading,
        )

    async def _on_trade_update(self, data) -> None:
        try:
            event_type = _str_enum(getattr(data, "event", ""))
            if event_type not in ("fill", "partial_fill"):
                return

            order = getattr(data, "order", None)
            if order is None:
                return

            order_type = _str_enum(
                getattr(order, "order_type", None) or getattr(order, "type", None)
            )
            if order_type not in _CLOSE_ORDER_TYPES:
                return

            symbol = getattr(order, "symbol", None)
            if not symbol:
                return

            # Ignorar fills de símbolos sin posición registrada (evita doble-conteo
            # si el bot reinicia con posiciones ya abiertas pero sin record_fill previo)
            if not self._risk.has_open_position(symbol):
                logger.debug(f"Bracket fill ignorado (sin posición registrada): {symbol}")
                return

            fill_price = float(
                getattr(data, "price", None)
                or getattr(order, "filled_avg_price", None)
                or 0
            )
            qty = int(float(
                getattr(data, "qty", None)
                or getattr(order, "filled_qty", None)
                or 0
            ))

            if fill_price <= 0 or qty <= 0:
                logger.warning(f"Bracket fill ignorado por precio/qty inválidos: {symbol}")
                return

            order_side = _str_enum(getattr(order, "side", ""))
            order_id = str(getattr(order, "id", "unknown"))
            if "trailing" in order_type:
                close_reason = "trailing_stop"
            elif order_type == "limit":
                close_reason = "take_profit"
            else:
                close_reason = "stop_loss"

            logger.info(
                f"Bracket child fill: {order_side.upper()} {qty} {symbol} "
                f"@ ${fill_price:.2f} [{close_reason}] | ID={order_id}"
            )

            pnl = self._risk.record_fill(symbol, order_side, fill_price, qty)

            self._trade_logger.log_trade({
                "symbol": symbol,
                "action": order_side.upper(),
                "price": fill_price,
                "qty": qty,
                "close_reason": close_reason,
                "order_id": order_id,
                "pnl": pnl,
                "source": "alpaca_bracket",
            })

            if pnl is not None:
                self._portfolio.remove_position(symbol, pnl)

            if self._event_bus is not None:
                self._event_bus.post(OrderFilledEvent(
                    symbol=symbol,
                    side=order_side,
                    shares=qty,
                    fill_price=fill_price,
                    order_id=order_id,
                    stop_price=0.0,
                    take_profit_price=0.0,
                    is_close=True,                                       # el trade_stream solo CIERRA
                    pnl=pnl,
                    # la orden de cierre es opuesta a la posición: BUY cubre un corto,
                    # SELL liquida un largo
                    position_side="short" if order_side == "buy" else "long",
                    close_reason=close_reason,
                ))

        except Exception as e:
            logger.error(f"Error procesando trade update: {e}", exc_info=True)

    def _force_stop(self, stream) -> None:
        """Para el loop interno de alpaca-py desde otro hilo (thread-safe)."""
        try:
            stream._should_run = False
            if stream._stop_stream_queue.empty():
                stream._stop_stream_queue.put_nowait({"should_stop": True})
        except Exception:
            pass

    def _run_stream(self) -> None:
        while self._running.is_set():
            connect_ts = time.monotonic()
            stream = self._create_stream()
            self._stream = stream
            stream.subscribe_trade_updates(self._on_trade_update)

            logger.info("AlpacaTradeStream conectando, escuchando trade updates...")

            # Ejecutar en hilo separado para poder pararlo desde fuera.
            # TradingStream tiene el mismo comportamiento que DataStream:
            # traga "connection limit" y reintenta en bucle sin sleep.
            inner = threading.Thread(target=stream.run, daemon=True, name="alpaca-trade-inner")
            inner.start()

            # Esperar hasta 20s a que _running=True (auth OK)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and inner.is_alive() and not stream._running:
                time.sleep(0.5)

            if inner.is_alive() and not stream._running:
                self._force_stop(stream)
                inner.join(timeout=5)
                delay = max(self._reconnect_delay, 120.0)
                logger.warning(
                    f"TradeStream: no se pudo autenticar en 20s (connection/rate limit). "
                    f"Esperando {delay:.0f}s antes de reconectar..."
                )
                self._reconnect_delay = min(self._reconnect_delay * 2, 300.0)
            elif inner.is_alive():
                logger.info("AlpacaTradeStream conectado correctamente.")
                inner.join()
                elapsed = time.monotonic() - connect_ts
                if elapsed > 60:
                    self._reconnect_delay = 5.0
                delay = self._reconnect_delay
                logger.warning(f"TradeStream desconectado tras {elapsed:.0f}s. Reconectando en {delay:.0f}s...")
            else:
                elapsed = time.monotonic() - connect_ts
                delay = self._reconnect_delay
                logger.warning(f"TradeStream terminó inesperadamente en {elapsed:.1f}s. Reconectando en {delay:.0f}s...")

            if not self._running.is_set():
                break

            time.sleep(delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, 300.0)

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(
            target=self._run_stream, daemon=True, name="trade-stream"
        )
        self._thread.start()
        logger.info("AlpacaTradeStream iniciado")

    def stop(self) -> None:
        self._running.clear()
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
        logger.info("AlpacaTradeStream detenido")
