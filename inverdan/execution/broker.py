"""Wrapper del cliente de trading de Alpaca."""
from __future__ import annotations

import time
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderStatus, OrderType, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)

from ..config.settings import Settings
from ..utils.logger import get_logger

logger = get_logger("execution.broker")


class AlpacaBroker:
    def __init__(self, settings: Settings):
        self._cfg = settings
        self._client = TradingClient(
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.api_secret,
            paper=settings.alpaca.paper_trading,
        )
        self._verify_account()

    def _verify_account(self) -> None:
        delays = [5, 15, 30, 60]
        for attempt, delay in enumerate(delays + [None], start=1):
            try:
                account = self._client.get_account()
                mode = "PAPER" if self._cfg.alpaca.paper_trading else "*** LIVE ***"
                logger.info(
                    f"Cuenta Alpaca ({mode}): "
                    f"equity=${float(account.equity):,.2f} "
                    f"buying_power=${float(account.buying_power):,.2f} "
                    f"status={account.status}"
                )
                if not self._cfg.alpaca.paper_trading:
                    logger.warning("¡ATENCIÓN! Operando con DINERO REAL")
                return
            except Exception as e:
                if delay is None:
                    raise
                logger.warning(
                    f"No se pudo conectar con Alpaca (intento {attempt}/4): {e}. "
                    f"Reintentando en {delay}s..."
                )
                time.sleep(delay)

    def get_account(self):
        return self._client.get_account()

    def get_positions(self) -> list:
        return self._client.get_all_positions()

    def get_portfolio_value(self) -> float:
        account = self._client.get_account()
        return float(account.equity)

    def get_buying_power(self) -> float:
        account = self._client.get_account()
        return float(account.buying_power)

    def get_open_orders(self, symbols: Optional[list[str]] = None) -> list:
        """Devuelve las órdenes activas (incluyendo hijas de brackets)."""
        try:
            req = GetOrdersRequest(
                status="open",
                nested=True,
                symbols=symbols,
            )
            return self._client.get_orders(filter=req) or []
        except Exception as e:
            logger.warning(f"No se pudieron leer órdenes abiertas: {e}")
            return []

    def get_bracket_stops(self, symbols: Optional[list[str]] = None) -> dict:
        """
        Recorre órdenes abiertas y devuelve por símbolo los precios
        de stop-loss y take-profit deducidos de las legs hijas.
        Estructura: {symbol: {"stop_loss": float, "take_profit": float}}
        """
        result: dict = {}
        for order in self.get_open_orders(symbols):
            try:
                sym = getattr(order, "symbol", None)
                if not sym:
                    continue
                # Recorremos legs hijas si existen (orden bracket)
                legs = getattr(order, "legs", None) or []
                candidates = [order, *legs]
                for o in candidates:
                    sym_o = getattr(o, "symbol", sym)
                    o_type = getattr(o, "order_type", None) or getattr(o, "type", None)
                    o_type = str(o_type).lower() if o_type else ""
                    sp = getattr(o, "stop_price", None)
                    lp = getattr(o, "limit_price", None)
                    if sp is not None and "stop" in o_type:
                        result.setdefault(sym_o, {})["stop_loss"] = float(sp)
                    if lp is not None and "limit" in o_type and "stop" not in o_type:
                        result.setdefault(sym_o, {})["take_profit"] = float(lp)
            except Exception:
                continue
        return result

    def submit_bracket_order(
        self,
        symbol: str,
        side: str,          # "buy" o "sell"
        qty: int,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[object]:
        """Orden bracket: entrada + stop-loss + take-profit en una sola orden."""
        try:
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)),
                take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
            )
            order = self._client.submit_order(request)
            logger.info(
                f"Orden BRACKET enviada: {side.upper()} {qty} {symbol} "
                f"SL={stop_loss:.2f} TP={take_profit:.2f} | ID={order.id}"
            )
            return order
        except Exception as e:
            logger.error(f"Error enviando orden {symbol}: {e}")
            return None

    def submit_stop_order(
        self,
        symbol: str,
        side: str,      # "buy" (para cubrir short) | "sell" (para proteger long)
        qty: int,
        stop_price: float,
    ) -> Optional[object]:
        """Stop-market GTC para proteger una posición ya abierta."""
        try:
            request = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                stop_price=round(stop_price, 2),
            )
            order = self._client.submit_order(request)
            logger.info(
                f"Stop GTC: {side.upper()} {qty} {symbol} "
                f"stop=${stop_price:.2f} | ID={order.id}"
            )
            return order
        except Exception as e:
            # «insufficient qty / held_for_orders» (código 40310000) es benigno:
            # las acciones ya están retenidas por otra orden. Se registra a nivel
            # DEBUG para no inundar el log de ERROR (el protector ya filtra estos
            # casos, esto es solo una salvaguarda ante carreras).
            msg = str(e)
            if "40310000" in msg or "insufficient qty" in msg:
                logger.debug(f"Stop omitido para {symbol}: acciones ya retenidas.")
            else:
                logger.error(f"Error enviando stop order {symbol}: {e}")
            return None

    def submit_trailing_stop_order(
        self, symbol: str, side: str, qty: int, trail_percent: float
    ) -> Optional[object]:
        """Trailing stop GTC: Alpaca traila el stop en servidor según trail_percent.

        Protege la posición dejando correr al ganador (el stop sigue al precio a
        `trail_percent` de distancia) sin cancelar/recolocar a mano en bucle.
        """
        try:
            request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                trail_percent=round(trail_percent, 2),
            )
            order = self._client.submit_order(request)
            logger.info(
                f"Trailing stop: {side.upper()} {qty} {symbol} "
                f"trail={trail_percent:.2f}% | ID={order.id}"
            )
            return order
        except Exception as e:
            logger.error(f"Error enviando trailing stop {symbol}: {e}")
            return None

    def cancel_orders_for_symbol(self, symbol: str) -> int:
        """Cancela las órdenes abiertas de un símbolo. Devuelve cuántas canceló."""
        n = 0
        for o in self.get_open_orders([symbol]):
            try:
                self._client.cancel_order_by_id(o.id)
                n += 1
            except Exception as e:
                logger.warning(f"No se pudo cancelar orden de {symbol}: {e}")
        return n

    def cancel_orphan_trailing_stops(self, position_symbols) -> int:
        """Cancela trailing stops de símbolos SIN posición (protecciones huérfanas).

        Cuando una posición se cierra, su trailing stop puede quedar suelto; al ser
        una orden a mercado, podría dispararse y ABRIR una posición no deseada.
        """
        n = 0
        for o in self.get_open_orders():
            ot = str(getattr(o, "order_type", None) or getattr(o, "type", "")).lower()
            sym = getattr(o, "symbol", None)
            if "trailing" in ot and sym and sym not in position_symbols:
                try:
                    self._client.cancel_order_by_id(o.id)
                    logger.warning(f"Trailing stop huérfano cancelado: {sym} (sin posición)")
                    n += 1
                except Exception as e:
                    logger.warning(f"No se pudo cancelar trailing huérfano {sym}: {e}")
        return n

    def submit_market_order(self, symbol: str, side: str, qty: int) -> Optional[object]:
        """Orden de mercado simple (para cierre de posiciones)."""
        try:
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(request)
            logger.info(f"Orden MERCADO: {side.upper()} {qty} {symbol} | ID={order.id}")
            return order
        except Exception as e:
            logger.error(f"Error enviando orden mercado {symbol}: {e}")
            return None

    def close_position(self, symbol: str) -> bool:
        try:
            self._client.close_position(symbol)
            logger.info(f"Posición cerrada: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error cerrando posición {symbol}: {e}")
            return False

    def cancel_all_orders(self) -> None:
        try:
            self._client.cancel_orders()
            logger.info("Todas las órdenes canceladas")
        except Exception as e:
            logger.error(f"Error cancelando órdenes: {e}")
