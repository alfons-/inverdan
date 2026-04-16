"""Wrapper del cliente de trading de Alpaca."""
from __future__ import annotations

from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
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
