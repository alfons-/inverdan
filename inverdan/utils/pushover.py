"""Notificaciones Pushover para eventos clave de Inverdan.

No se envían avisos con título «Cambio de tendencia»; solo señales BUY/SELL
(respetando min_signal_confidence), fills y rechazos de orden relevantes.
"""
from __future__ import annotations

import threading
import requests

from inverdan.events.bus import EventBus, SignalEvent, OrderFilledEvent, OrderRejectedEvent
from inverdan.utils.logger import get_logger

_PUSHOVER_URL = "https://api.pushover.net/1/messages.json"

logger = get_logger("pushover")


class PushoverNotifier:
    """Suscribe al EventBus y envía notificaciones push via Pushover."""

    def __init__(self, api_token: str, user_key: str, event_bus: EventBus,
                 min_signal_confidence: float = 0.0):
        self._token = api_token
        self._user = user_key
        self._min_confidence = min_signal_confidence

        event_bus.subscribe(SignalEvent, self._on_signal)
        event_bus.subscribe(OrderFilledEvent, self._on_order_filled)
        event_bus.subscribe(OrderRejectedEvent, self._on_order_rejected)

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _on_signal(self, event: SignalEvent) -> None:
        if event.confidence < self._min_confidence:
            return
        action_icon = "📈" if event.action == "BUY" else "📉"
        self._send(
            title=f"{action_icon} Señal {event.action} — {event.symbol}",
            message=(
                f"Precio: ${event.price:.2f}\n"
                f"Confianza: {event.confidence:.1%}\n"
                f"{event.reasoning}"
            ),
            priority=0,
        )

    def _on_order_filled(self, event: OrderFilledEvent) -> None:
        is_buy = event.side.upper() == "BUY"
        # Los cierres automáticos (stop/TP de Alpaca) llegan con stop_price=0
        auto_close = not is_buy and event.stop_price == 0.0

        if is_buy:
            title = f"🟢 Posición abierta — {event.symbol}"
            message = (
                f"Compra {event.shares} acciones @ ${event.fill_price:.2f}\n"
                f"Stop: ${event.stop_price:.2f}  |  TP: ${event.take_profit_price:.2f}"
            )
        elif auto_close:
            title = f"🔔 Cierre automático — {event.symbol}"
            message = (
                f"Venta {event.shares} acciones @ ${event.fill_price:.2f}\n"
                f"(Stop-loss o Take-profit alcanzado)"
            )
        else:
            title = f"🔴 Operación ejecutada — {event.symbol}"
            message = (
                f"Venta {event.shares} acciones @ ${event.fill_price:.2f}\n"
                f"Stop: ${event.stop_price:.2f}  |  TP: ${event.take_profit_price:.2f}"
            )

        self._send(title=title, message=message, priority=1)

    # Rechazos rutinarios que no necesitan notificación
    _SILENT_REJECTIONS = (
        "Ya hay posición abierta",
        "Señal HOLD",
        "Confianza baja",
        "Precio $",
        "Tamaño de posición = 0",
    )

    def _on_order_rejected(self, event: OrderRejectedEvent) -> None:
        if any(event.reason.startswith(s) for s in self._SILENT_REJECTIONS):
            return
        self._send(
            title=f"⚠️ Orden rechazada — {event.symbol}",
            message=event.reason,
            priority=0,
        )

    # ── HTTP ──────────────────────────────────────────────────────────────────

    def _send(self, title: str, message: str, priority: int = 0) -> None:
        threading.Thread(
            target=self._post,
            args=(title, message, priority),
            daemon=True,
            name="pushover-send",
        ).start()

    def _post(self, title: str, message: str, priority: int) -> None:
        try:
            resp = requests.post(
                _PUSHOVER_URL,
                data={
                    "token": self._token,
                    "user": self._user,
                    "title": title,
                    "message": message,
                    "priority": priority,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning("Pushover respondió %s: %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.warning("Error enviando notificación Pushover: %s", exc)
