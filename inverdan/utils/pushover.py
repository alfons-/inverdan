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
                 min_signal_confidence: float = 0.0, device: str = ""):
        self._token = api_token
        self._user = user_key
        self._min_confidence = min_signal_confidence
        # Dispositivos destino ("a,b"). Vacío = todos los de la cuenta.
        self._device = device.strip()

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

    # Traducción del motivo de cierre para el aviso
    _CLOSE_REASON_ES = {
        "stop_loss": "stop-loss",
        "take_profit": "take-profit",
        "trailing_stop": "trailing stop",
    }

    def _on_order_filled(self, event: OrderFilledEvent) -> None:
        # Distingue abrir/cerrar y largo/corto a partir de los campos del evento
        # (el side de la orden NO basta: cubrir un corto es un BUY).
        lado = "largo" if event.position_side == "long" else "corto"

        if event.is_close:
            pnl = event.pnl or 0.0
            icon = "🟢" if pnl >= 0 else "🔴"
            motivo = self._CLOSE_REASON_ES.get(event.close_reason, event.close_reason or "cierre")
            self._send(
                title=f"{icon} Cierra {lado} — {event.symbol}",
                message=(
                    f"Cierra {event.shares} @ ${event.fill_price:.2f}\n"
                    f"Resultado: {pnl:+.2f} $  ({motivo})"
                ),
                priority=1,
            )
        else:
            icon = "📈" if event.position_side == "long" else "📉"
            self._send(
                title=f"{icon} Abre {lado} — {event.symbol}",
                message=(
                    f"{event.shares} acciones @ ${event.fill_price:.2f}\n"
                    f"Stop: ${event.stop_price:.2f}  |  TP: ${event.take_profit_price:.2f}"
                ),
                priority=1,
            )

    # Rechazos rutinarios que no necesitan notificación
    _SILENT_REJECTIONS = (
        "Ya hay posición abierta",
        "Orden abierta",
        "Señal HOLD",
        "Confianza baja",
        "Precio $",
        "Tamaño de posición = 0",
    )

    def _on_order_rejected(self, event: OrderRejectedEvent) -> None:
        reason = event.reason
        # Veto de la capa LLM: aviso propio y con prioridad alta (es relevante).
        if reason.startswith("Veto LLM"):
            motivo = reason.split(":", 1)[1].strip() if ":" in reason else reason
            self._send(
                title=f"🚫 Veto LLM — {event.symbol}",
                message=f"La revisión LLM vetó esta operación.\n{motivo}",
                priority=1,
            )
            return
        if any(reason.startswith(s) for s in self._SILENT_REJECTIONS):
            return
        self._send(
            title=f"⚠️ Orden rechazada — {event.symbol}",
            message=reason,
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
            data = {
                "token": self._token,
                "user": self._user,
                # Prefijo de marca: en la cuenta compartida deja claro que el aviso
                # es de Inverdan (y no de otra app/usuario de la misma cuenta).
                "title": f"Inverdan · {title}",
                "message": message,
                "priority": priority,
            }
            if self._device:
                data["device"] = self._device
            resp = requests.post(_PUSHOVER_URL, data=data, timeout=10)
            if resp.status_code != 200:
                logger.warning("Pushover respondió %s: %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.warning("Error enviando notificación Pushover: %s", exc)
