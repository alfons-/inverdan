"""Gestión de riesgo: circuit breakers, sizing de posición, stops."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from ..config.settings import Settings
from ..signals.signal_types import Signal
from ..utils.logger import get_logger

logger = get_logger("execution.risk")


@dataclass
class _OpenPosition:
    """Estado interno de una posición para contabilizar P&L correctamente.

    Guarda el lado y la cantidad además del precio de entrada: sin el lado no se
    puede calcular el P&L de un corto (cerrar un corto es un BUY, no un SELL), y
    sin la cantidad no se manejan los fills parciales.
    """
    entry: float
    side: str   # "long" | "short"
    qty: int
    realized: float = 0.0   # P&L acumulado de los fills parciales del cierre


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
        self._open_positions: Dict[str, _OpenPosition] = {}  # symbol -> posición
        self._open_order_symbols: set = set()  # símbolos con órdenes ABIERTAS (sin rellenar) en el broker
        self._total_exposure: float = 0.0
        self._circuit_open: bool = False
        self._last_sync_signature: Optional[tuple] = None  # evita logs repetidos en cada sync
        # Cooldown por símbolo: racha de cierres perdedores y bloqueo temporal.
        # {symbol: (racha, fecha_última_pérdida)} y {symbol: bloqueado_hasta}.
        # Corta el patrón whipsaw de reentrar una y otra vez en el mismo valor
        # (NVDA jul-2026: 3 cortos seguidos parados en un rally = -453).
        self._symbol_loss_streak: Dict[str, tuple] = {}
        self._symbol_cooldown_until: Dict[str, datetime] = {}

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

            # Cooldown por símbolo (ambos lados): tras N pérdidas consecutivas en
            # el mismo valor, no se opera durante unos días. El whipsaw pierde en
            # las dos direcciones, así que el bloqueo no distingue largo/corto.
            until = self._symbol_cooldown_until.get(signal.symbol)
            if until is not None:
                if datetime.now(timezone.utc) < until:
                    streak = self._symbol_loss_streak.get(signal.symbol, (0, None))[0]
                    return False, (
                        f"Cooldown {signal.symbol}: {streak} pérdidas seguidas "
                        f"(hasta {until:%d-%b %H:%M} UTC)"
                    )
                self._symbol_cooldown_until.pop(signal.symbol, None)  # expirado

            # Precio mínimo
            if signal.price < cfg.min_stock_price:
                return False, f"Precio ${signal.price:.2f} < mínimo ${cfg.min_stock_price}"

            # Confianza mínima (ahora la confianza refleja calidad ADX/volumen, así
            # que este filtro veta de verdad los trades flojos)
            if signal.confidence < cfg.min_confidence:
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

            # No duplicar posición existente (BUY ni SELL — Alpaca no acepta
            # bracket orders sobre posiciones ya abiertas)
            if signal.symbol in self._open_positions:
                return False, f"Ya hay posición abierta en {signal.symbol}"

            # Tampoco operar si ya hay una orden ABIERTA (sin rellenar) en el
            # símbolo: Alpaca rechazaría la nueva orden (p. ej. «cannot open a
            # short sell while a long buy order is open»). Lo cortamos aquí para
            # evitar ese rechazo del broker y el ruido de notificaciones.
            if signal.symbol in self._open_order_symbols:
                return False, f"Orden abierta en {signal.symbol}"

            # Exposición máxima total
            if self._total_exposure >= cfg.max_total_exposure * portfolio_value:
                return False, f"Exposición máxima alcanzada ({self._total_exposure/portfolio_value:.1%})"

            return True, ""

    def size_position(
        self,
        signal: Signal,
        portfolio_value: float,
        atr: float,
        buying_power: float = 0.0,
    ) -> int:
        """
        Calcula el número de acciones usando ATR-based position sizing.

        El tamaño se acota, en este orden, por:
          1. Riesgo por operación (ATR): max_position_pct / (2*ATR).
          2. % máximo del portfolio por posición (max_position_pct).
          3. Presupuesto de exposición restante (capital propio, SIN margen).
          4. Buying power real de Alpaca (salvaguarda anti-rechazo del bróker).

        El parámetro buying_power es solo la salvaguarda (4): NO representa
        capital sin deuda, porque incluye el margen que presta el bróker.
        """
        if atr <= 0 or signal.price <= 0:
            return 0

        max_risk = portfolio_value * self._cfg.max_position_pct
        risk_per_share = 2.0 * atr  # Stop-loss a 2*ATR
        shares = int(max_risk / risk_per_share)

        # Límite por % de portfolio por posición
        max_shares_by_value = int((portfolio_value * self._cfg.max_position_pct) / signal.price)
        shares = min(shares, max_shares_by_value)

        # Límite por presupuesto de exposición restante (capital propio, sin
        # margen). Como max_total_exposure ≤ 1.0 (forzado por Pydantic), este
        # tope nunca supera el equity → es estructuralmente imposible endeudarse.
        with self._lock:
            current_exposure = self._total_exposure
        remaining_budget = self._cfg.max_total_exposure * portfolio_value - current_exposure
        max_shares_by_budget = int(max(remaining_budget, 0.0) / signal.price)
        shares = min(shares, max_shares_by_budget)

        # Salvaguarda secundaria: no exceder el buying power real de Alpaca para
        # evitar rechazos del bróker. NO es un límite «sin deuda»: lo incluye.
        if buying_power > 0:
            max_shares_by_bp = int(buying_power / signal.price)
            shares = min(shares, max_shares_by_bp)

        # max(0, …): si no queda presupuesto devolvemos 0 y el executor descarta
        # la orden (qty <= 0). Nunca forzamos una compra que requiera margen.
        return max(0, shares)

    def compute_stops(
        self, entry_price: float, atr: float, action: str
    ) -> tuple[float, float]:
        """Calcula stop-loss y take-profit basados en ATR.

        El ATR proviene de barras de 1 minuto y puede ser muy pequeño comparado
        con la volatilidad real diaria. Se aplica un mínimo del 1 % del precio
        de entrada para evitar stops que se disparen de inmediato por el spread
        o el ruido normal intradía.
        """
        min_dist = entry_price * self._cfg.min_stop_pct   # suelo configurable
        sl_dist = max(atr * self._cfg.stop_loss_atr_multiplier, min_dist)
        tp_dist = max(atr * self._cfg.take_profit_atr_multiplier, min_dist * 2)

        if action == "BUY":
            stop_loss = entry_price - sl_dist
            take_profit = entry_price + tp_dist
        else:  # SELL (short)
            stop_loss = entry_price + sl_dist
            take_profit = entry_price - tp_dist

        return round(stop_loss, 2), round(take_profit, 2)

    def sync_from_broker(self, broker) -> None:
        """
        Inicializa el estado interno (_open_positions, _open_order_symbols y
        _total_exposure) a partir del estado real del broker. Sin esto, tras un
        reinicio el bot creía no tener posiciones/órdenes abiertas y podía
        duplicar señales o intentar abrir el lado contrario de una orden viva
        (que Alpaca rechaza con «Orden rechazada por broker»).
        """
        try:
            positions = broker.get_positions()
        except Exception as e:
            logger.warning(f"sync_from_broker: no se pudieron leer posiciones: {e}")
            return

        # Símbolos con órdenes ABIERTAS (sin rellenar). Se rastrean aparte de las
        # posiciones: una orden bracket pendiente todavía no es una posición, pero
        # Alpaca rechaza abrir el lado contrario mientras la orden siga viva.
        open_order_symbols: set = set()
        try:
            for o in broker.get_open_orders():
                for obj in (o, *(getattr(o, "legs", None) or [])):
                    sym = getattr(obj, "symbol", None)
                    if sym:
                        open_order_symbols.add(sym)
        except (AttributeError, TypeError):
            pass  # broker sin get_open_orders (tests/mocks)
        except Exception as e:
            logger.warning(f"sync_from_broker: no se pudieron leer órdenes abiertas: {e}")

        with self._lock:
            self._open_positions.clear()
            self._open_order_symbols = open_order_symbols
            self._total_exposure = 0.0
            count = 0
            for pos in positions:
                try:
                    symbol = pos.symbol
                    qty_raw = float(pos.qty)
                    qty = abs(int(qty_raw))
                    entry = float(pos.avg_entry_price)
                    side = "long" if qty_raw > 0 else "short"
                    self._open_positions[symbol] = _OpenPosition(entry=entry, side=side, qty=qty)
                    self._total_exposure += entry * qty
                    count += 1
                except Exception:
                    continue
            # Loguear solo cuando el estado cambia (evita spam cada 30 s)
            signature = (count, round(self._total_exposure, 2))
            if count and signature != self._last_sync_signature:
                logger.info(
                    f"RiskManager sincronizado con broker: {count} posiciones, "
                    f"exposición total ${self._total_exposure:,.2f}"
                )
            self._last_sync_signature = signature

    def record_fill(
        self, symbol: str, side: str, price: float, shares: int
    ) -> Optional[float]:
        """
        Registra un fill ejecutado y devuelve el P&L realizado si cierra (total o
        parcialmente) una posición, o None si abre/amplía.

        `side` es el lado de la ORDEN ("buy"/"sell"). Apertura vs cierre se decide
        según la posición existente, no según el lado:
          - LARGO:  abre=buy,  cierra=sell   → P&L = (salida − entrada)·qty
          - CORTO:  abre=sell, cierra=buy     → P&L = (entrada − salida)·qty
        Maneja fills parciales reduciendo la cantidad de la posición.
        """
        with self._lock:
            existing = self._open_positions.get(symbol)

            # ── Sin posición previa → apertura ───────────────────────────────
            if existing is None:
                pos_side = "long" if side == "buy" else "short"
                self._open_positions[symbol] = _OpenPosition(
                    entry=price, side=pos_side, qty=shares
                )
                self._total_exposure += price * shares
                return None

            closes = (
                (existing.side == "long" and side == "sell")
                or (existing.side == "short" and side == "buy")
            )

            # ── Mismo sentido → amplía posición (media ponderada) ────────────
            if not closes:
                total_qty = existing.qty + shares
                existing.entry = (
                    (existing.entry * existing.qty + price * shares) / total_qty
                    if total_qty else price
                )
                existing.qty = total_qty
                self._total_exposure += price * shares
                return None

            # ── Sentido contrario → cierre (total o parcial) ─────────────────
            close_qty = min(shares, existing.qty)
            if existing.side == "long":
                pnl = (price - existing.entry) * close_qty
            else:  # short
                pnl = (existing.entry - price) * close_qty

            self._total_exposure -= existing.entry * close_qty
            self._daily_pnl += pnl
            existing.realized += pnl     # acumula el P&L de los fills parciales
            existing.qty -= close_qty

            # La racha de pérdidas consecutivas solo se evalúa cuando la posición
            # se cierra POR COMPLETO, sobre el P&L total acumulado. Así un cierre
            # en varios fills parciales cuenta como UNA operación, no como varias
            # (evita que el circuit breaker salte antes de tiempo).
            if existing.qty <= 0:
                if existing.realized < 0:
                    self._consecutive_losses += 1
                    self._update_symbol_streak(symbol)
                else:
                    self._consecutive_losses = 0
                    # Un cierre ganador limpia la racha y cualquier cooldown del símbolo
                    self._symbol_loss_streak.pop(symbol, None)
                    self._symbol_cooldown_until.pop(symbol, None)
                self._open_positions.pop(symbol, None)

            logger.info(
                f"Trade cerrado {symbol} ({existing.side}, {close_qty} acc): "
                f"PnL=${pnl:+.2f} | PnL día: ${self._daily_pnl:+.2f} | "
                f"Pérdidas consecutivas: {self._consecutive_losses}"
            )
            return round(pnl, 4)

    def _update_symbol_streak(self, symbol: str) -> None:
        """Suma una pérdida a la racha del símbolo y activa el cooldown si toca.
        Llamar con el lock cogido. Dos pérdidas cuentan como consecutivas si la
        anterior ocurrió dentro de la ventana (symbol_cooldown_days)."""
        threshold = getattr(self._cfg, "symbol_cooldown_losses", 0)
        if threshold <= 0:
            return
        window = timedelta(days=getattr(self._cfg, "symbol_cooldown_days", 5.0))
        now = datetime.now(timezone.utc)
        streak, last = self._symbol_loss_streak.get(symbol, (0, None))
        streak = streak + 1 if (last is not None and now - last <= window) else 1
        self._symbol_loss_streak[symbol] = (streak, now)
        if streak >= threshold:
            until = now + window
            self._symbol_cooldown_until[symbol] = until
            logger.warning(
                f"Cooldown activado en {symbol}: {streak} pérdidas consecutivas → "
                f"sin operar hasta {until:%Y-%m-%d %H:%M} UTC"
            )

    def active_cooldowns(self) -> Dict[str, str]:
        """{símbolo: hasta-cuándo (ISO)} de los cooldowns vigentes (para el dashboard)."""
        now = datetime.now(timezone.utc)
        with self._lock:
            expired = [s for s, u in self._symbol_cooldown_until.items() if u <= now]
            for s in expired:
                self._symbol_cooldown_until.pop(s, None)
            return {s: u.isoformat(timespec="minutes")
                    for s, u in self._symbol_cooldown_until.items()}

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

    def has_open_position(self, symbol: str) -> bool:
        with self._lock:
            return symbol in self._open_positions

    @property
    def open_positions_count(self) -> int:
        return len(self._open_positions)
