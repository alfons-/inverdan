"""Stream de datos: WebSocket con fallback automático a REST polling."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ..config.settings import Settings
from ..utils.logger import get_logger
from .buffer import BufferRegistry, OHLCVBar

logger = get_logger("data.stream")

# Número de fallos WebSocket consecutivos antes de activar el polling REST
_WS_FAIL_THRESHOLD = 3
# Segundos entre polls REST
_POLL_INTERVAL = 60
# Cada cuántos polls intentar reconectar por WebSocket
_WS_RETRY_AFTER_POLLS = 10


class MarketStream:
    """
    Gestiona datos de mercado en tiempo real.

    Modo primario : WebSocket de Alpaca (latencia ~ms).
    Fallback       : REST polling cada 60s si el WebSocket falla 3 veces
                     seguidas (connection limit, rate limit, etc.).
    Reconexión     : cada 10 polls (~10 min) intenta volver al WebSocket.
    """

    def __init__(
        self,
        settings: Settings,
        buffer_registry: BufferRegistry,
        on_bar: Optional[Callable] = None,
    ):
        self._cfg = settings
        self._buffers = buffer_registry
        self._on_bar = on_bar

        self._stream: Optional[StockDataStream] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

        self._ws_fail_count: int = 0
        self._ws_reconnect_delay: float = 120.0
        self._polling: bool = False

        # REST client (sin rate limit por conexión persistente)
        self._rest_client = StockHistoricalDataClient(
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.api_secret,
        )
        # Último timestamp procesado por símbolo (evita barras duplicadas)
        self._last_bar_ts: Dict[str, pd.Timestamp] = {}

    # ── WebSocket ─────────────────────────────────────────────────────────────

    def _create_stream(self) -> StockDataStream:
        return StockDataStream(
            api_key=self._cfg.alpaca.api_key,
            secret_key=self._cfg.alpaca.api_secret,
            feed=self._cfg.alpaca.data_feed,
        )

    async def _bar_handler(self, bar) -> None:
        try:
            ohlcv = OHLCVBar(
                timestamp=bar.timestamp,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=int(bar.volume),
            )
            buf = self._buffers.get_or_create(bar.symbol)
            buf.update(ohlcv)
            if self._on_bar:
                self._on_bar(bar.symbol, ohlcv)
            logger.debug(f"[WS] Bar {bar.symbol}: {ohlcv.close:.2f} vol={ohlcv.volume:,}")
        except Exception as e:
            logger.error(f"Error procesando bar WS {bar.symbol}: {e}")

    def _force_stop(self, stream) -> None:
        """Para el loop interno de alpaca-py desde otro hilo (thread-safe)."""
        try:
            stream._should_run = False
            if stream._stop_stream_queue.empty():
                stream._stop_stream_queue.put_nowait({"should_stop": True})
        except Exception:
            pass

    def _try_websocket(self) -> bool:
        """
        Intenta conectar por WebSocket durante 20s.
        Retorna True si conectó, False si falló.
        """
        stream = self._create_stream()
        self._stream = stream
        stream.subscribe_bars(self._bar_handler, *self._cfg.symbols)

        logger.info(f"[WS] Conectando... símbolos: {self._cfg.symbols}")
        inner = threading.Thread(target=stream.run, daemon=True, name="alpaca-ws-inner")
        inner.start()

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and inner.is_alive() and not stream._running:
            time.sleep(0.5)

        if inner.is_alive() and not stream._running:
            # Falló la autenticación (connection limit, 429, etc.)
            self._force_stop(stream)
            inner.join(timeout=5)
            return False

        if inner.is_alive():
            # Conectado — esperar hasta desconexión natural
            logger.info("[WS] Conectado correctamente. Recibiendo barras...")
            self._ws_fail_count = 0
            self._ws_reconnect_delay = 120.0
            inner.join()
            elapsed_msg = "Stream WS desconectado"
            logger.warning(f"{elapsed_msg}. Reintentando...")
            return True  # Estuvo conectado (se desconectó después)

        # Hilo terminó solo antes de los 20s
        self._force_stop(stream)
        return False

    # ── REST polling ──────────────────────────────────────────────────────────

    def _poll_once(self) -> None:
        """Pide la última barra cerrada para todos los símbolos via REST."""
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(minutes=5)

            request = StockBarsRequest(
                symbol_or_symbols=self._cfg.symbols,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=start,
                end=end,
                feed=self._cfg.alpaca.data_feed,
            )
            bars_response = self._rest_client.get_stock_bars(request)
            df = bars_response.df

            if df.empty:
                return

            for symbol in self._cfg.symbols:
                try:
                    # Extraer barras del símbolo
                    if isinstance(df.index, pd.MultiIndex):
                        if symbol not in df.index.get_level_values(0):
                            continue
                        sym_df = df.loc[symbol]
                    else:
                        sym_df = df

                    if sym_df.empty:
                        continue

                    ts = sym_df.index[-1]
                    last = sym_df.iloc[-1]

                    # Saltar si ya procesamos esta barra
                    prev = self._last_bar_ts.get(symbol)
                    if prev is not None and ts <= prev:
                        continue
                    self._last_bar_ts[symbol] = ts

                    ohlcv = OHLCVBar(
                        timestamp=ts.to_pydatetime()
                            if hasattr(ts, "to_pydatetime") else ts,
                        open=float(last["open"]),
                        high=float(last["high"]),
                        low=float(last["low"]),
                        close=float(last["close"]),
                        volume=int(last["volume"]),
                    )
                    buf = self._buffers.get_or_create(symbol)
                    buf.update(ohlcv)
                    if self._on_bar:
                        self._on_bar(symbol, ohlcv)
                    logger.debug(f"[REST] Bar {symbol}: {ohlcv.close:.2f}")

                except Exception as e:
                    logger.error(f"[REST] Error procesando {symbol}: {e}")

        except Exception as e:
            logger.warning(f"[REST] Error en poll: {e}")

    def _run_polling(self) -> None:
        """Loop de polling REST con reintentos periódicos del WebSocket."""
        poll_count = 0
        logger.info(
            f"[REST] Modo polling activo — una barra cada {_POLL_INTERVAL}s. "
            f"Reintentará WebSocket cada {_WS_RETRY_AFTER_POLLS} polls."
        )

        while self._running.is_set():
            # Intentar reconectar por WebSocket cada N polls
            if poll_count % _WS_RETRY_AFTER_POLLS == 0 and poll_count > 0:
                logger.info("[WS] Reintentando conexión WebSocket desde modo polling...")
                connected = self._try_websocket()
                if connected:
                    # WebSocket reconectado y luego caído — volver al polling
                    self._ws_fail_count = 0
                    logger.info("[REST] WebSocket caído de nuevo, volviendo a polling.")
                else:
                    self._ws_fail_count += 1
                    logger.info(
                        f"[WS] Sigue sin conectar (intento {self._ws_fail_count}). "
                        f"Continuando en modo REST."
                    )
                    if not self._running.is_set():
                        break

            self._poll_once()
            poll_count += 1

            # Esperar hasta el próximo poll
            for _ in range(_POLL_INTERVAL):
                if not self._running.is_set():
                    return
                time.sleep(1)

    # ── Loop principal ────────────────────────────────────────────────────────

    def _run_stream(self) -> None:
        while self._running.is_set():
            if self._ws_fail_count >= _WS_FAIL_THRESHOLD:
                # Demasiados fallos seguidos → activar polling
                if not self._polling:
                    self._polling = True
                    logger.warning(
                        f"[WS] {_WS_FAIL_THRESHOLD} fallos consecutivos. "
                        f"Activando modo REST polling."
                    )
                self._run_polling()
                # Si _run_polling retorna, es porque _running se limpió
                break

            # Intentar WebSocket
            ok = self._try_websocket()
            if not ok:
                self._ws_fail_count += 1
                delay = min(self._ws_reconnect_delay, 300.0)
                logger.warning(
                    f"[WS] Fallo {self._ws_fail_count}/{_WS_FAIL_THRESHOLD}. "
                    f"Esperando {delay:.0f}s..."
                )
                self._ws_reconnect_delay = min(self._ws_reconnect_delay * 2, 300.0)
                for _ in range(int(delay)):
                    if not self._running.is_set():
                        return
                    time.sleep(1)
            else:
                # Estuvo conectado pero se cayó: resetear contadores
                self._ws_fail_count = 0
                self._ws_reconnect_delay = 5.0
                self._polling = False

    # ── API pública ───────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(
            target=self._run_stream, daemon=True, name="market-stream"
        )
        self._thread.start()
        logger.info("MarketStream iniciado (WS con fallback REST)")

    def stop(self) -> None:
        self._running.clear()
        if self._stream:
            try:
                self._force_stop(self._stream)
            except Exception:
                pass
        logger.info("MarketStream detenido")
