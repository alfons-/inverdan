"""Stream de datos en tiempo real via Alpaca WebSocket."""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Callable, List, Optional

from alpaca.data.live import StockDataStream

from ..config.settings import Settings
from ..utils.logger import get_logger
from .buffer import BufferRegistry, OHLCVBar

logger = get_logger("data.stream")


class MarketStream:
    """Gestiona el WebSocket de Alpaca para datos en tiempo real."""

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
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = threading.Event()
        self._reconnect_delay = 1.0

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

            logger.debug(f"Bar {bar.symbol}: {ohlcv.close:.2f} vol={ohlcv.volume:,}")
        except Exception as e:
            logger.error(f"Error procesando bar {bar.symbol}: {e}")

    def _run_stream(self) -> None:
        while self._running.is_set():
            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)

                self._stream = self._create_stream()
                self._stream.subscribe_bars(self._bar_handler, *self._cfg.symbols)

                logger.info(f"Conectado al stream. Símbolos: {self._cfg.symbols}")
                self._reconnect_delay = 1.0
                self._stream.run()

            except Exception as e:
                if not self._running.is_set():
                    break
                logger.warning(f"Stream desconectado: {e}. Reconectando en {self._reconnect_delay:.0f}s...")
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60.0)
            finally:
                if self._loop and not self._loop.is_closed():
                    self._loop.close()

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run_stream, daemon=True, name="market-stream")
        self._thread.start()
        logger.info("MarketStream iniciado")

    def stop(self) -> None:
        self._running.clear()
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
        logger.info("MarketStream detenido")
