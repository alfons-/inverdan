#!/usr/bin/env python3
"""
INVERDAN - Sistema Automático de Monitorización de Mercados de Inversión
========================================================================
Uso:
    python main.py                    # Modo monitorización (sin ejecución automática)
    python main.py --auto-trade       # Activa trading automático
    python main.py --symbols AAPL TSLA NVDA
    python main.py --no-dashboard     # Solo logs, sin interfaz gráfica
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from inverdan.config.settings import load_settings
from inverdan.data.buffer import BufferRegistry
from inverdan.data.stream import MarketStream
from inverdan.data.historical import HistoricalDataClient
from inverdan.events.bus import EventBus, BarReadyEvent, SignalEvent, OrderFilledEvent
from inverdan.indicators.calculator import IndicatorCalculator
from inverdan.ml.features import build_feature_vector
from inverdan.ml.registry import ModelRegistry
from inverdan.signals.aggregator import SignalAggregator
from inverdan.execution.broker import AlpacaBroker
from inverdan.execution.risk import RiskManager
from inverdan.execution.portfolio import PortfolioTracker
from inverdan.execution.executor import TradeExecutor
from inverdan.dashboard.renderer import DashboardRenderer
from inverdan.dashboard.state import DashboardState
from inverdan.utils.logger import setup_logger, get_logger
from inverdan.utils.market_hours import is_market_open


def parse_args():
    parser = argparse.ArgumentParser(description="INVERDAN - Monitor de Mercados")
    parser.add_argument("--config", default="config.yaml", help="Fichero de configuración")
    parser.add_argument("--symbols", nargs="+", help="Símbolos a monitorizar")
    parser.add_argument("--auto-trade", action="store_true", help="Activar trading automático")
    parser.add_argument("--no-dashboard", action="store_true", help="Sin dashboard visual")
    return parser.parse_args()


def main():
    args = parse_args()

    # Cargar configuración
    settings = load_settings(args.config)
    if args.symbols:
        settings.symbols = args.symbols

    # Configurar logging
    setup_logger(settings.logs_path)
    logger = get_logger("main")

    logger.info("=" * 60)
    logger.info("INVERDAN iniciando...")
    logger.info(f"Símbolos: {settings.symbols}")
    logger.info(f"Modo: {'PAPER' if settings.alpaca.paper_trading else '*** LIVE ***'}")
    logger.info(f"Auto-trade: {args.auto_trade}")

    if not is_market_open():
        logger.warning("El mercado está cerrado en este momento. El sistema esperará datos históricos.")

    # ── Componentes principales ──────────────────────────────────────────────
    event_bus = EventBus()
    buffer_registry = BufferRegistry(maxlen=settings.bar_buffer_size)
    indicator_calculator = IndicatorCalculator(settings)

    # Broker y portfolio
    broker = AlpacaBroker(settings)
    risk_manager = RiskManager(settings)
    portfolio_tracker = PortfolioTracker()
    portfolio_tracker.sync_from_broker(broker)

    # Modelos ML
    model_registry = ModelRegistry(settings.models_path)
    loaded = model_registry.load_all(settings.symbols)
    if loaded == 0:
        logger.warning(
            "No se encontraron modelos ML. Ejecuta primero: python train.py\n"
            "El sistema funcionará solo con señales de reglas técnicas."
        )

    # Señales
    aggregator = SignalAggregator(settings, model_registry)

    # Estado del dashboard
    dash_state = DashboardState(
        max_signals=settings.dashboard.signal_history_count,
        max_logs=settings.dashboard.max_log_lines,
    )

    # Ejecutor de operaciones
    executor = TradeExecutor(settings, broker, risk_manager, portfolio_tracker, event_bus)
    if not args.auto_trade:
        executor.pause()
        logger.info("Auto-trade DESACTIVADO. Use --auto-trade para activar.")

    # ── Callback: nueva barra de datos ──────────────────────────────────────
    def on_bar(symbol: str, bar) -> None:
        buf = buffer_registry.get_or_create(symbol)
        df = buf.get_dataframe()

        if len(df) < settings.ml.min_bars_required:
            return

        # Calcular indicadores
        snap = indicator_calculator.compute(df)

        # Actualizar precio en portfolio
        portfolio_tracker.update_price(symbol, snap.close)
        dash_state.update_price(symbol, snap.close)

        # Generar señal
        signal = aggregator.evaluate(symbol, snap, timestamp=df.index[-1])

        # Publicar en dashboard
        dash_state.add_signal(signal)

        # Solo publicar al bus si no es HOLD (para el executor)
        if signal.action != "HOLD":
            event_bus.post(SignalEvent(
                symbol=signal.symbol,
                action=signal.action,
                confidence=signal.confidence,
                price=signal.price,
                reasoning=signal.reasoning,
                timestamp=signal.timestamp,
                indicators=signal.indicators,
            ))

    # ── Suscriptores del bus ─────────────────────────────────────────────────
    def on_order_filled(event: OrderFilledEvent):
        msg = (
            f"FILL: {event.side.upper()} {event.shares} {event.symbol} "
            f"@ ${event.fill_price:.2f}"
        )
        dash_state.add_log(msg)
        portfolio_tracker.sync_from_broker(broker)

    event_bus.subscribe(OrderFilledEvent, on_order_filled)

    # ── Iniciar stream de mercado ────────────────────────────────────────────
    stream = MarketStream(settings, buffer_registry, on_bar=on_bar)

    # Precargar datos históricos para warm-up de indicadores
    logger.info("Precargando datos históricos para warm-up...")
    hist_client = HistoricalDataClient(settings)
    for sym in settings.symbols:
        try:
            df = hist_client.fetch_bars(sym, days=5, cache=True)
            if not df.empty:
                buf = buffer_registry.get_or_create(sym)
                for ts, row in df.iterrows():
                    from inverdan.data.buffer import OHLCVBar
                    ohlcv = OHLCVBar(
                        timestamp=ts.to_pydatetime(),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=int(row["volume"]),
                    )
                    buf.update(ohlcv)
                logger.info(f"Warm-up {sym}: {len(df)} barras cargadas")
        except Exception as e:
            logger.warning(f"No se pudo cargar histórico para {sym}: {e}")

    stream.start()

    # ── Bus de eventos (hilo dedicado) ───────────────────────────────────────
    bus_thread = threading.Thread(target=event_bus.dispatch_loop, daemon=True, name="event-bus")
    bus_thread.start()

    # ── Sincronización periódica del portfolio ───────────────────────────────
    def portfolio_sync_loop():
        while True:
            time.sleep(30)
            try:
                portfolio_tracker.sync_from_broker(broker)
                snap = portfolio_tracker.get_snapshot()
                dash_state.update_portfolio(snap)
            except Exception:
                pass

    sync_thread = threading.Thread(target=portfolio_sync_loop, daemon=True, name="portfolio-sync")
    sync_thread.start()

    # Sincronización inicial
    snap = portfolio_tracker.get_snapshot()
    dash_state.update_portfolio(snap)

    # ── Escritura periódica de state.json para el dashboard web ─────────────
    _state_file = Path(__file__).parent / "state.json"

    def write_state_loop():
        while True:
            try:
                snap = portfolio_tracker.get_snapshot()
                state_data = {
                    "auto_trade": dash_state.auto_trade,
                    "portfolio": {
                        "equity": snap.equity,
                        "buying_power": snap.buying_power,
                        "daily_pnl": round(snap.daily_pnl, 2),
                        "total_unrealized_pnl": round(snap.total_unrealized_pnl, 2),
                        "trades_today": snap.trades_today,
                        "wins_today": snap.wins_today,
                        "losses_today": snap.losses_today,
                    },
                    "positions": [
                        {
                            "symbol": p.symbol,
                            "side": p.side,
                            "qty": p.qty,
                            "entry_price": p.entry_price,
                            "current_price": p.current_price,
                            "stop_loss": p.stop_loss,
                            "take_profit": p.take_profit,
                            "unrealized_pnl": round(p.unrealized_pnl, 2),
                            "unrealized_pnl_pct": round(p.unrealized_pnl_pct, 4),
                        }
                        for p in snap.positions
                    ],
                    "risk": {
                        "circuit_open": risk_manager.circuit_open,
                        "daily_pnl": round(risk_manager.daily_pnl, 2),
                        "open_positions": risk_manager.open_positions_count,
                    },
                    "updated_at": datetime.utcnow().isoformat(),
                }
                # Leer auto_trade del fichero por si el dashboard lo cambió
                if _state_file.exists():
                    try:
                        existing = json.loads(_state_file.read_text())
                        if "auto_trade" in existing:
                            new_at = existing["auto_trade"]
                            if new_at != dash_state.auto_trade:
                                if new_at:
                                    executor.resume()
                                else:
                                    executor.pause()
                                dash_state._auto_trade = new_at
                        state_data["auto_trade"] = dash_state.auto_trade
                    except Exception:
                        pass

                _state_file.write_text(json.dumps(state_data))
            except Exception:
                pass
            time.sleep(3)

    state_thread = threading.Thread(target=write_state_loop, daemon=True, name="state-writer")
    state_thread.start()

    # ── Manejo de señales del sistema ────────────────────────────────────────
    def shutdown(sig, frame):
        logger.info("Apagando sistema...")
        stream.stop()
        event_bus.shutdown()
        logger.info("Sistema apagado correctamente.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Dashboard ────────────────────────────────────────────────────────────
    if args.auto_trade:
        dash_state._auto_trade = True

    if not args.no_dashboard:
        renderer = DashboardRenderer(settings, dash_state)
        try:
            renderer.run()  # Bloquea en el hilo principal
        except KeyboardInterrupt:
            shutdown(None, None)
    else:
        logger.info("Ejecutando en modo sin dashboard. Ctrl+C para salir.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            shutdown(None, None)


if __name__ == "__main__":
    main()
