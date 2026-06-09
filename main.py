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
import os
import signal
import sys
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Silenciar dos warnings benignos de sklearn que se re-emiten en CADA predicción
# del RandomForest (sklearn invalida el registro de warnings en cada predict).
# Con launchd sin rotación de stderr llegaron a acumular 4 GB en bot_stderr.log.
warnings.filterwarnings("ignore", message=r".*sklearn\.utils\.parallel\.delayed.*")
warnings.filterwarnings("ignore", message=r"X does not have valid feature names.*")

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
from inverdan.signals.trend_filter import DailyTrendProvider
from inverdan.execution.broker import AlpacaBroker
from inverdan.execution.risk import RiskManager
from inverdan.execution.portfolio import PortfolioTracker
from inverdan.execution.executor import TradeExecutor
from inverdan.execution.trade_stream import AlpacaTradeStream
from inverdan.dashboard.renderer import DashboardRenderer
from inverdan.dashboard.state import DashboardState
from inverdan.utils.logger import setup_logger, get_logger, TradeLogger
from inverdan.utils.market_hours import is_market_open, now_et
from inverdan.utils.pushover import PushoverNotifier


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

    # Evitar instancias múltiples: comprobar si ya hay un proceso corriendo
    pid_file = Path(__file__).parent / "bot.pid"
    if pid_file.exists():
        try:
            existing_pid = int(pid_file.read_text().strip())
            import psutil
            if psutil.pid_exists(existing_pid):
                proc = psutil.Process(existing_pid)
                cmdline = " ".join(proc.cmdline())
                is_bot = "main.py" in cmdline and proc.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
                if is_bot:
                    logger.error(f"Ya hay una instancia corriendo (PID {existing_pid}). Saliendo.")
                    sys.exit(1)
        except Exception:
            pass
        # PID obsoleto — limpiar
        pid_file.unlink(missing_ok=True)
    pid_file.write_text(str(os.getpid()))
    # Registrar limpieza inmediatamente para que funcione aunque el bot crashee
    import atexit
    atexit.register(lambda: pid_file.unlink(missing_ok=True))

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

    # Sincronizar primero el portfolio (rellena posiciones, equity, stops)
    portfolio_tracker.sync_from_broker(broker)

    # Y a continuación el RiskManager para que conozca las posiciones existentes:
    # sin esto, _open_positions y _total_exposure arrancan vacíos y se podrían
    # duplicar BUYs sobre símbolos ya en cartera o saltarse el límite de exposición.
    risk_manager.sync_from_broker(broker)

    initial_snap = portfolio_tracker.get_snapshot()
    if initial_snap.positions:
        logger.info(
            f"Portfolio inicial: {len(initial_snap.positions)} posiciones abiertas, "
            f"equity=${initial_snap.equity:,.2f}, "
            f"unrealized_pnl=${initial_snap.total_unrealized_pnl:+,.2f}"
        )
        for p in initial_snap.positions:
            logger.info(
                f"  - {p.symbol} {p.side.upper()} {p.qty}@${p.entry_price:.2f} "
                f"SL=${p.stop_loss:.2f} TP=${p.take_profit:.2f}"
            )

    # Modelos ML
    model_registry = ModelRegistry(settings.models_path)
    loaded = model_registry.load_all(settings.symbols)
    if loaded == 0:
        logger.warning(
            "No se encontraron modelos ML. Ejecuta primero: python train.py\n"
            "El sistema funcionará solo con señales de reglas técnicas."
        )

    # Cliente histórico (warm-up de indicadores + tendencia mayor diaria)
    hist_client = HistoricalDataClient(settings)

    # Filtro de tendencia mayor (p. ej. SMA50 diaria). Se calcula al arrancar para
    # que las primeras señales ya respeten la tendencia del timeframe superior.
    trend_provider = DailyTrendProvider(settings, hist_client)
    logger.info(
        f"Calculando tendencia mayor ({settings.risk.trend_timeframe} "
        f"SMA{settings.risk.trend_sma_period})..."
    )
    trend_provider.refresh(settings.symbols)

    # Señales (con filtro de tendencia mayor)
    aggregator = SignalAggregator(settings, model_registry, trend_provider=trend_provider)

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

    # Logger de trades y señales
    trade_logger = TradeLogger(settings.logs_path)

    # Stream de trade updates (registra PnL de cierres automáticos SL/TP)
    trade_stream = AlpacaTradeStream(
        settings,
        risk_manager,
        trade_logger,
        portfolio_tracker,
        event_bus,
    )

    # Último snapshot de indicadores por símbolo (para el dashboard)
    last_snaps: dict = {}

    # ── Callback: nueva barra de datos ──────────────────────────────────────
    def on_bar(symbol: str, bar) -> None:
        buf = buffer_registry.get_or_create(symbol)
        df = buf.get_dataframe()

        if len(df) < settings.ml.min_bars_required:
            return

        # Calcular indicadores
        snap = indicator_calculator.compute(df)

        # Guardar snapshot para el dashboard de mercado
        try:
            ts = df.index[-1]
            last_snaps[symbol] = {
                "close":        round(snap.close, 2),
                "volume":       int(snap.volume),
                "vwap":         round(snap.vwap, 2),
                "rsi":          round(snap.rsi, 1),
                "macd":         round(snap.macd, 4),
                "macd_signal":  round(snap.macd_signal, 4),
                "macd_hist":    round(snap.macd_hist, 4),
                "bb_upper":     round(snap.bb_upper, 2),
                "bb_middle":    round(snap.bb_middle, 2),
                "bb_lower":     round(snap.bb_lower, 2),
                "bb_pct":       round(snap.bb_pct, 3),
                "atr":          round(snap.atr, 2),
                "adx":          round(snap.adx, 1),
                "ema_fast":     round(snap.ema_fast, 2),
                "ema_slow":     round(snap.ema_slow, 2),
                "sma_200":      round(snap.sma_200, 2),
                "volume_ratio": round(snap.volume_ratio, 2),
                "stoch_k":      round(snap.stoch_k, 1),
                "stoch_d":      round(snap.stoch_d, 1),
                "bars":         len(df),
                "updated_at":   ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            }
        except Exception:
            pass

        # Actualizar precio en portfolio
        portfolio_tracker.update_price(symbol, snap.close)
        dash_state.update_price(symbol, snap.close)

        # Generar señal
        signal = aggregator.evaluate(symbol, snap, timestamp=df.index[-1])

        # Publicar en dashboard
        dash_state.add_signal(signal)

        # Solo publicar al bus si no es HOLD (para el executor)
        if signal.action != "HOLD":
            trade_logger.log_signal(signal.to_dict())
            event_bus.post(SignalEvent(
                symbol=signal.symbol,
                action=signal.action,
                confidence=signal.confidence,
                price=signal.price,
                reasoning=signal.reasoning,
                timestamp=signal.timestamp,
                indicators=signal.indicators,
            ))

    # ── Pushover ─────────────────────────────────────────────────────────────
    if settings.pushover.enabled:
        PushoverNotifier(
            api_token=settings.pushover.api_token,
            user_key=settings.pushover.user_key,
            event_bus=event_bus,
            min_signal_confidence=settings.pushover.min_signal_confidence,
            device=settings.pushover.device,
        )
        logger.info(
            "Notificaciones Pushover activadas"
            + (f" (dispositivos: {settings.pushover.device})" if settings.pushover.device else " (todos los dispositivos)")
            + "."
        )

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
    trade_stream.start()

    # ── Bus de eventos (hilo dedicado) ───────────────────────────────────────
    bus_thread = threading.Thread(target=event_bus.dispatch_loop, daemon=True, name="event-bus")
    bus_thread.start()

    # ── Sincronización periódica del portfolio ───────────────────────────────
    def portfolio_sync_loop():
        while True:
            time.sleep(30)
            try:
                portfolio_tracker.sync_from_broker(broker)
                risk_manager.sync_from_broker(broker)   # mantiene _open_positions alineado con Alpaca
                snap = portfolio_tracker.get_snapshot()
                dash_state.update_portfolio(snap)
            except Exception:
                pass

    sync_thread = threading.Thread(target=portfolio_sync_loop, daemon=True, name="portfolio-sync")
    sync_thread.start()

    # ── Refresco periódico de la tendencia mayor (diaria) ────────────────────
    def trend_refresh_loop():
        while True:
            time.sleep(3 * 3600)   # las velas diarias cambian 1 vez/día; cada 3 h sobra
            try:
                trend_provider.refresh(settings.symbols)
            except Exception:
                pass

    threading.Thread(target=trend_refresh_loop, daemon=True, name="trend-refresh").start()

    # ── Protector de posiciones sin stop-loss ────────────────────────────────
    def _symbols_with_held_shares() -> set:
        """Símbolos cuyas acciones ya están retenidas por alguna orden abierta.

        Cualquier orden abierta (stop, take-profit limit, leg de bracket…) retiene
        las acciones de la posición. Mientras exista, Alpaca rechaza colocar otro
        stop con «insufficient qty / held_for_orders». Por eso, si hay CUALQUIER
        orden para el símbolo, el protector no debe intentar añadir otra: hacerlo
        solo genera errores en bucle (el bug que llenó el log de 657 ERROR).
        """
        held = set()
        for o in broker.get_open_orders():
            sym = getattr(o, "symbol", None)
            if sym:
                held.add(sym)
            for leg in (getattr(o, "legs", None) or []):
                leg_sym = getattr(leg, "symbol", None)
                if leg_sym:
                    held.add(leg_sym)
        return held

    def protect_positions_loop():
        """
        Garantiza que toda posición abierta tenga un stop-loss activo.

        En cada iteración consulta las órdenes reales de Alpaca (no un estado
        cacheado) para saber qué símbolos ya están protegidos, y coloca un
        stop-market GTC solo en los que no lo estén:
          - SHORT → stop en current_price * 1.015  (+1.5 %)
          - LONG  → stop en current_price * 0.985  (-1.5 %)
        """
        time.sleep(15)   # Dejar al bot arrancar completamente

        while True:
            try:
                snap = portfolio_tracker.get_snapshot()
                if snap.positions:
                    held = _symbols_with_held_shares()
                    for pos in snap.positions:
                        # Si las acciones ya están retenidas por cualquier orden
                        # (stop o TP), no se puede ni se debe añadir otro stop.
                        if pos.symbol in held:
                            continue

                        ref_price = pos.current_price or pos.entry_price
                        if pos.side == "short":
                            stop_price = round(ref_price * 1.015, 2)
                            side = "buy"
                        else:
                            stop_price = round(ref_price * 0.985, 2)
                            side = "sell"

                        logger.warning(
                            f"Posición {pos.symbol} ({pos.side} {pos.qty}) sin stop-loss. "
                            f"Colocando stop @ ${stop_price:.2f}"
                        )
                        broker.submit_stop_order(
                            symbol=pos.symbol,
                            side=side,
                            qty=pos.qty,
                            stop_price=stop_price,
                        )
            except Exception as e:
                logger.warning(f"protect_positions_loop error: {e}")

            time.sleep(60)   # Revisar cada minuto

    protect_thread = threading.Thread(target=protect_positions_loop, daemon=True, name="pos-protector")
    protect_thread.start()

    # ── Reseteo diario de contadores al abrir el mercado ─────────────────────
    def daily_reset_loop():
        """
        Pone a cero el «PnL del día», las pérdidas consecutivas y el circuit
        breaker al comienzo de cada nueva jornada bursátil.

        Sin esto, reset_daily() nunca se llamaba y el «PnL del día» en realidad
        acumulaba desde el último arranque del bot (engañoso si corría varios
        días seguidos). Se resetea cuando, en una fecha ET distinta a la última,
        el mercado está abierto (es decir, justo al abrir).
        """
        last_reset_date = now_et().date()   # no resetear nada más arrancar
        while True:
            time.sleep(60)
            try:
                today = now_et().date()
                if today != last_reset_date and is_market_open():
                    risk_manager.reset_daily()
                    portfolio_tracker.reset_daily()
                    last_reset_date = today
                    logger.info(f"Nueva jornada {today}: contadores diarios reseteados.")
            except Exception:
                pass

    daily_reset_thread = threading.Thread(target=daily_reset_loop, daemon=True, name="daily-reset")
    daily_reset_thread.start()

    # Sincronización inicial
    snap = portfolio_tracker.get_snapshot()
    dash_state.update_portfolio(snap)

    # ── Escritura periódica de state.json para el dashboard web ─────────────
    _state_file = Path(__file__).parent / "state.json"

    def write_state_loop():
        while True:
            try:
                snap = portfolio_tracker.get_snapshot()
                # Exposición bruta (valor de las posiciones) y capital propio libre:
                # lo que se puede invertir sin recurrir al margen (sin endeudarse).
                gross_exposure = sum(p.market_value for p in snap.positions)
                available_no_margin = snap.equity - gross_exposure
                state_data = {
                    "auto_trade": dash_state.auto_trade,
                    "portfolio": {
                        "equity": snap.equity,
                        "buying_power": snap.buying_power,
                        "available_no_margin": round(available_no_margin, 2),
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
                    "updated_at": datetime.now(timezone.utc).isoformat(),
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

                state_data["market"] = last_snaps
                # Escritura atómica (tmp + replace): el dashboard nunca lee un
                # JSON a medio escribir (antes provocaba lecturas vacías).
                _tmp_file = _state_file.with_suffix(".json.tmp")
                _tmp_file.write_text(json.dumps(state_data))
                os.replace(_tmp_file, _state_file)
            except Exception:
                pass
            time.sleep(3)

    state_thread = threading.Thread(target=write_state_loop, daemon=True, name="state-writer")
    state_thread.start()

    # ── Manejo de señales del sistema ────────────────────────────────────────
    def shutdown(sig, frame):
        logger.info("Apagando sistema...")
        stream.stop()
        trade_stream.stop()
        event_bus.shutdown()
        pid_file.unlink(missing_ok=True)
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
