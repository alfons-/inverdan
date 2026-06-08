"""Tests del gestor de riesgo."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock
from datetime import datetime

from inverdan.execution.risk import RiskManager
from inverdan.signals.signal_types import Signal


def make_settings():
    cfg = MagicMock()
    cfg.risk.max_position_pct = 0.05
    cfg.risk.max_total_exposure = 0.80
    cfg.risk.stop_loss_atr_multiplier = 2.0
    cfg.risk.take_profit_atr_multiplier = 4.0
    cfg.risk.max_daily_loss_pct = 0.05
    cfg.risk.max_consecutive_losses = 5
    cfg.risk.max_orders_per_minute = 3
    cfg.risk.min_stock_price = 5.0
    return cfg


def make_signal(action="BUY", price=150.0, confidence=0.75, symbol="AAPL"):
    return Signal(
        symbol=symbol, action=action, confidence=confidence,
        price=price, reasoning="test", indicators={"atr": 1.5}
    )


class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager(make_settings())

    def test_hold_signal_rejected(self):
        ok, reason = self.rm.approve(make_signal("HOLD"), 100_000)
        assert not ok
        assert "HOLD" in reason

    def test_low_confidence_rejected(self):
        ok, reason = self.rm.approve(make_signal(confidence=0.3), 100_000)
        assert not ok
        assert "onfianza" in reason.lower() or "confianza" in reason.lower()

    def test_low_price_rejected(self):
        ok, reason = self.rm.approve(make_signal(price=3.0), 100_000)
        assert not ok
        assert "mínimo" in reason.lower() or "precio" in reason.lower()

    def test_valid_signal_approved(self):
        ok, reason = self.rm.approve(make_signal(), 100_000)
        assert ok
        assert reason == ""

    def test_duplicate_position_rejected(self):
        self.rm.record_fill("AAPL", "buy", 150.0, 10)
        ok, reason = self.rm.approve(make_signal(), 100_000)
        assert not ok

    def test_open_order_blocks_new_signal(self):
        # Con una orden ABIERTA (sin rellenar) en el símbolo, approve la rechaza
        # antes de llegar al broker, evitando el rechazo de Alpaca
        # "cannot open a short sell while a long buy order is open".
        self.rm._open_order_symbols = {"AAPL"}
        ok, reason = self.rm.approve(make_signal(symbol="AAPL"), 100_000)
        assert not ok
        assert "Orden abierta" in reason

    def test_position_sizing(self):
        qty = self.rm.size_position(make_signal(), 100_000, atr=1.5)
        assert qty > 0
        # Con portfolio 100k, max 5% = 5000, risk_per_share = 3.0
        # Max shares = 5000/3 ≈ 1666, pero también limitado a 5000/150 ≈ 33
        assert qty <= 34

    # ── Tope por presupuesto de exposición restante (sizing sin margen) ───────
    def test_sizing_capped_by_remaining_exposure_budget(self):
        # Exposición al 79% del equity (100k): solo quedan $1.000 de presupuesto
        # (techo 80% − 79%), que a $150/acción son 6 acciones, muy por debajo del
        # tope del 5% (33). El sizing debe respetar el presupuesto, no el 5%.
        self.rm._total_exposure = 79_000
        qty = self.rm.size_position(make_signal(), 100_000, atr=1.5)
        assert qty == 6
        # La exposición resultante no supera el techo del 80% → sin margen.
        assert 79_000 + qty * 150 <= 0.80 * 100_000

    def test_sizing_returns_zero_when_budget_exhausted(self):
        # Con la exposición ya en el techo no debe abrirse nada. Antes el
        # max(1, …) forzaba 1 acción, que habría requerido margen.
        self.rm._total_exposure = 80_000
        assert self.rm.size_position(make_signal(), 100_000, atr=1.5) == 0

    def test_sizing_buying_power_cannot_exceed_budget(self):
        # Un buying_power enorme (margen ~3,5×) NO debe permitir superar el
        # presupuesto de exposición sin deuda: sigue topado a 6 acciones.
        self.rm._total_exposure = 79_000
        qty = self.rm.size_position(
            make_signal(), 100_000, atr=1.5, buying_power=350_000
        )
        assert qty == 6

    def test_compute_stops_buy(self):
        sl, tp = self.rm.compute_stops(150.0, atr=1.5, action="BUY")
        assert sl < 150.0   # Stop-loss por debajo del precio
        assert tp > 150.0   # Take-profit por encima

    def test_compute_stops_sell(self):
        sl, tp = self.rm.compute_stops(150.0, atr=1.5, action="SELL")
        assert sl > 150.0   # Stop-loss por encima (short)
        assert tp < 150.0   # Take-profit por debajo

    def test_daily_loss_circuit_breaker(self):
        self.rm._daily_pnl = -5500.0  # Pérdida del 5.5% sobre 100k
        ok, reason = self.rm.approve(make_signal(), 100_000)
        assert not ok
        assert self.rm.circuit_open

    def test_consecutive_losses_circuit_breaker(self):
        self.rm._consecutive_losses = 5
        ok, reason = self.rm.approve(make_signal(), 100_000)
        assert not ok

    def test_reset_daily_clears_state(self):
        self.rm._daily_pnl = -9999.0
        self.rm._circuit_open = True
        self.rm.reset_daily()
        assert self.rm._daily_pnl == 0.0
        assert not self.rm.circuit_open

    # ── P&L de cierres (regresión del bug de contabilidad de cortos) ──────────
    def test_long_close_pnl(self):
        assert self.rm.record_fill("NVDA", "buy", 200.0, 5) is None   # abrir largo
        pnl = self.rm.record_fill("NVDA", "sell", 210.0, 5)            # cerrar
        assert pnl == 50.0   # (210-200)*5

    def test_short_close_pnl(self):
        # Abrir corto es un SELL; cerrarlo es un BUY. Antes esto daba 0/None.
        assert self.rm.record_fill("TSLA", "sell", 400.0, 10) is None
        pnl = self.rm.record_fill("TSLA", "buy", 390.0, 10)
        assert pnl == 100.0   # (400-390)*10, beneficio en un corto

    def test_short_loss_pnl(self):
        self.rm.record_fill("MSFT", "sell", 445.0, 8)
        pnl = self.rm.record_fill("MSFT", "buy", 465.0, 8)
        assert pnl == -160.0  # (445-465)*8, pérdida en un corto

    def test_partial_close_fills(self):
        # Fills parciales (caso real MSFT 5+3) deben sumar el P&L total
        self.rm.record_fill("MSFT", "sell", 445.0, 8)
        p1 = self.rm.record_fill("MSFT", "buy", 465.0, 5)
        p2 = self.rm.record_fill("MSFT", "buy", 465.0, 3)
        assert round(p1 + p2, 2) == -160.0
        assert not self.rm.has_open_position("MSFT")  # cerrada del todo

    def test_partial_fills_count_as_one_loss(self):
        # Un cierre perdedor en 3 fills parciales = UNA pérdida consecutiva, no 3
        self.rm.record_fill("NVDA", "buy", 210.0, 4)     # abrir largo
        self.rm.record_fill("NVDA", "sell", 208.0, 2)    # parcial: sigue abierta
        assert self.rm._consecutive_losses == 0
        self.rm.record_fill("NVDA", "sell", 208.0, 1)    # parcial
        self.rm.record_fill("NVDA", "sell", 208.0, 1)    # cierre completo
        assert self.rm._consecutive_losses == 1

    def test_winning_close_resets_streak(self):
        self.rm._consecutive_losses = 3
        self.rm.record_fill("X", "buy", 100.0, 2)
        self.rm.record_fill("X", "sell", 105.0, 2)       # ganancia
        assert self.rm._consecutive_losses == 0
