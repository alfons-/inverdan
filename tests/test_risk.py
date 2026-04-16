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

    def test_position_sizing(self):
        qty = self.rm.size_position(make_signal(), 100_000, atr=1.5)
        assert qty > 0
        # Con portfolio 100k, max 5% = 5000, risk_per_share = 3.0
        # Max shares = 5000/3 ≈ 1666, pero también limitado a 5000/150 ≈ 33
        assert qty <= 34

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
