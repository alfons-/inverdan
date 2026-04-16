"""Tests de señales técnicas."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from inverdan.indicators.calculator import IndicatorSnapshot
from inverdan.signals.rules import rule_based_signal
from inverdan.signals.aggregator import SignalAggregator


def make_snap(**kwargs) -> IndicatorSnapshot:
    defaults = dict(
        valid=True, close=150.0, volume=1_000_000, vwap=149.5,
        rsi=50.0, macd=0.0, macd_signal=0.0, macd_hist=0.0,
        bb_pct=0.5, bb_upper=155.0, bb_lower=145.0, bb_middle=150.0,
        ema_fast=150.0, ema_slow=150.0, ema_crossover=0.0,
        adx=25.0, stoch_k=50.0, stoch_d=50.0,
        atr=1.5, sma_200=140.0, price_vs_sma200=0.07,
        price_vs_vwap=0.003, cci=0.0, williams_r=-50.0,
        bb_width=0.06, obv=1e6, volume_sma=800_000, volume_ratio=1.25,
    )
    defaults.update(kwargs)
    return IndicatorSnapshot(**defaults)


class TestRuleBasedSignal:
    def test_oversold_rsi_buys(self):
        snap = make_snap(rsi=25.0, bb_pct=0.05, macd_hist=0.2, ema_crossover=0.003)
        action, reasons = rule_based_signal(snap)
        assert action == "BUY"

    def test_overbought_rsi_sells(self):
        snap = make_snap(rsi=75.0, bb_pct=0.95, macd_hist=-0.2, ema_crossover=-0.003)
        action, reasons = rule_based_signal(snap)
        assert action == "SELL"

    def test_neutral_holds(self):
        snap = make_snap(rsi=50.0)
        action, _ = rule_based_signal(snap)
        assert action == "HOLD"

    def test_invalid_snap_holds(self):
        snap = IndicatorSnapshot(valid=False)
        action, _ = rule_based_signal(snap)
        assert action == "HOLD"

    def test_volume_amplifies_buy(self):
        snap = make_snap(rsi=28.0, bb_pct=0.08, macd_hist=0.3, ema_crossover=0.004,
                         volume_ratio=2.0, price_vs_sma200=0.03)
        action, reasons = rule_based_signal(snap)
        assert action == "BUY"
        assert any("Volumen" in r for r in reasons)


class TestSignalAggregator:
    def test_contradictory_signals_hold(self):
        from inverdan.signals.aggregator import SignalAggregator
        from unittest.mock import MagicMock

        settings = MagicMock()
        settings.ml.confidence_threshold = 0.65
        registry = MagicMock()
        registry.predict.return_value = ("SELL", 0.75)

        agg = SignalAggregator.__new__(SignalAggregator)
        agg._cfg = settings
        agg._registry = registry

        action, conf, reasons = SignalAggregator._aggregate("BUY", "SELL", 0.75, 0.65)
        assert action == "HOLD"

    def test_agreement_increases_confidence(self):
        action, conf, reasons = SignalAggregator._aggregate("BUY", "BUY", 0.80, 0.65)
        assert action == "BUY"
        assert conf > 0.80

    def test_no_model_uses_rules(self):
        action, conf, _ = SignalAggregator._aggregate("BUY", "HOLD", 0.0, 0.65)
        assert action == "BUY"
        assert conf > 0.5
