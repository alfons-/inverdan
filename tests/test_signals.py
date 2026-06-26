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
        action, reasons, strength = rule_based_signal(snap)
        assert action == "BUY"

    def test_overbought_rsi_sells(self):
        # price_vs_sma200 negativo: tendencia bajista, el filtro permite cortos
        snap = make_snap(rsi=75.0, bb_pct=0.95, macd_hist=-0.2, ema_crossover=-0.003,
                         price_vs_sma200=-0.07)
        action, reasons, strength = rule_based_signal(snap)
        assert action == "SELL"

    def test_short_vetoed_in_uptrend(self):
        # Misma señal de venta pero en tendencia alcista (precio > SMA200): vetada
        snap = make_snap(rsi=75.0, bb_pct=0.95, macd_hist=-0.2, ema_crossover=-0.003,
                         price_vs_sma200=0.07)
        action, reasons, strength = rule_based_signal(snap)
        assert action == "HOLD"
        assert any("Veto" in r for r in reasons)

    # ── Filtro de tendencia DIARIA (timeframe superior) ───────────────────────
    def test_short_vetoed_by_daily_uptrend(self):
        # Intradía permitiría el corto (price_vs_sma200 < 0), pero la tendencia
        # diaria es alcista (daily_trend > 0) → vetado.
        snap = make_snap(rsi=75.0, bb_pct=0.95, ema_crossover=-0.003,
                         price_vs_sma200=-0.07)
        action, reasons, _ = rule_based_signal(snap, daily_trend=0.05)
        assert action == "HOLD"
        assert any("Veto" in r for r in reasons)

    def test_short_allowed_when_daily_downtrend(self):
        # Intradía vetaría (price_vs_sma200 > 0), pero la tendencia diaria es
        # bajista (daily_trend < 0) → el corto SÍ se permite.
        snap = make_snap(rsi=75.0, bb_pct=0.95, ema_crossover=-0.003,
                         price_vs_sma200=0.07)
        action, _, _ = rule_based_signal(snap, daily_trend=-0.05)
        assert action == "SELL"

    def test_buy_vetoed_by_daily_downtrend(self):
        snap = make_snap(rsi=25.0, bb_pct=0.05, ema_crossover=0.003,
                         price_vs_sma200=0.07)
        action, reasons, _ = rule_based_signal(snap, daily_trend=-0.05)
        assert action == "HOLD"
        assert any("Veto" in r for r in reasons)

    def test_trend_buffer_vetoes_flat_trend(self):
        # Tendencia plana (dentro de ±buffer del SMA): se vetan AMBAS direcciones
        # para no hacer flip-flop en valores laterales.
        sell = make_snap(rsi=75.0, bb_pct=0.95, ema_crossover=-0.003, price_vs_sma200=0.0)
        a_sell, r_sell, _ = rule_based_signal(sell, daily_trend=0.005, trend_buffer=0.01)
        assert a_sell == "HOLD" and any("Veto" in r for r in r_sell)
        buy = make_snap(rsi=25.0, bb_pct=0.05, ema_crossover=0.003, price_vs_sma200=0.0)
        a_buy, _, _ = rule_based_signal(buy, daily_trend=0.005, trend_buffer=0.01)
        assert a_buy == "HOLD"   # +0,5% < buffer 1% → largo también vetado

    def test_trend_buffer_allows_clear_trend(self):
        # Fuera del buffer (tendencia clara) sí se permite operar.
        buy = make_snap(rsi=25.0, bb_pct=0.05, ema_crossover=0.003, price_vs_sma200=0.0)
        a, _, _ = rule_based_signal(buy, daily_trend=0.03, trend_buffer=0.01)
        assert a == "BUY"   # +3% > buffer 1%

    def test_daily_trend_none_falls_back_to_intraday(self):
        # Sin dato diario (None) se usa el SMA200 intradía: aquí es bajista
        # (price_vs_sma200 < 0), así que el corto se permite.
        snap = make_snap(rsi=75.0, bb_pct=0.95, ema_crossover=-0.003,
                         price_vs_sma200=-0.07)
        action, _, _ = rule_based_signal(snap, daily_trend=None)
        assert action == "SELL"

    def test_neutral_holds(self):
        snap = make_snap(rsi=50.0)
        action, _, _ = rule_based_signal(snap)
        assert action == "HOLD"

    def test_invalid_snap_holds(self):
        snap = IndicatorSnapshot(valid=False)
        action, _, _ = rule_based_signal(snap)
        assert action == "HOLD"

    def test_volume_amplifies_buy(self):
        snap = make_snap(rsi=28.0, bb_pct=0.08, macd_hist=0.3, ema_crossover=0.004,
                         volume_ratio=2.0, price_vs_sma200=0.03)
        action, reasons, strength = rule_based_signal(snap)
        assert action == "BUY"
        assert any("Volumen" in r for r in reasons)

    # ── Filtro de ADX (régimen) ───────────────────────────────────────────────
    def test_high_adx_vetoes_entry(self):
        # Señal de venta válida pero con ADX por encima del techo → vetada
        snap = make_snap(rsi=75.0, bb_pct=0.95, ema_crossover=-0.003,
                         price_vs_sma200=-0.07, adx=50.0)
        action, reasons, _ = rule_based_signal(snap, max_adx=40.0)
        assert action == "HOLD"
        assert any("ADX" in r for r in reasons)

    def test_adx_filter_disabled_by_default(self):
        # Sin max_adx (0 = off) la misma señal con ADX alto SÍ se permite
        snap = make_snap(rsi=75.0, bb_pct=0.95, ema_crossover=-0.003,
                         price_vs_sma200=-0.07, adx=50.0)
        action, _, _ = rule_based_signal(snap)
        assert action == "SELL"

    def test_adx_below_ceiling_allows_entry(self):
        snap = make_snap(rsi=75.0, bb_pct=0.95, ema_crossover=-0.003,
                         price_vs_sma200=-0.07, adx=20.0)
        action, _, _ = rule_based_signal(snap, max_adx=40.0)
        assert action == "SELL"


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

        action, conf, reasons = SignalAggregator._aggregate("BUY", 5, "SELL", 0.75, 0.65)
        assert action == "HOLD"

    def test_agreement_increases_confidence(self):
        action, conf, reasons = SignalAggregator._aggregate("BUY", 5, "BUY", 0.80, 0.65)
        assert action == "BUY"
        assert conf > 0.80

    def test_no_model_uses_rules(self):
        action, conf, _ = SignalAggregator._aggregate("BUY", 5, "HOLD", 0.0, 0.65)
        assert action == "BUY"
        assert conf > 0.5

    def test_strong_signal_higher_confidence(self):
        # Una señal de 8 puntos debe dar más confianza que una de 4
        _, weak, _ = SignalAggregator._aggregate("BUY", 4, "HOLD", 0.0, 0.65)
        _, strong, _ = SignalAggregator._aggregate("BUY", 8, "HOLD", 0.0, 0.65)
        assert strong > weak

    # ── ML como confirmación/veto (umbral 0,40), nunca iniciador ──────────────
    def test_ml_does_not_initiate_when_rules_hold(self):
        # Reglas en HOLD y ML activo: el ML NO debe abrir operación por sí solo
        action, conf, _ = SignalAggregator._aggregate("HOLD", 0, "BUY", 0.55, 0.40)
        assert action == "HOLD"

    def test_ml_vetoes_when_disagrees_at_new_threshold(self):
        # Con umbral 0,40 un ML que contradice a las reglas (conf>=0,40) las veta
        action, _, _ = SignalAggregator._aggregate("BUY", 5, "SELL", 0.45, 0.40)
        assert action == "HOLD"

    def test_ml_confirms_agreement_at_new_threshold(self):
        action, conf, _ = SignalAggregator._aggregate("BUY", 5, "BUY", 0.45, 0.40)
        assert action == "BUY"
        assert conf > 0.6

    # ── Multiplicador de calidad (ADX/volumen) sobre la confianza ─────────────
    def test_quality_penalizes_high_adx(self):
        q_low = SignalAggregator._quality_multiplier(make_snap(adx=15.0, volume_ratio=1.5))
        q_high = SignalAggregator._quality_multiplier(make_snap(adx=38.0, volume_ratio=1.5))
        assert q_high < q_low
        assert q_high < 1.0          # ADX alto penaliza la confianza

    def test_quality_rewards_volume(self):
        q_lo = SignalAggregator._quality_multiplier(make_snap(adx=20.0, volume_ratio=1.0))
        q_hi = SignalAggregator._quality_multiplier(make_snap(adx=20.0, volume_ratio=2.5))
        assert q_hi > q_lo           # más volumen, más confianza

    def test_quality_neutral_around_baseline(self):
        # ADX 20 y volumen 1.5x → factor ~1.0 (no altera la confianza)
        q = SignalAggregator._quality_multiplier(make_snap(adx=20.0, volume_ratio=1.5))
        assert 0.97 <= q <= 1.03
