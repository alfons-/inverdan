"""Tests de calculador de indicadores."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from inverdan.indicators.calculator import IndicatorCalculator


def make_settings():
    cfg = MagicMock()
    cfg.indicators.rsi_period = 14
    cfg.indicators.macd_fast = 12
    cfg.indicators.macd_slow = 26
    cfg.indicators.macd_signal = 9
    cfg.indicators.bb_period = 20
    cfg.indicators.bb_std = 2.0
    cfg.indicators.atr_period = 14
    cfg.indicators.stoch_k = 14
    cfg.indicators.stoch_d = 3
    cfg.indicators.ema_fast = 9
    cfg.indicators.ema_slow = 21
    cfg.indicators.sma_200 = 200
    return cfg


def make_ohlcv(n: int = 100, base: float = 150.0) -> pd.DataFrame:
    np.random.seed(42)
    close = base + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = (np.random.randint(500_000, 2_000_000, n)).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                          "close": close, "volume": volume, "vwap": close}, index=idx)


class TestIndicatorCalculator:
    def setup_method(self):
        self.calc = IndicatorCalculator(make_settings())

    def test_empty_df_returns_invalid(self):
        snap = self.calc.compute(pd.DataFrame())
        assert not snap.valid

    def test_too_short_df_returns_invalid(self):
        snap = self.calc.compute(make_ohlcv(10))
        assert not snap.valid

    def test_valid_df_computes_indicators(self):
        snap = self.calc.compute(make_ohlcv(100))
        assert snap.valid
        assert 0 <= snap.rsi <= 100
        assert snap.close > 0
        assert snap.atr >= 0
        assert snap.bb_upper > snap.bb_lower

    def test_rsi_bounds(self):
        df = make_ohlcv(200)
        snap = self.calc.compute(df)
        assert 0 <= snap.rsi <= 100

    def test_bb_pct_reasonable(self):
        df = make_ohlcv(100)
        snap = self.calc.compute(df)
        # bb_pct puede salirse de [0,1] en mercados muy volátiles
        assert isinstance(snap.bb_pct, float)
        assert not np.isnan(snap.bb_pct)

    def test_price_vs_sma200(self):
        df = make_ohlcv(300)  # Necesitamos >200 barras para SMA200
        snap = self.calc.compute(df)
        assert snap.valid
        # Con 300 barras el precio debería estar cercano a SMA200
        assert abs(snap.price_vs_sma200) < 0.5
