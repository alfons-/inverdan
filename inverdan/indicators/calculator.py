"""Calculador de indicadores técnicos usando la librería ta."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import ta
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False

from ..config.settings import Settings
from ..utils.logger import get_logger

logger = get_logger("indicators")


@dataclass
class IndicatorSnapshot:
    """Snapshot de todos los indicadores para una barra concreta."""
    # Precio
    close: float = 0.0
    volume: float = 0.0
    vwap: float = 0.0

    # Tendencia
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    sma_200: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    adx: float = 0.0

    # Momentum
    rsi: float = 50.0
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    cci: float = 0.0
    williams_r: float = -50.0

    # Volatilidad
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    bb_pct: float = 0.5
    atr: float = 0.0

    # Volumen
    obv: float = 0.0
    volume_sma: float = 0.0
    volume_ratio: float = 1.0

    # Derivados
    price_vs_sma200: float = 0.0   # close/sma200 - 1
    price_vs_vwap: float = 0.0     # close/vwap - 1
    ema_crossover: float = 0.0     # ema_fast/ema_slow - 1

    valid: bool = False


class IndicatorCalculator:
    """Calcula todos los indicadores técnicos sobre un DataFrame OHLCV."""

    def __init__(self, settings: Settings):
        self._cfg = settings.indicators

    def compute(self, df: pd.DataFrame) -> IndicatorSnapshot:
        snap = IndicatorSnapshot()
        if df is None or len(df) < 20:
            return snap

        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]

            # ---- Tendencia ----
            snap.ema_fast = self._last(
                close.ewm(span=self._cfg.ema_fast, adjust=False).mean()
            )
            snap.ema_slow = self._last(
                close.ewm(span=self._cfg.ema_slow, adjust=False).mean()
            )
            sma200 = close.rolling(min(self._cfg.sma_200, len(df))).mean()
            snap.sma_200 = self._last(sma200)

            # MACD
            ema_fast = close.ewm(span=self._cfg.macd_fast, adjust=False).mean()
            ema_slow = close.ewm(span=self._cfg.macd_slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            macd_sig = macd_line.ewm(span=self._cfg.macd_signal, adjust=False).mean()
            snap.macd = self._last(macd_line)
            snap.macd_signal = self._last(macd_sig)
            snap.macd_hist = snap.macd - snap.macd_signal

            # ADX
            if _TA_AVAILABLE and len(df) >= 14:
                adx = ta.trend.ADXIndicator(high, low, close, window=14)
                snap.adx = self._last(adx.adx())

            # ---- Momentum ----
            if _TA_AVAILABLE:
                rsi = ta.momentum.RSIIndicator(close, window=self._cfg.rsi_period)
                snap.rsi = self._last(rsi.rsi())

                stoch = ta.momentum.StochasticOscillator(
                    high, low, close,
                    window=self._cfg.stoch_k,
                    smooth_window=self._cfg.stoch_d
                )
                snap.stoch_k = self._last(stoch.stoch())
                snap.stoch_d = self._last(stoch.stoch_signal())

                cci = ta.trend.CCIIndicator(high, low, close, window=20)
                snap.cci = self._last(cci.cci())

                wr = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14)
                snap.williams_r = self._last(wr.williams_r())
            else:
                snap.rsi = self._rsi_manual(close, self._cfg.rsi_period)

            # ---- Volatilidad ----
            if _TA_AVAILABLE:
                bb = ta.volatility.BollingerBands(
                    close, window=self._cfg.bb_period, window_dev=self._cfg.bb_std
                )
                snap.bb_upper = self._last(bb.bollinger_hband())
                snap.bb_middle = self._last(bb.bollinger_mavg())
                snap.bb_lower = self._last(bb.bollinger_lband())
                snap.bb_width = self._last(bb.bollinger_wband())
                snap.bb_pct = self._last(bb.bollinger_pband())

                atr = ta.volatility.AverageTrueRange(high, low, close, window=self._cfg.atr_period)
                snap.atr = self._last(atr.average_true_range())
            else:
                # Bollinger manual
                sma = close.rolling(self._cfg.bb_period).mean()
                std = close.rolling(self._cfg.bb_period).std()
                snap.bb_middle = self._last(sma)
                snap.bb_upper = self._last(sma + self._cfg.bb_std * std)
                snap.bb_lower = self._last(sma - self._cfg.bb_std * std)
                snap.bb_width = (snap.bb_upper - snap.bb_lower) / snap.bb_middle if snap.bb_middle else 0
                band = snap.bb_upper - snap.bb_lower
                snap.bb_pct = (snap.close - snap.bb_lower) / band if band else 0.5

            # ---- Volumen ----
            if _TA_AVAILABLE:
                obv = ta.volume.OnBalanceVolumeIndicator(close, volume)
                snap.obv = self._last(obv.on_balance_volume())

            vol_sma = volume.rolling(20).mean()
            snap.volume_sma = self._last(vol_sma)
            snap.volume_ratio = volume.iloc[-1] / snap.volume_sma if snap.volume_sma > 0 else 1.0

            # ---- Precio actual ----
            snap.close = float(close.iloc[-1])
            snap.volume = float(volume.iloc[-1])
            snap.vwap = float(df["vwap"].iloc[-1]) if "vwap" in df.columns else snap.close

            # ---- Derivados ----
            if snap.sma_200 > 0:
                snap.price_vs_sma200 = snap.close / snap.sma_200 - 1
            if snap.vwap > 0:
                snap.price_vs_vwap = snap.close / snap.vwap - 1
            if snap.ema_slow > 0:
                snap.ema_crossover = snap.ema_fast / snap.ema_slow - 1

            snap.valid = True

        except Exception as e:
            logger.error(f"Error calculando indicadores: {e}")

        return snap

    @staticmethod
    def _last(series: pd.Series) -> float:
        try:
            val = series.iloc[-1]
            return float(val) if not np.isnan(val) else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _rsi_manual(close: pd.Series, period: int = 14) -> float:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return IndicatorCalculator._last(rsi)
