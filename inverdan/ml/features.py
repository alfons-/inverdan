"""Feature engineering: convierte indicadores en vector de características para ML."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from ..indicators.calculator import IndicatorSnapshot
from ..config.settings import Settings
from ..utils.logger import get_logger

logger = get_logger("ml.features")

FEATURE_NAMES = [
    # Momentum
    "rsi", "rsi_norm",             # RSI raw y normalizado [0,1]
    "stoch_k", "stoch_d",
    "cci_norm",                     # CCI normalizado
    "williams_r_norm",

    # Tendencia
    "macd_hist_norm",
    "ema_crossover",
    "adx_norm",
    "price_vs_sma200",
    "price_vs_vwap",

    # Volatilidad
    "bb_pct",                       # Posición relativa dentro de BB [0,1]
    "bb_width_norm",
    "atr_norm",                     # ATR normalizado por precio

    # Volumen
    "volume_ratio",
    "obv_norm",

    # Patrones de vela
    "body_ratio",                   # (close-open) / (high-low)
    "upper_shadow",                 # sombra superior
    "lower_shadow",                 # sombra inferior

    # Temporal (hora del día)
    "hour_sin", "hour_cos",
]

N_FEATURES = len(FEATURE_NAMES)


def build_feature_vector(snap: IndicatorSnapshot, timestamp: Optional[pd.Timestamp] = None) -> np.ndarray:
    """Construye el vector de características a partir de un IndicatorSnapshot."""
    try:
        hour = timestamp.hour if timestamp is not None else 12

        # Normalización CCI [-200, +200] -> [-1, 1]
        cci_norm = np.clip(snap.cci / 200.0, -1, 1)
        # Williams R [-100, 0] -> [0, 1]
        wr_norm = (snap.williams_r + 100) / 100.0
        # MACD hist normalizado por ATR
        macd_hist_norm = snap.macd_hist / snap.atr if snap.atr > 0 else 0
        macd_hist_norm = np.clip(macd_hist_norm, -3, 3)
        # ATR normalizado por precio
        atr_norm = snap.atr / snap.close if snap.close > 0 else 0
        # OBV (cambio porcentual difícil sin historia, usar 0 como fallback)
        obv_norm = np.clip(snap.obv / 1e6, -1, 1)  # Escalar arbitrariamente
        # Patrones de vela (necesitamos open)
        body_ratio = 0.0
        upper_shadow = 0.0
        lower_shadow = 0.0

        # Ciclo diario (hora de mercado 9.5 - 16h -> ángulo)
        market_hour = max(0, min(hour - 9.5, 6.5))
        angle = market_hour / 6.5 * 2 * np.pi
        hour_sin = np.sin(angle)
        hour_cos = np.cos(angle)

        vec = np.array([
            snap.rsi / 100.0,
            snap.rsi / 100.0,
            snap.stoch_k / 100.0,
            snap.stoch_d / 100.0,
            cci_norm,
            wr_norm,
            macd_hist_norm / 3.0,   # escalar a [-1, 1]
            np.clip(snap.ema_crossover, -0.05, 0.05) / 0.05,
            np.clip(snap.adx / 100.0, 0, 1),
            np.clip(snap.price_vs_sma200, -0.2, 0.2) / 0.2,
            np.clip(snap.price_vs_vwap, -0.05, 0.05) / 0.05,
            snap.bb_pct,
            np.clip(snap.bb_width, 0, 0.1) / 0.1,
            np.clip(atr_norm, 0, 0.05) / 0.05,
            np.clip(snap.volume_ratio, 0, 5) / 5.0,
            obv_norm,
            body_ratio,
            upper_shadow,
            lower_shadow,
            hour_sin,
            hour_cos,
        ], dtype=np.float32)

        # Reemplazar NaN/Inf
        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
        return vec

    except Exception as e:
        logger.error(f"Error construyendo features: {e}")
        return np.zeros(N_FEATURES, dtype=np.float32)


def build_feature_matrix(df: pd.DataFrame, indicators_history: list) -> pd.DataFrame:
    """Construye matriz de características para entrenamiento."""
    rows = []
    for i, snap in enumerate(indicators_history):
        if snap.valid:
            ts = df.index[i] if i < len(df) else None
            vec = build_feature_vector(snap, ts)
            rows.append(vec)
        else:
            rows.append(np.zeros(N_FEATURES, dtype=np.float32))
    return pd.DataFrame(rows, columns=FEATURE_NAMES, index=df.index[:len(rows)])
