"""Etiquetado de datos para entrenamiento supervisado."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.settings import Settings


def label_series(
    close: pd.Series,
    forward_periods: int = 5,
    buy_threshold: float = 0.005,
    sell_threshold: float = -0.005,
) -> pd.Series:
    """
    Etiqueta cada barra según el retorno futuro a N periodos.

    Retorna: Serie con valores 1 (BUY), -1 (SELL), 0 (HOLD).
    Las últimas `forward_periods` filas son NaN (sin etiqueta).

    IMPORTANTE: Nunca usar shuffle=True con esta serie — es temporal.
    """
    future_close = close.shift(-forward_periods)
    forward_return = (future_close - close) / close

    labels = pd.Series(0, index=close.index, dtype=int)
    labels[forward_return > buy_threshold] = 1
    labels[forward_return < sell_threshold] = -1
    # Las últimas N filas no tienen etiqueta válida
    labels.iloc[-forward_periods:] = np.nan

    return labels


def prepare_training_data(
    features_df: pd.DataFrame,
    close: pd.Series,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepara X, y para entrenamiento eliminando NaN y aplicando split temporal."""
    tr = settings.training

    labels = label_series(
        close,
        forward_periods=tr.forward_return_periods,
        buy_threshold=tr.buy_threshold,
        sell_threshold=tr.sell_threshold,
    )

    # Alinear features y labels
    combined = features_df.join(labels.rename("label"), how="inner")
    combined = combined.dropna()

    X = combined.drop(columns=["label"])
    y = combined["label"].astype(int)

    # Split temporal (NO random shuffle)
    split_idx = int(len(X) * (1 - tr.test_split))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test
