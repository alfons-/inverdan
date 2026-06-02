"""Etiquetado de datos para entrenamiento supervisado."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.settings import Settings


def forward_returns(close: pd.Series, forward_periods: int = 5) -> pd.Series:
    """Retorno futuro a N periodos. Las últimas N filas quedan NaN."""
    future_close = close.shift(-forward_periods)
    return (future_close - close) / close


def label_series(
    close: pd.Series,
    forward_periods: int = 5,
    buy_threshold: float = 0.005,
    sell_threshold: float = -0.005,
) -> pd.Series:
    """
    Etiquetado por umbral fijo (legado).

    Retorna: Serie con valores 1 (BUY), -1 (SELL), 0 (HOLD).
    Con umbrales agresivos (p. ej. ±0.5 % a 5 min) genera ~99 % HOLD, lo que
    degenera el modelo a un predictor constante. Preferir el etiquetado por
    cuantiles de `prepare_training_data`.
    """
    fr = forward_returns(close, forward_periods)
    labels = pd.Series(0, index=close.index, dtype=int)
    labels[fr > buy_threshold] = 1
    labels[fr < sell_threshold] = -1
    labels.iloc[-forward_periods:] = np.nan
    return labels


def label_by_quantile(
    fr: pd.Series,
    lower: float,
    upper: float,
) -> pd.Series:
    """
    Etiqueta según dónde cae el retorno futuro respecto a dos umbrales:
      fr <= lower  → SELL (-1)
      fr >= upper  → BUY  (1)
      en medio     → HOLD (0)

    `lower`/`upper` se calculan como cuantiles del retorno SOLO sobre el
    tramo de entrenamiento, evitando fuga de información del test.
    """
    labels = pd.Series(0, index=fr.index, dtype=int)
    labels[fr <= lower] = -1
    labels[fr >= upper] = 1
    return labels


def prepare_training_data(
    features_df: pd.DataFrame,
    close: pd.Series,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepara (X_train, X_test, y_train, y_test) con etiquetado por cuantiles.

    Los umbrales de cuantil se ajustan únicamente con los retornos del tramo de
    entrenamiento y luego se aplican a train y test por igual. Esto produce
    clases equilibradas (~33 % cada una en train) sin mirar el futuro del test.
    """
    tr = settings.training

    fr = forward_returns(close, tr.forward_return_periods)
    fr.name = "_fwd_return"

    # Alinear features con el retorno futuro y descartar NaN (incluye la cola sin futuro)
    combined = features_df.join(fr, how="inner").dropna()

    X = combined.drop(columns=["_fwd_return"])
    fr_aligned = combined["_fwd_return"]

    # Split temporal (NO random shuffle: la serie es temporal)
    split_idx = int(len(X) * (1 - tr.test_split))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    fr_train, fr_test = fr_aligned.iloc[:split_idx], fr_aligned.iloc[split_idx:]

    # Umbrales de cuantil ajustados SOLO con el tramo de entrenamiento
    lower = float(fr_train.quantile(tr.label_quantile))
    upper = float(fr_train.quantile(1.0 - tr.label_quantile))

    # Salvaguarda: si el mercado es tan plano que los cuantiles coinciden,
    # forzar una separación mínima simétrica para no colapsar las clases.
    if lower >= upper:
        eps = 1e-4
        lower, upper = -eps, eps

    y_train = label_by_quantile(fr_train, lower, upper)
    y_test = label_by_quantile(fr_test, lower, upper)

    return X_train, X_test, y_train, y_test
