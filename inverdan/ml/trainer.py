"""Pipeline de entrenamiento. Usado por train.py, nunca por el sistema en vivo."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from ..config.settings import Settings
from ..data.historical import HistoricalDataClient
from ..indicators.calculator import IndicatorCalculator, IndicatorSnapshot
from ..ml.features import build_feature_matrix, build_feature_vector
from ..ml.labeler import prepare_training_data
from ..ml.random_forest import RandomForestModel
from ..utils.logger import get_logger

logger = get_logger("ml.trainer")


def train_symbol(
    symbol: str,
    settings: Settings,
    output_path: Path,
) -> bool:
    """Entrena y guarda un modelo para un símbolo. Retorna True si éxito."""
    logger.info(f"=== Entrenando modelo para {symbol} ===")

    # 1. Descargar datos históricos
    client = HistoricalDataClient(settings)
    df = client.fetch_bars(symbol, days=settings.training.lookback_days, cache=True)

    if df.empty or len(df) < 200:
        logger.error(f"Datos insuficientes para {symbol}: {len(df)} barras")
        return False

    logger.info(f"Datos descargados: {len(df)} barras para {symbol}")

    # 2. Calcular indicadores para cada barra (ventana deslizante)
    calculator = IndicatorCalculator(settings)
    min_window = max(settings.indicators.sma_200, 60)
    snaps: List[IndicatorSnapshot] = []

    for i in range(len(df)):
        window = df.iloc[max(0, i - min_window): i + 1]
        if len(window) < 50:
            snaps.append(IndicatorSnapshot())
        else:
            snap = calculator.compute(window)
            snaps.append(snap)

    logger.info(f"Indicadores calculados para {symbol}")

    # 3. Construir matriz de features
    features_df = build_feature_matrix(df, snaps)

    # 4. Etiquetas y split temporal
    X_train, X_test, y_train, y_test = prepare_training_data(
        features_df, df["close"], settings
    )

    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    logger.info(f"Distribución train: {y_train.value_counts().to_dict()}")

    # 5. Entrenar modelo
    model = RandomForestModel()
    tr = settings.training
    model.train(
        X_train, y_train,
        n_estimators=tr.n_estimators,
        max_depth=tr.max_depth,
        random_state=tr.random_state,
    )

    # 6. Evaluar
    report = model.evaluate(X_test, y_test)
    acc = report.get("accuracy", 0)
    logger.info(f"Accuracy en test para {symbol}: {acc:.3f}")

    # 7. Guardar
    model.save(output_path, symbol)
    return True
