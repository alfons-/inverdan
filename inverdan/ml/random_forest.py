"""Modelo Random Forest para predicción de señales BUY/SELL/HOLD."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from ..utils.logger import get_logger

logger = get_logger("ml.rf")

_CLASSES = {-1: "SELL", 0: "HOLD", 1: "BUY"}


class RandomForestModel:
    """Wrapper del modelo RandomForest con persistencia y hot-reload."""

    def __init__(self):
        self._model: Optional[RandomForestClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._trained_at: Optional[str] = None
        self._symbol: Optional[str] = None

    def train(
        self,
        X_train,
        y_train,
        n_estimators: int = 200,
        max_depth: int = 10,
        random_state: int = 42,
    ) -> None:
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)

        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self._model.fit(X_scaled, y_train)
        self._trained_at = datetime.utcnow().isoformat()
        logger.info(f"Modelo entrenado con {len(X_train)} muestras")

    def evaluate(self, X_test, y_test) -> dict:
        if self._model is None:
            return {}
        X_scaled = self._scaler.transform(X_test)
        y_pred = self._model.predict(X_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        return report

    def predict_proba(self, feature_vector: np.ndarray) -> Tuple[str, float]:
        """
        Retorna (acción, confianza).
        acción: "BUY", "SELL" o "HOLD"
        confianza: probabilidad de la clase predicha [0, 1]
        """
        if self._model is None or self._scaler is None:
            return "HOLD", 0.0

        try:
            x = feature_vector.reshape(1, -1)
            x_scaled = self._scaler.transform(x)
            probs = self._model.predict_proba(x_scaled)[0]
            classes = self._model.classes_

            best_idx = np.argmax(probs)
            best_class = int(classes[best_idx])
            confidence = float(probs[best_idx])
            action = _CLASSES.get(best_class, "HOLD")

            return action, confidence
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return "HOLD", 0.0

    def save(self, path: Path, symbol: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / f"rf_{symbol}.joblib")
        joblib.dump(self._scaler, path / f"scaler_{symbol}.joblib")

        meta = {
            "symbol": symbol,
            "trained_at": self._trained_at,
            "model_type": "random_forest",
        }
        with open(path / f"meta_{symbol}.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Modelo guardado en {path} para {symbol}")

    def load(self, path: Path, symbol: str) -> bool:
        model_file = path / f"rf_{symbol}.joblib"
        scaler_file = path / f"scaler_{symbol}.joblib"
        meta_file = path / f"meta_{symbol}.json"

        if not model_file.exists() or not scaler_file.exists():
            logger.warning(f"No se encontró modelo para {symbol} en {path}")
            return False

        self._model = joblib.load(model_file)
        self._scaler = joblib.load(scaler_file)
        self._symbol = symbol

        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            self._trained_at = meta.get("trained_at")

        # Advertir si el modelo es antiguo (>30 días)
        if self._trained_at:
            trained = datetime.fromisoformat(self._trained_at)
            age_days = (datetime.utcnow() - trained).days
            if age_days > 30:
                logger.warning(f"AVISO: Modelo de {symbol} tiene {age_days} días. Considera reentrenar.")

        logger.info(f"Modelo cargado para {symbol} (entrenado: {self._trained_at})")
        return True

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._scaler is not None
