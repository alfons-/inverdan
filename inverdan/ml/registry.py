"""Registro de modelos ML con soporte de hot-reload."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Optional

from .random_forest import RandomForestModel
from ..utils.logger import get_logger

logger = get_logger("ml.registry")


class ModelRegistry:
    """Gestiona los modelos ML por símbolo. Thread-safe."""

    def __init__(self, model_path: Path):
        self._path = model_path
        self._models: Dict[str, RandomForestModel] = {}
        self._lock = threading.RLock()

    def load_all(self, symbols: list) -> int:
        loaded = 0
        for sym in symbols:
            if self.load(sym):
                loaded += 1
        logger.info(f"Cargados {loaded}/{len(symbols)} modelos")
        return loaded

    def load(self, symbol: str) -> bool:
        model = RandomForestModel()
        success = model.load(self._path, symbol)
        if success:
            with self._lock:
                self._models[symbol] = model
        return success

    def get(self, symbol: str) -> Optional[RandomForestModel]:
        with self._lock:
            return self._models.get(symbol)

    def predict(self, symbol: str, feature_vector) -> tuple[str, float]:
        """Retorna (acción, confianza). HOLD con confianza 0 si no hay modelo."""
        model = self.get(symbol)
        if model and model.is_ready:
            return model.predict_proba(feature_vector)
        return "HOLD", 0.0

    def reload(self, symbol: str) -> bool:
        """Hot-reload de un modelo sin parar el sistema."""
        logger.info(f"Recargando modelo {symbol}...")
        return self.load(symbol)

    @property
    def available_symbols(self) -> list:
        with self._lock:
            return list(self._models.keys())
