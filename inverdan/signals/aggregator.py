"""Agregador de señales: fusiona reglas técnicas + ML para decisión final."""
from __future__ import annotations

import numpy as np
from datetime import datetime

from ..config.settings import Settings
from ..indicators.calculator import IndicatorSnapshot
from ..ml.features import build_feature_vector
from ..ml.registry import ModelRegistry
from ..signals.rules import rule_based_signal
from ..signals.signal_types import Signal
from ..utils.logger import get_logger
from ..utils.market_hours import is_market_open

logger = get_logger("signals.aggregator")


class SignalAggregator:
    """
    Tres capas de decisión:
      1. Reglas técnicas clásicas
      2. Random Forest ML
      Acuerdo de al menos 2 capas con confianza >= threshold → señal activa.
    """

    def __init__(self, settings: Settings, registry: ModelRegistry):
        self._cfg = settings
        self._registry = registry

    def evaluate(
        self,
        symbol: str,
        snap: IndicatorSnapshot,
        timestamp: datetime | None = None,
    ) -> Signal:
        # Verificar que el mercado esté abierto
        if not is_market_open():
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                price=snap.close,
                reasoning="Mercado cerrado",
                timestamp=timestamp or datetime.utcnow(),
            )

        if not snap.valid:
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                price=snap.close,
                reasoning="Indicadores insuficientes",
                timestamp=timestamp or datetime.utcnow(),
            )

        # Capa 1: Reglas técnicas
        rule_signal, rule_reasons = rule_based_signal(snap)

        # Capa 2: Random Forest
        feature_vec = build_feature_vector(snap, timestamp)
        ml_action, ml_conf = self._registry.predict(symbol, feature_vec)

        # Agregación: requiere acuerdo entre capas
        threshold = self._cfg.ml.confidence_threshold
        final_action, final_conf, extra_reasons = self._aggregate(
            rule_signal, ml_action, ml_conf, threshold
        )

        all_reasons = rule_reasons + extra_reasons
        reasoning = " | ".join(all_reasons[:5]) if all_reasons else "Sin señal clara"

        signal = Signal(
            symbol=symbol,
            action=final_action,
            confidence=final_conf,
            price=snap.close,
            reasoning=reasoning,
            timestamp=timestamp or datetime.utcnow(),
            rule_signal=rule_signal,
            ml_signal=ml_action,
            ml_confidence=ml_conf,
            indicators={
                "rsi": snap.rsi,
                "macd_hist": snap.macd_hist,
                "bb_pct": snap.bb_pct,
                "volume_ratio": snap.volume_ratio,
                "atr": snap.atr,
                "adx": snap.adx,
            },
        )

        if final_action != "HOLD":
            logger.info(
                f"SEÑAL {final_action} {symbol} @ {snap.close:.2f} "
                f"(conf={final_conf:.2f}, rule={rule_signal}, ml={ml_action}:{ml_conf:.2f})"
            )

        return signal

    @staticmethod
    def _aggregate(
        rule: str,
        ml: str,
        ml_conf: float,
        threshold: float,
    ) -> tuple[str, float, list[str]]:
        reasons = []

        ml_active = ml != "HOLD" and ml_conf >= threshold

        # Si ML no tiene modelo (confianza = 0), confiar solo en reglas
        if ml_conf == 0.0:
            if rule != "HOLD":
                reasons.append(f"Solo reglas técnicas (sin modelo ML)")
                return rule, 0.6, reasons
            return "HOLD", 0.0, []

        # Acuerdo total: máxima confianza
        if rule == ml and ml_active:
            combined = 0.5 + ml_conf * 0.5
            reasons.append(f"Reglas y ML de acuerdo ({ml})")
            return ml, combined, reasons

        # ML activo pero reglas en HOLD
        if ml_active and rule == "HOLD":
            reasons.append(f"ML {ml} (conf={ml_conf:.2f}), reglas neutras")
            return ml, ml_conf * 0.8, reasons

        # Reglas activas pero ML en HOLD o baja confianza
        if rule != "HOLD" and (ml == "HOLD" or ml_conf < threshold):
            reasons.append(f"Reglas técnicas {rule}, ML indeciso")
            return rule, 0.6, reasons

        # Señales contradictorias → no operar
        if rule != "HOLD" and ml != "HOLD" and rule != ml:
            reasons.append(f"Señales contradictorias: reglas={rule} vs ML={ml} → HOLD")
            return "HOLD", 0.0, reasons

        return "HOLD", 0.0, []
