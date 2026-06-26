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

    def __init__(self, settings: Settings, registry: ModelRegistry, trend_provider=None):
        self._cfg = settings
        self._registry = registry
        self._trend_provider = trend_provider  # tendencia mayor (timeframe superior); puede ser None

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

        # Capa 1: Reglas técnicas (con filtro de tendencia mayor del timeframe superior)
        daily_trend = self._trend_provider.get(symbol) if self._trend_provider else None
        trend_buffer = getattr(self._cfg.risk, "trend_buffer_pct", 0.0)
        max_adx = getattr(self._cfg.risk, "max_adx", 0.0)
        rule_signal, rule_reasons, rule_strength = rule_based_signal(
            snap, daily_trend=daily_trend, trend_buffer=trend_buffer, max_adx=max_adx
        )

        # Capa 2: Random Forest
        feature_vec = build_feature_vector(snap, timestamp)
        ml_action, ml_conf = self._registry.predict(symbol, feature_vec)

        # Agregación: requiere acuerdo entre capas
        threshold = self._cfg.ml.confidence_threshold
        final_action, final_conf, extra_reasons = self._aggregate(
            rule_signal, rule_strength, ml_action, ml_conf, threshold
        )

        # Ajuste por CALIDAD del contexto: el nº de reglas (rule_strength) no
        # discriminaba ganadoras de perdedoras (μ 0.59 vs 0.58). Lo que sí predice
        # es ADX bajo + volumen alto (ver análisis de operativa). Modulamos la
        # confianza con eso para que el filtro min_confidence (risk) vete trades flojos.
        if final_action != "HOLD":
            q = self._quality_multiplier(snap)
            final_conf = round(max(0.30, min(0.95, final_conf * q)), 2)
            adx_txt = f"{snap.adx:.0f}" if snap.adx is not None else "?"
            extra_reasons.append(f"calidad x{q:.2f} (ADX {adx_txt}, vol x{snap.volume_ratio:.1f})")

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
    def _rule_confidence(strength: int) -> float:
        """
        Mapea la fuerza de la señal (puntos de la dirección dominante, mínimo 4
        para disparar) a una confianza graduada en [0.55, 0.90]. Sustituye el
        antiguo valor fijo de 0.60 para que el umbral min_signal_confidence pueda
        distinguir señales fuertes de marginales.
        """
        return round(min(0.5 + 0.05 * (strength - 3), 0.90), 2)

    @staticmethod
    def _quality_multiplier(snap) -> float:
        """Factor de calidad del contexto para modular la confianza. Del análisis de
        operativa: las ganadoras entraban con ADX bajo (~18) y volumen alto (~2x); las
        perdedoras con ADX alto (~35) y volumen bajo (~1.5x).
          - ADX: penaliza tendencias fuertes (la reversión a la media falla ahí).
            Neutro ~20, baja hasta x0.6 hacia ADX 40.
          - Volumen: premia la confirmación. Neutro 1.5x, hasta x1.15 con volumen alto.
        """
        adx = snap.adx if snap.adx is not None else 20.0
        vol = snap.volume_ratio if snap.volume_ratio is not None else 1.0
        adx_adj = max(0.6, min(1.1, 1.0 - (adx - 20.0) * 0.02))
        vol_adj = max(0.85, min(1.15, 1.0 + (vol - 1.5) * 0.15))
        return round(adx_adj * vol_adj, 3)

    @classmethod
    def _aggregate(
        cls,
        rule: str,
        rule_strength: int,
        ml: str,
        ml_conf: float,
        threshold: float,
    ) -> tuple[str, float, list[str]]:
        reasons = []

        ml_active = ml != "HOLD" and ml_conf >= threshold
        rule_conf = cls._rule_confidence(rule_strength)

        # Si ML no tiene modelo (confianza = 0), confiar solo en reglas
        if ml_conf == 0.0:
            if rule != "HOLD":
                reasons.append("Solo reglas técnicas (sin modelo ML)")
                return rule, rule_conf, reasons
            return "HOLD", 0.0, []

        # Acuerdo total: combina fuerza de reglas y confianza ML
        if rule == ml and ml_active:
            combined = max(0.5 + ml_conf * 0.5, rule_conf)
            reasons.append(f"Reglas y ML de acuerdo ({ml})")
            return ml, combined, reasons

        # ML activo pero reglas en HOLD → el ML NO inicia operaciones por sí solo;
        # solo confirma o veta señales de las reglas. Su confianza en 3 clases es
        # demasiado baja (≈0.4) para fiarse de una entrada que proponga él solo.
        if ml_active and rule == "HOLD":
            return "HOLD", 0.0, []

        # Reglas activas pero ML en HOLD o baja confianza
        if rule != "HOLD" and (ml == "HOLD" or ml_conf < threshold):
            reasons.append(f"Reglas técnicas {rule}, ML indeciso")
            return rule, rule_conf, reasons

        # Señales contradictorias → no operar
        if rule != "HOLD" and ml != "HOLD" and rule != ml:
            reasons.append(f"Señales contradictorias: reglas={rule} vs ML={ml} → HOLD")
            return "HOLD", 0.0, reasons

        return "HOLD", 0.0, []
