"""Tipos de datos para señales de trading."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class Signal:
    symbol: str
    action: str          # "BUY" | "SELL" | "HOLD"
    confidence: float    # [0, 1]
    price: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    rule_signal: str = "HOLD"
    ml_signal: str = "HOLD"
    ml_confidence: float = 0.0
    indicators: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "price": self.price,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "rule_signal": self.rule_signal,
            "ml_signal": self.ml_signal,
            "ml_confidence": round(self.ml_confidence, 4),
            **{f"ind_{k}": round(v, 4) if isinstance(v, float) else v
               for k, v in self.indicators.items()},
        }
