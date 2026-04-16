"""Señales técnicas basadas en reglas (capa 1 del agregador)."""
from __future__ import annotations

from ..indicators.calculator import IndicatorSnapshot


def rule_based_signal(snap: IndicatorSnapshot) -> tuple[str, list[str]]:
    """
    Aplica reglas técnicas clásicas y retorna (señal, razones).
    Señal: "BUY", "SELL" o "HOLD"
    """
    if not snap.valid:
        return "HOLD", ["Indicadores no disponibles"]

    buy_points = 0
    sell_points = 0
    reasons = []

    # ---- RSI ----
    if snap.rsi < 30:
        buy_points += 2
        reasons.append(f"RSI sobrevendido ({snap.rsi:.1f})")
    elif snap.rsi < 40:
        buy_points += 1
        reasons.append(f"RSI bajo ({snap.rsi:.1f})")
    elif snap.rsi > 70:
        sell_points += 2
        reasons.append(f"RSI sobrecomprado ({snap.rsi:.1f})")
    elif snap.rsi > 60:
        sell_points += 1
        reasons.append(f"RSI alto ({snap.rsi:.1f})")

    # ---- MACD ----
    if snap.macd_hist > 0 and snap.macd > snap.macd_signal:
        buy_points += 1
        reasons.append("MACD cruce alcista")
    elif snap.macd_hist < 0 and snap.macd < snap.macd_signal:
        sell_points += 1
        reasons.append("MACD cruce bajista")

    # ---- Bandas de Bollinger ----
    if snap.bb_pct < 0.1:
        buy_points += 2
        reasons.append(f"Precio cerca de BB inferior ({snap.bb_pct:.2f})")
    elif snap.bb_pct > 0.9:
        sell_points += 2
        reasons.append(f"Precio cerca de BB superior ({snap.bb_pct:.2f})")

    # ---- EMA Crossover ----
    if snap.ema_crossover > 0.002:
        buy_points += 1
        reasons.append(f"EMA cruce alcista ({snap.ema_crossover:.3f})")
    elif snap.ema_crossover < -0.002:
        sell_points += 1
        reasons.append(f"EMA cruce bajista ({snap.ema_crossover:.3f})")

    # ---- Precio vs SMA200 (tendencia mayor) ----
    if snap.price_vs_sma200 > 0.02:
        buy_points += 1
        reasons.append("Precio sobre SMA200 (tendencia alcista)")
    elif snap.price_vs_sma200 < -0.02:
        sell_points += 1
        reasons.append("Precio bajo SMA200 (tendencia bajista)")

    # ---- Stochastic ----
    if snap.stoch_k < 20 and snap.stoch_d < 20:
        buy_points += 1
        reasons.append(f"Stochastic sobrevendido (K={snap.stoch_k:.1f})")
    elif snap.stoch_k > 80 and snap.stoch_d > 80:
        sell_points += 1
        reasons.append(f"Stochastic sobrecomprado (K={snap.stoch_k:.1f})")

    # ---- Volumen (confirmación) ----
    if snap.volume_ratio > 1.5:
        # El volumen amplifica la señal dominante
        if buy_points > sell_points:
            buy_points += 1
            reasons.append(f"Volumen alto confirma (x{snap.volume_ratio:.1f})")
        elif sell_points > buy_points:
            sell_points += 1
            reasons.append(f"Volumen alto confirma bajada (x{snap.volume_ratio:.1f})")

    # ---- Decisión final ----
    if buy_points >= 4 and buy_points > sell_points + 1:
        return "BUY", reasons
    elif sell_points >= 4 and sell_points > buy_points + 1:
        return "SELL", reasons
    else:
        return "HOLD", reasons
