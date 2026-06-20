"""Señales técnicas basadas en reglas (capa 1 del agregador)."""
from __future__ import annotations

from ..indicators.calculator import IndicatorSnapshot


def rule_based_signal(
    snap: IndicatorSnapshot,
    daily_trend: float | None = None,
    trend_buffer: float = 0.0,
) -> tuple[str, list[str], int]:
    """
    Aplica reglas técnicas clásicas y retorna (señal, razones, fuerza).
    Señal: "BUY", "SELL" o "HOLD"
    Fuerza: puntos de la dirección dominante (0 si HOLD). Permite al agregador
    graduar la confianza en lugar de usar un valor fijo.

    daily_trend: precio/SMA-1 del timeframe mayor (p. ej. SMA50 diaria) usado por
    el filtro de tendencia. Si es None se cae al SMA200 intradía como respaldo.
    """
    if not snap.valid:
        return "HOLD", ["Indicadores no disponibles"], 0

    buy_points = 0
    sell_points = 0
    reasons = []

    # ---- RSI ----
    # Umbrales menos agresivos: RSI 60 es normal en una subida, no una venta.
    # El tier débil de venta sube a >65 y el de compra baja a <35 (simétrico).
    if snap.rsi < 30:
        buy_points += 2
        reasons.append(f"RSI sobrevendido ({snap.rsi:.1f})")
    elif snap.rsi < 35:
        buy_points += 1
        reasons.append(f"RSI bajo ({snap.rsi:.1f})")
    elif snap.rsi > 70:
        sell_points += 2
        reasons.append(f"RSI sobrecomprado ({snap.rsi:.1f})")
    elif snap.rsi > 65:
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

    # ---- Decisión preliminar ----
    if buy_points >= 4 and buy_points > sell_points + 1:
        action, strength = "BUY", buy_points
    elif sell_points >= 4 and sell_points > buy_points + 1:
        action, strength = "SELL", sell_points
    else:
        return "HOLD", reasons, 0

    # ---- Filtro de tendencia mayor (ESTRICTO) ----
    # No operar contra la tendencia mayor: la estrategia es de reversión a la
    # media y, sin este filtro, abría cortos en plena tendencia alcista (la causa
    # del -24%). Se usa la tendencia del timeframe superior (p. ej. SMA50 diaria)
    # si está disponible; si no, el SMA200 intradía como respaldo.
    #   tendencia alcista → se vetan los cortos (SELL)
    #   tendencia bajista → se vetan los largos (BUY)
    major_trend = daily_trend if daily_trend is not None else snap.price_vs_sma200
    # Con trend_buffer, además de vetar ir contra la tendencia, se veta operar si
    # la tendencia es PLANA (precio dentro de ±buffer del SMA): evita el flip-flop
    # de dirección en valores laterales (p. ej. NVDA oscilando sobre su SMA50).
    if action == "SELL" and major_trend > -trend_buffer:
        reasons.append("Veto: corto bloqueado (tendencia mayor no bajista)")
        return "HOLD", reasons, 0
    if action == "BUY" and major_trend < trend_buffer:
        reasons.append("Veto: largo bloqueado (tendencia mayor no alcista)")
        return "HOLD", reasons, 0

    return action, reasons, strength
