from __future__ import annotations

from logic.snapshot import MarketSnapshot

def calculate_trend_score(s: MarketSnapshot) -> int:
    score = 0

    # EMA-Struktur (max 30)
    if s.ema9 > s.ema20 > s.ema50:
        score += 30
    elif s.ema9 > s.ema20:
        score += 15

    # RSI (max 20)
    if 50 <= s.rsi <= 65:
        score += 20
    elif 45 <= s.rsi < 50 or 65 < s.rsi <= 70:
        score += 10

    # Volumen-Ratio (max 20)
    if s.volume_ratio > 1.5:
        score += 20
    elif s.volume_ratio > 1.1:
        score += 10

    # ATR (max 15)
    if s.atr / s.price > 0.015:
        score += 15

    # Marktphase (max 15)
    if s.market_state == "OPEN":
        score += 15
    elif s.market_state == "PRE":
        score += 8

    return min(score, 100)
