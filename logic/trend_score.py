def calculate_trend_score(s):
    score = 0
    # EMA-Struktur
    if s.ema9 > s.ema20 > s.ema50:
        score += 30
    elif s.ema9 > s.ema20:
        score += 15
    # RSI
    if 50 <= s.rsi <= 65:
        score += 20
    elif 45 <= s.rsi < 50 or 65 < s.rsi <= 70:
        score += 10
    # Volumen
    if s.volume_ratio > 1.5:
        score += 20
    elif s.volume_ratio > 1.1:
        score += 10
    # ATR
    if s.atr / s.price > 0.015:
        score += 15
    # Marktphase
    if s.market_state == "OPEN":
        score += 15
    elif s.market_state == "PRE":
        score += 8

    return min(score, 100)
