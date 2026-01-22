def decide_swing(snapshot: MarketSnapshot) -> tuple[str, list[str]]:
    reasons = []

    if snapshot.rsi > 80:
        reasons.append("RSI stark überkauft – Pullback möglich")
        return "🟡 Vorsicht / warten", reasons

    if snapshot.rsi < 30:
        reasons.append("RSI stark überverkauft – potenzieller Einstieg Long")
        return "🟢 Swing Long möglich", reasons

    if snapshot.ema20 > snapshot.ema50 and snapshot.price > snapshot.ema20:
        reasons.append("Preis über EMA20 + EMA20 > EMA50 → Aufwärtstrend")
        if snapshot.volume_ratio > 1.5:
            reasons.append("Volumen stark erhöht → Momentum")
        return "🟢 Swing Long", reasons

    reasons.append("Kein klarer Swing-Trend erkennbar")
    return "🟡 Neutral / abwarten", reasons
