def decide_swing(snapshot: MarketSnapshot) -> tuple[str, list[str]]:
    reasons = []

    if snapshot.rsi > 80:
        reasons.append("RSI stark überkauft – Pullback möglich")
        return "🟡 Vorsicht / warten", reasons

    if snapshot.rsi < 30:
        reasons.append("RSI stark überverkauft – potenzieller Einstieg")
        return "🟢 Swing Long möglich", reasons

    if snapshot.ema20 > snapshot.ema50 and snapshot.price > snapshot.ema20:
        reasons.append("Preis über EMA20 + EMA20 > EMA50")
        if snapshot.volume_ratio > 1.5:
            reasons.append("Starkes Volumen")
        return "🟢 Swing Long", reasons

    reasons.append("Kein klarer Swing-Setup erkennbar")
    return "🟡 Neutral / abwarten", reasons
