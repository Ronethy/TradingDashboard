def decide_daytrade(snapshot: MarketSnapshot) -> tuple[str, list[str]]:
    reasons = []

    if snapshot.market_state == "PRE":
        return "⚪ Pre-Market – warte auf Open", reasons

    if snapshot.rsi > 75:
        reasons.append("RSI überkauft (>75)")
        return "🔴 Vermeiden", reasons

    if snapshot.rsi < 25:
        reasons.append("RSI stark überverkauft")
        return "🟡 Abwarten / Long nur mit Bestätigung", reasons

    if snapshot.ema9 > snapshot.ema20 > snapshot.ema50:
        reasons.append("Bullisches EMA-Stacking")
        return "🟢 Long Daytrade möglich", reasons

    if snapshot.ema9 < snapshot.ema20:
        reasons.append("Kurzfristiger Abwärtstrend")
        return "🔴 Short oder meiden", reasons

    reasons.append("Kein klares Intraday-Setup")
    return "🟡 Neutral", reasons
