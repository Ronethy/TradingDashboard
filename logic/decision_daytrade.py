from logic.snapshot import MarketSnapshot
from typing import Tuple, List

def decide_daytrade(snapshot: MarketSnapshot) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if snapshot.market_state == "PRE":
        return "⚪ Pre-Market – warte auf Open", reasons

    if snapshot.rsi > 75:
        reasons.append("RSI überkauft (>75) → zu riskant für Daytrade")
        return "🔴 Vermeiden", reasons

    if snapshot.rsi < 25:
        reasons.append("RSI stark überverkauft – potenziell Bounce, aber Vorsicht")
        return "🟡 Abwarten / Long nur mit Bestätigung", reasons

    if snapshot.ema9 > snapshot.ema20 > snapshot.ema50:
        reasons.append("Perfektes EMA-Stacking bullisch")
        return "🟢 Long Daytrade möglich", reasons

    if snapshot.ema9 < snapshot.ema20:
        reasons.append("EMA9 unter EMA20 → Abwärtstrend im Kurzfristigen")
        return "🔴 Short oder meiden", reasons

    reasons.append("Kein klares Setup – neutral")
    return "🟡 Neutral", reasons
