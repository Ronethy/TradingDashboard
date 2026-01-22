from logic.decision_base import score_to_ampel

def decide_daytrade(snapshot):
    reasons = []
    score = 0

    # Gatekeeper
    if snapshot.market_state not in ["PRE", "OPEN"]:
        return "🔴 Rot – Kein Trade", ["Markt geschlossen"]

    if snapshot.volume_ratio < 1.5:
        return "🔴 Rot – Kein Trade", ["Zu wenig Volumen"]

    if snapshot.rsi > 80:
        return "🔴 Rot – Kein Trade", ["RSI überhitzt"]

    # Scoring
    if snapshot.ema9 > snapshot.ema20:
        score += 25
        reasons.append("Kurzfristiger Aufwärtstrend")

    if snapshot.volume_ratio > 2:
        score += 20
        reasons.append("Starker Volumen-Impuls")

    if snapshot.atr > 0:
        score += 15
        reasons.append("Bewegung vorhanden (ATR)")

    if snapshot.market_state == "OPEN":
        score += 10
        reasons.append("Regulärer Handel")

    ampel = score_to_ampel(score, green=70, yellow=50)
    return ampel, reasons
