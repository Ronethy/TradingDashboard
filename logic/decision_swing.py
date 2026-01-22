from logic.decision_base import score_to_ampel

def decide_swing(snapshot):
    reasons = []
    score = 0

    # Gatekeeper
    if snapshot.ema20 <= snapshot.ema50:
        return "🔴 Rot – Kein Trade", ["Kein stabiler Aufwärtstrend"]

    if not (40 <= snapshot.rsi <= 65):
        return "🔴 Rot – Kein Trade", ["RSI außerhalb gesunder Zone"]

    # Scoring
    score += 30
    reasons.append("Trendstruktur intakt")

    if snapshot.volume_ratio > 1.2:
        score += 15
        reasons.append("Volumen bestätigt Bewegung")

    if snapshot.atr > 0:
        score += 10
        reasons.append("Ausreichende Volatilität")

    ampel = score_to_ampel(score, green=65, yellow=50)
    return ampel, reasons
