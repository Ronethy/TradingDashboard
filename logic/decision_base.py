def score_to_ampel(score: int, green: int = 70, yellow: int = 45) -> str:
    if score >= green:
        return "🟢 Grün – Trade erlaubt"
    elif score >= yellow:
        return "🟡 Gelb – Beobachten"
    else:
        return "🔴 Rot – Kein Trade"
