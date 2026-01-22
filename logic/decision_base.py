def score_to_ampel(score: int, green: int, yellow: int):
    if score >= green:
        return "🟢 Grün – Trade erlaubt"
    elif score >= yellow:
        return "🟡 Gelb – Beobachten"
    else:
        return "🔴 Rot – Kein Trade"
