def option_bias(snapshot, score):
    if score < 60:
        return "NEUTRAL"

    if snapshot.ema9 > snapshot.ema20 > snapshot.ema50:
        return "CALL"

    if snapshot.ema9 < snapshot.ema20 < snapshot.ema50:
        return "PUT"

    return "NEUTRAL"
