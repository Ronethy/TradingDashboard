def get_option_bias(snapshot, score):
    if score < 60:
        return "NEUTRAL / kein Options-Trade"
    if snapshot.ema9 > snapshot.ema20 > snapshot.ema50:
        return "BULLISH → CALLs / Long-Bias"
    if snapshot.ema9 < snapshot.ema20 < snapshot.ema50:
        return "BEARISH → PUTs / Short-Bias"
    return "NEUTRAL / Spreads oder warten"
