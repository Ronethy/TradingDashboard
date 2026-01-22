def option_bias(trend_score):
    if trend_score >= 70:
        return "Bullish Calls"
    if trend_score >= 40:
        return "Neutral / Spreads"
    return "Bearish / Puts"
