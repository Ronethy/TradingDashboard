from logic.indicators import ema, rsi

def calculate_trend_score(df):
    if len(df) < 50:
        return 0

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi"] = rsi(df["close"])

    score = 0

    if df["ema20"].iloc[-1] > df["ema50"].iloc[-1]:
        score += 40

    if df["close"].iloc[-1] > df["ema20"].iloc[-1]:
        score += 30

    if 50 < df["rsi"].iloc[-1] < 70:
        score += 30

    return score
