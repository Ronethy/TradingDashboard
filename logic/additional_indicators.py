import pandas as pd

def rsi_divergence(df, rsi_period=14, lookback=30):
    """
    Einfache RSI-Divergenz-Erkennung (letzte lookback Bars)
    Gibt zurück: 'Bullish', 'Bearish' oder 'Keine'
    """
    if len(df) < lookback + rsi_period:
        return "Keine"

    df = df.copy()
    df['rsi'] = pd.Series(  # RSI neu berechnen, falls nicht vorhanden
        100 - (100 / (1 + (df['close'].diff().clip(lower=0).rolling(rsi_period).mean() /
                       -df['close'].diff().clip(upper=0).rolling(rsi_period).mean())))
    )

    recent_low_price = df['low'].iloc[-lookback:].min()
    recent_low_rsi = df['rsi'].iloc[df['low'].iloc[-lookback:].idxmin()]

    prev_low_price = df['low'].iloc[-lookback*2:-lookback].min()
    prev_low_rsi = df['rsi'].iloc[df['low'].iloc[-lookback*2:-lookback].idxmin()]

    if recent_low_price < prev_low_price and recent_low_rsi > prev_low_rsi:
        return "Bullish Divergenz (potenzieller Boden)"
    if recent_low_price > prev_low_price and recent_low_rsi < prev_low_rsi:
        return "Bearish Divergenz (potenzieller Top)"

    return "Keine klare Divergenz"


def macd_info(df, fast=12, slow=26, signal=9):
    """
    Berechnet MACD + Signal + Histogram
    Gibt Dict mit aktuellen Werten + Interpretation
    """
    if len(df) < slow + signal:
        return {"macd": None, "signal": None, "hist": None, "text": "Nicht genug Daten"}

    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    current_macd = macd_line.iloc[-1]
    current_signal = signal_line.iloc[-1]
    current_hist = histogram.iloc[-1]

    if current_macd > current_signal and current_hist > 0:
        text = "Bullish (MACD über Signal + positives Histogramm)"
    elif current_macd < current_signal and current_hist < 0:
        text = "Bearish (MACD unter Signal + negatives Histogramm)"
    else:
        text = "Neutral / Übergangsphase"

    return {
        "MACD": round(current_macd, 4),
        "Signal": round(current_signal, 4),
        "Histogramm": round(current_hist, 4),
        "Interpretation": text
    }
