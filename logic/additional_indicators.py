import pandas as pd

def rsi_divergence(df, rsi_period=14, lookback=30):
    """
    Einfache RSI-Divergenz-Erkennung (letzte lookback Bars)
    Gibt zurück: 'Bullish', 'Bearish' oder 'Keine'
    """
    if len(df) < lookback + rsi_period * 2:
        return "Keine (zu wenig Daten)"

    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=1, drop=True)  # Symbol-Level entfernen, falls vorhanden

    # RSI berechnen
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Slices bilden – nach Index sortieren
    df = df.sort_index()
    recent_slice = df.iloc[-lookback:]
    prev_slice = df.iloc[-lookback*2:-lookback]

    if len(recent_slice) < 2 or len(prev_slice) < 2:
        return "Keine"

    # Preis- und RSI-Tiefs finden
    recent_low_price = recent_slice['low'].min()
    recent_low_idx = recent_slice['low'].idxmin()
    recent_low_rsi = recent_slice.loc[recent_low_idx, 'rsi']

    prev_low_price = prev_slice['low'].min()
    prev_low_idx = prev_slice['low'].idxmin()
    prev_low_rsi = prev_slice.loc[prev_low_idx, 'rsi']

    if recent_low_price < prev_low_price and recent_low_rsi > prev_low_rsi:
        return "Bullish Divergenz (potenzieller Boden)"
    if recent_low_price > prev_low_price and recent_low_rsi < prev_low_rsi:
        return "Bearish Divergenz (potenzieller Top)"

    return "Keine klare Divergenz"


def macd_info(df, fast=12, slow=26, signal=9):
    """
    Berechnet MACD + Signal + Histogram
    Gibt Dict mit aktuellen Werten + Interpretation zurück
    """
    if len(df) < slow + signal:
        return {"MACD": None, "Signal": None, "Histogramm": None, "Interpretation": "Nicht genug Daten"}

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
