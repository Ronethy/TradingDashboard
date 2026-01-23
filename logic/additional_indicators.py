import pandas as pd
import numpy as np

def rsi_divergence(df, rsi_period=14, lookback=30):
    """
    RSI-Divergenz-Erkennung (letzte lookback Bars)
    Gibt 'Bullish', 'Bearish' oder 'Keine' zurück.
    """
    if len(df) < lookback + rsi_period * 2:
        return "Keine (zu wenig Daten)"

    # Kopie machen & MultiIndex entfernen
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=1, drop=True)  # Symbol-Level droppen

    # Sicherstellen: chronologisch sortiert
    df = df.sort_index()

    # RSI sicher berechnen
    close = df['close']
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)  # Division durch 0 vermeiden
    df['rsi'] = 100 - (100 / (1 + rs))

    # Slices bilden
    recent_slice = df.iloc[-lookback:]
    prev_slice = df.iloc[-lookback*2:-lookback]

    if len(recent_slice) < 2 or len(prev_slice) < 2:
        return "Keine"

    # Tiefs finden
    recent_low_idx = recent_slice['low'].idxmin()
    recent_low_price = recent_slice['low'].min()
    recent_low_rsi = recent_slice.loc[recent_low_idx, 'rsi']

    prev_low_idx = prev_slice['low'].idxmin()
    prev_low_price = prev_slice['low'].min()
    prev_low_rsi = prev_slice.loc[prev_low_idx, 'rsi']

    if pd.isna(recent_low_rsi) or pd.isna(prev_low_rsi):
        return "Keine (NaN in RSI)"

    if recent_low_price < prev_low_price and recent_low_rsi > prev_low_rsi:
        return "Bullish Divergenz (potenzieller Boden)"
    if recent_low_price > prev_low_price and recent_low_rsi < prev_low_rsi:
        return "Bearish Divergenz (potenzieller Top)"

    return "Keine klare Divergenz"


def macd_info(df, fast=12, slow=26, signal=9):
    """
    MACD + Signal + Histogram + Interpretation
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

    if pd.isna(current_macd) or pd.isna(current_signal):
        return {"MACD": None, "Signal": None, "Histogramm": None, "Interpretation": "NaN-Werte in MACD"}

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
