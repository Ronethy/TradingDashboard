def rsi_divergence(df, rsi_period=14, lookback=30):
    """
    Einfache RSI-Divergenz-Erkennung (letzte lookback Bars)
    Gibt zurück: 'Bullish', 'Bearish' oder 'Keine'
    """
    if len(df) < lookback + rsi_period:
        return "Keine"

    df = df.copy()
    # RSI berechnen (falls nicht vorhanden)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Letzte lookback Bars
    recent_slice = df.iloc[-lookback:]
    prev_slice = df.iloc[-lookback*2:-lookback]

    if len(recent_slice) < 2 or len(prev_slice) < 2:
        return "Keine"

    # Preis-Tiefs finden
    recent_low_idx = recent_slice['low'].idxmin()
    recent_low_price = recent_slice['low'].min()
    recent_low_rsi = recent_slice.loc[recent_low_idx, 'rsi']

    prev_low_idx = prev_slice['low'].idxmin()
    prev_low_price = prev_slice['low'].min()
    prev_low_rsi = prev_slice.loc[prev_low_idx, 'rsi']

    if recent_low_price < prev_low_price and recent_low_rsi > prev_low_rsi:
        return "Bullish Divergenz (potenzieller Boden)"
    if recent_low_price > prev_low_price and recent_low_rsi < prev_low_rsi:
        return "Bearish Divergenz (potenzieller Top)"

    return "Keine klare Divergenz"
