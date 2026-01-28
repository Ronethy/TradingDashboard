import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf  # Für VIX, Market Cap, Beta, etc.
from datetime import datetime, timedelta

# Funktion für erweiterte Markt-Kontext (VIX, S&P Trend, etc.)
def get_market_context():
    try:
        # VIX holen
        vix = yf.download("^VIX", period="1d")
        vix_level = vix['Close'].iloc[-1]
        vix_trend = "steigend" if vix['Close'].pct_change().iloc[-1] > 0 else "fallend"
        vix_category = "<15 (niedrig)" if vix_level < 15 else "15-20 (mittel)" if vix_level < 20 else ">20 (hoch)"

        # S&P 500 Trend
        sp500 = yf.download("^GSPC", period="6mo")
        sp500['EMA20'] = sp500['Close'].ewm(span=20, adjust=False).mean()
        sp500['EMA50'] = sp500['Close'].ewm(span=50, adjust=False).mean()
        sp500_trend = "über EMA20/50" if sp500['Close'].iloc[-1] > sp500['EMA20'].iloc[-1] and sp500['Close'].iloc[-1] > sp500['EMA50'].iloc[-1] else "unter EMA20/50"

        # Makro-Termine (statisch – erweitere bei Bedarf)
        macro_events = [
            "Fed-Zinsentscheid: 4.–5. Februar 2026",
            "CPI/PPI: 11. Februar 2026",
            "Non-Farm Payrolls: 6. Februar 2026"
        ]

        return vix_level, vix_trend, vix_category, sp500_trend, macro_events
    except:
        return None, None, None, None, []

# Funktion für erweiterte Aktien-Daten (Beta, Market Cap, etc.)
def get_stock_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = info.get('marketCap', 'N/A') / 1e9 if info.get('marketCap') else 'N/A'
        beta = info.get('beta', 'N/A')
        sector = info.get('sector', 'N/A')
        short_interest = info.get('shortPercentOfFloat', 'N/A') * 100 if info.get('shortPercentOfFloat') else 'N/A'
        return market_cap, beta, sector, short_interest
    except:
        return 'N/A', 'N/A', 'N/A', 'N/A'

# Funktion für Volumen & Struktur (RVOL)
def get_volume_structure(df):
    if df.empty:
        return 'N/A'
    avg_volume = df['volume'].mean()
    latest_volume = df['volume'].iloc[-1]
    rvol = latest_volume / avg_volume if avg_volume > 0 else 1.0
    return rvol

# Funktion für Risiko- & Trade-Planung (Positionsgröße vorschlagen)
def get_risk_management(snap, capital=100000, risk_per_trade=0.01):
    if snap.atr == 0:
        return 'N/A'
    position_size = (capital * risk_per_trade) / snap.atr
    return round(position_size, 2)

# Funktion für erweiterte Option-Bias (Call/Put-Empfehlung)
def get_extended_option_bias(snap, score, vix_level):
    if score >= 70:
        return "Stark bullish – Priorisiere Calls. Suche Strikes über EMA50. RSI niedrig: Guter Einstieg. Hohes Volume bestätigt Trend. Bei VIX <20: Gute Vega-Chance."
    elif score >= 40:
        return "Neutral – Beobachte. Calls wenn RSI <50, Puts wenn RSI >70. Warte auf MACD-Crossover. Bei VIX 15-20: Spreads bevorzugen."
    else:
        return "Bearish – Priorisiere Puts. Suche Strikes unter EMA20. Hoher RSI: Potenzieller Abverkauf. Niedriges Volume: Schwäche. Bei VIX >20: Strangles in Betracht."

# Funktion für erweiterten Chart mit Fibonacci
def get_extended_chart(df, ticker, timeframe_str):
    if df.empty:
        return None

    # Fibonacci Levels berechnen
    fib_high = df['high'].max()
    fib_low = df['low'].min()
    fib_levels = {
        '0%': fib_low,
        '23.6%': fib_low + 0.236 * (fib_high - fib_low),
        '38.2%': fib_low + 0.382 * (fib_high - fib_low),
        '50%': fib_low + 0.5 * (fib_high - fib_low),
        '61.8%': fib_low + 0.618 * (fib_high - fib_low),
        '100%': fib_high
    }

    # Basis-Indikatoren (aus bestehendem Code)
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["RSI"] = rsi(df["close"])
    df["ATR"] = atr(df)

    # Bollinger Bands (aus bestehendem Code)
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)

    # MACD (aus bestehendem Code)
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    # RSI Divergenz (aus bestehendem Code)
    div = rsi_divergence(df)
    low_points = []
    if "Bullish" in div or "Bearish" in div:
        recent_low_idx = df['low'].argmin()
        prev_low_idx = df['low'][:-30].argmin() if len(df) > 30 else None
        low_points = [recent_low_idx] if recent_low_idx is not None else []
        if prev_low_idx is not None:
            low_points.append(prev_low_idx)

    # Chart aufbauen (basierend auf bestehendem)
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )

    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="OHLC"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], name="EMA20", line=dict(color="#00BFFF")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], name="EMA50", line=dict(color="#FF8C00")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper", line=dict(color="rgba(255,0,0,0.5)", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower", line=dict(color="rgba(0,255,0,0.5)", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_mid"], name="BB Mid", line=dict(color="rgba(128,128,128,0.7)")), row=1, col=1)

    # Fibonacci Levels als hlines hinzufügen
    for level, value in fib_levels.items():
        fig.add_hline(y=value, line_dash="dot", line_color="purple", annotation_text=level, row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD", line=dict(color="blue")), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal_line, name="Signal", line=dict(color="orange")), row=4, col=1)
    fig.add_trace(go.Bar(x=df.index, y=histogram, name="Histogram", marker_color="grey"), row=4, col=1)

    for idx in low_points:
        fig.add_annotation(
            x=df.index[idx],
            y=df['low'].iloc[idx],
            text="Low",
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red" if "Bearish" in div else "green",
            row=1, col=1
        )

    fig.update_layout(
        height=700,
        title=f"{ticker} – {timeframe_str} ({len(df)} Kerzen)",
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
        xaxis=dict(
            autorange=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1 Tag", step="day", stepmode="backward"),
                    dict(count=3, label="3 Tage", step="day", stepmode="backward"),
                    dict(count=7, label="1 Woche", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ),
        yaxis=dict(autorange=True)
    )

    return fig

# Funktion für den gesamten Inhalt des neuen Tabs
def show_extended_analysis(ticker, snap, score, timeframe_str, df):
    st.subheader("🧠 Erweiterte Analyse – Vollständige Datenbasis")

    # 1. Markt- & Makro-Kontext
    vix_level, vix_trend, vix_category, sp500_trend, macro_events = get_market_context()
    st.markdown("**1. Markt- & Makro-Kontext**")
    st.write(f"VIX-Level: {vix_level:.2f} ({vix_trend}, Kategorie: {vix_category})")
    st.write(f"S&P 500 Trend: {sp500_trend}")
    st.write("Makro-Termine:")
    for event in macro_events:
        st.write(f"- {event}")

    # 2. Aktien-spezifische Daten – Technisch
    st.markdown("**2. Aktien-spezifische Daten – Technisch**")
    st.write("Trend & Struktur: Multi-Timeframe-Analyse (aus Chart)")

    # 3. Aktien-spezifische Daten – Fundamental
    st.markdown("**3. Aktien-spezifische Daten – Fundamental**")
    market_cap, beta, sector, short_interest = get_stock_fundamentals(ticker)
    st.write(f"Market Cap: {market_cap} Mrd. USD")
    st.write(f"Beta: {beta}")
    st.write(f"Sektor: {sector}")
    st.write(f"Short Interest: {short_interest}%")

    # 4. Volumen & Marktstruktur
    st.markdown("**4. Volumen & Marktstruktur**")
    rvol = get_volume_structure(df)
    st.write(f"Relatives Volumen (RVOL): {rvol:.2f}")

    # 5. Risiko- & Trade-Planung
    st.markdown("**5. Risiko- & Trade-Planung**")
    position_size = get_risk_management(snap)
    st.write(f"Vorgeschlagene Positionsgröße (bei 100k Kapital, 1% Risiko): {position_size} Aktien")

    # 6. Options-spezifische Daten
    st.markdown("**6. Options-spezifische Daten**")
    # IV simulieren (in echt mit alpaca-py holen)
    iv = 25.0  # Platzhalter – erweitere mit realen Daten
    st.write(f"Implizite Volatilität (IV): {iv}% (Percentile: 40%)")

    # 7. Strategie-Entscheidung
    st.markdown("**7. Strategie-Entscheidung**")
    extended_bias = get_extended_option_bias(snap, score, vix_level)
    st.write(extended_bias)

    # Erweiterter Chart mit Fibonacci
    fig = get_extended_chart(df, ticker, timeframe_str)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
