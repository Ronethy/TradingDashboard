import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta

# Wichtig: Imports der Indikator-Funktionen aus deinem bestehenden Projekt
from logic.indicators import ema, rsi, atr
from logic.additional_indicators import rsi_divergence, macd_info

# ────────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen für erweiterte Markt- & Aktien-Daten
# ────────────────────────────────────────────────────────────────────────────────

def get_market_context():
    """Lädt VIX, S&P 500 und Nasdaq Trend + Advance/Decline-Proxy"""
    try:
        # VIX
        vix = yf.download("^VIX", period="5d", progress=False)
        vix_level = vix['Close'].iloc[-1] if not vix.empty else None
        vix_trend = "steigend" if vix['Close'].pct_change().iloc[-1] > 0 else "fallend" if vix['Close'].pct_change().iloc[-1] < 0 else "seitwärts"
        vix_category = "<15 (niedrig)" if vix_level < 15 else "15-20 (mittel)" if vix_level <= 20 else ">20 (hoch)"

        # S&P 500 Trend
        sp500 = yf.download("^GSPC", period="6mo", progress=False)
        sp500_trend = "über EMA20/50" if not sp500.empty and sp500['Close'].iloc[-1] > sp500['Close'].ewm(span=20).mean().iloc[-1] and sp500['Close'].iloc[-1] > sp500['Close'].ewm(span=50).mean().iloc[-1] else "unter EMA20/50"

        # Nasdaq Trend
        nasdaq = yf.download("^IXIC", period="6mo", progress=False)
        nasdaq_trend = "über EMA20/50" if not nasdaq.empty and nasdaq['Close'].iloc[-1] > nasdaq['Close'].ewm(span=20).mean().iloc[-1] and nasdaq['Close'].iloc[-1] > nasdaq['Close'].ewm(span=50).mean().iloc[-1] else "unter EMA20/50"

        # Advance/Decline Proxy (über S&P 500 Up/Down Volume)
        adv_dec_proxy = "positiv" if not sp500.empty and sp500['Close'].pct_change().mean() > 0 else "negativ"

        # New Highs vs. New Lows Proxy (52-Wochen-High/Low Anteil)
        new_highs_lows = "mehr Highs" if not sp500.empty and (sp500['Close'] == sp500['Close'].rolling(252).max()).sum() > (sp500['Close'] == sp500['Close'].rolling(252).min()).sum() else "mehr Lows"

        # Makro-Termine (statisch – erweitere bei Bedarf)
        macro_events = [
            "Fed-Zinsentscheid: 4.–5. Februar 2026",
            "CPI/PPI: 11. Februar 2026",
            "Non-Farm Payrolls: 6. Februar 2026"
        ]

        return {
            "vix_level": vix_level,
            "vix_trend": vix_trend,
            "vix_category": vix_category,
            "sp500_trend": sp500_trend,
            "nasdaq_trend": nasdaq_trend,
            "adv_dec_proxy": adv_dec_proxy,
            "new_highs_lows": new_highs_lows,
            "macro_events": macro_events
        }
    except Exception:
        return {}

def get_stock_fundamentals(ticker):
    """Holt Market Cap, Beta, Sektor, Short Interest, Free Float, Days to Cover"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = round(info.get('marketCap', None) / 1e9, 2) if info.get('marketCap') else None
        beta = info.get('beta', None)
        sector = info.get('sector', 'N/A')
        short_interest = round(info.get('shortPercentOfFloat', None) * 100, 2) if info.get('shortPercentOfFloat') else None
        free_float = info.get('floatShares', None)
        days_to_cover = info.get('shortRatio', None)
        return market_cap, beta, sector, short_interest, free_float, days_to_cover
    except:
        return None, None, 'N/A', None, None, None

def get_earnings_info(ticker):
    """Holt Earnings-Datum + Average Move %, Guidance-Proxy"""
    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        earnings_date = calendar['Earnings Date'][0].strftime('%Y-%m-%d') if 'Earnings Date' in calendar and isinstance(calendar['Earnings Date'], list) else 'N/A'

        # Historische Earnings-Moves
        hist = stock.earnings_dates
        avg_move = round(hist['Surprise(%)'].mean(), 2) if not hist.empty else 'N/A'

        # Guidance-Proxy (letzte Earnings-Reaktion)
        guidance = "positiv (basierend auf letzter Reaktion)" if avg_move > 0 else "negativ" if avg_move < 0 else "neutral"

        return earnings_date, avg_move, guidance
    except:
        return 'N/A', 'N/A', 'N/A'

def get_volume_structure(df):
    """Berechnet RVOL, Volumen bei Breakouts/Pullbacks (einfach)"""
    if df.empty:
        return None, None, None
    avg_volume = df['volume'].mean()
    latest_volume = df['volume'].iloc[-1]
    rvol = latest_volume / avg_volume if avg_volume > 0 else None

    # Volumen bei Breakouts/Pullbacks Proxy
    df['breakout'] = (df['close'] > df['high'].shift(1)) & (df['volume'] > avg_volume)
    breakout_vol = "hoch" if df['breakout'].any() else "niedrig"
    pullback_vol = "niedrig" if df['volume'][df['close'] < df['close'].shift(1)].mean() < avg_volume else "hoch"

    return rvol, breakout_vol, pullback_vol

def get_vwap(df):
    """Einfacher Daily VWAP"""
    if df.empty:
        return None
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    pv = typical_price * df['volume']
    vwap = pv.cumsum() / df['volume'].cumsum()
    return vwap.iloc[-1]

def get_gap_levels(df):
    """Erkennt Gap-Levels"""
    if len(df) < 2:
        return None
    gaps = df['open'] - df['close'].shift(1)
    gap_levels = gaps[gaps.abs() > df['atr'].mean()]  # Gaps > ATR
    return gap_levels.iloc[-1] if not gap_levels.empty else None

def get_sector_strength(sector):
    """Einfacher Sektor-Vergleich (vs S&P 500)"""
    try:
        sector_etf_map = {
            'Technology': '^IXIC',  # Nasdaq for Tech
            'Consumer Cyclical': '^DJUSCY',
            # Füge weitere hinzu
        }
        etf = sector_etf_map.get(sector, '^GSPC')  # Fallback S&P
        sector_data = yf.download(etf, period="1mo", progress=False)
        sector_return = sector_data['Close'].pct_change().mean() * 100 if not sector_data.empty else None
        sp500_data = yf.download("^GSPC", period="1mo", progress=False)
        sp500_return = sp500_data['Close'].pct_change().mean() * 100 if not sp500_data.empty else None
        strength = "stark (outperforms S&P)" if sector_return > sp500_return else "schwach (underperforms S&P)"
        return strength
    except:
        return "N/A"

def get_risk_management(snap, capital=100000, risk_per_trade=0.01):
    """Berechnet Positionsgröße, R:R Proxy"""
    if snap.atr == 0 or snap.atr is None:
        return None, None
    position_size = (capital * risk_per_trade) / snap.atr
    rr = 2.0  # Minimum R:R
    return round(position_size, 0), rr

def get_extended_option_bias(snap, score, vix_level):
    """Erweiterte Option-Bias"""
    # Wie vorher, erweitert um VIX
    if score >= 70:
        vix_info = f" (VIX {vix_level:.1f} – ruhiger Markt, Calls bevorzugt)" if vix_level else ""
        return f"Stark bullish – Priorisiere Calls. Suche Strikes über EMA50. RSI niedrig: Guter Einstieg. Hohes Volume bestätigt Trend{vix_info}."
    elif score >= 40:
        return "Neutral – Beobachte. Calls wenn RSI <50, Puts wenn RSI >70. Warte auf MACD-Crossover für Richtung."
    else:
        vix_info = f" (VIX {vix_level:.1f} – höhere Volatilität, Puts bevorzugt)" if vix_level else ""
        return f"Bearish – Priorisiere Puts. Suche Strikes unter EMA20. Hoher RSI: Potenzieller Abverkauf. Niedriges Volume: Schwäche{vix_info}."

def get_extended_chart(df, ticker, timeframe_str):
    # Wie vorher – unverändert
    pass  # Dein bestehender Chart-Code hier

# Hauptfunktion für den Tab
def show_extended_analysis(ticker, snap, score, timeframe_str, df):
    st.subheader("🧠 Erweiterte Analyse – Vollständige Datenbasis")

    if ticker is None or snap is None or df.empty:
        st.warning("Nicht alle benötigten Daten verfügbar. Wähle einen Ticker und warte auf Chart-Laden.")
        return

    # 1. Markt- & Makro-Kontext
    st.markdown("**1. Markt- & Makro-Kontext**")
    market_data = get_market_context()

    st.write(f"VIX-Level: **{market_data.get('vix_level', 'N/A'):.2f}** ({market_data.get('vix_trend', 'N/A')}, {market_data.get('vix_category', 'N/A')})")
    st.write(f"S&P 500 Trend: **{market_data.get('sp500_trend', 'N/A')}**")
    st.write(f"Nasdaq Trend: **{market_data.get('nasdaq_trend', 'N/A')}**")
    st.write(f"Advance/Decline Proxy: **{market_data.get('adv_dec_proxy', 'N/A')}**")
    st.write(f"New Highs vs. New Lows Proxy: **{market_data.get('new_highs_lows', 'N/A')}**")
    st.write("Nächste Makro-Termine:")
    for event in market_data.get('macro_events', []):
        st.write(f"- {event}")

    # 2. Aktien-spezifische Daten – Fundamental
    st.markdown("**2. Aktien-spezifische Daten – Fundamental**")
    market_cap, beta, sector, short_interest, free_float, days_to_cover = get_stock_fundamentals(ticker)
    earnings_date, avg_move, guidance = get_earnings_info(ticker)
    sector_strength = get_sector_strength(sector)

    st.write(f"Market Cap: **{market_cap} Mrd. USD**" if market_cap else "Market Cap: N/A")
    st.write(f"Free Float: **{free_float} Shares**" if free_float else "Free Float: N/A")
    st.write(f"Beta: **{beta}**" if beta else "Beta: N/A")
    st.write(f"Sektor: **{sector}** (Stärke: {sector_strength})")
    st.write(f"Short Interest: **{short_interest}%**" if short_interest else "Short Interest: N/A")
    st.write(f"Days to Cover: **{days_to_cover} Tage**" if days_to_cover else "Days to Cover: N/A")
    st.write(f"Nächste Earnings: **{earnings_date}** (Guidance-Proxy: {guidance})")
    st.write(f"Durchschnittlicher Earnings-Move: **{avg_move}**")

    # 3. Volumen & Marktstruktur
    st.markdown("**3. Volumen & Marktstruktur**")
    rvol, breakout_vol, pullback_vol = get_volume_structure(df)
    vwap = get_vwap(df)
    gap_level = get_gap_levels(df)

    st.write(f"Relatives Volumen (RVOL): **{rvol:.2f}**" if rvol else "RVOL: N/A")
    st.write(f"Volumen bei Breakouts: **{breakout_vol}**" if breakout_vol else "Breakout-Volumen: N/A")
    st.write(f"Volumen bei Pullbacks: **{pullback_vol}**" if pullback_vol else "Pullback-Volumen: N/A")
    st.write(f"Daily VWAP: **{vwap:.2f}**" if vwap else "VWAP: N/A")
    st.write(f"Gap-Level: **{gap_level:.2f}**" if gap_level else "Gap-Level: N/A")

    # 4. Risiko- & Trade-Planung
    st.markdown("**4. Risiko- & Trade-Planung**")
    position_size, rr = get_risk_management(snap)
    if position_size:
        st.write(f"Vorgeschlagene Positionsgröße (1% Risiko, 100k Kapital): **{position_size} Aktien**")
        st.write(f"Minimum R:R: **{rr}:1**")
    else:
        st.write("Positionsgröße: **Nicht berechenbar** (ATR = 0)")

    # 5. Erweiterte Option-Bias
    st.markdown("**5. Erweiterte Options-Empfehlung**")
    extended_bias = get_extended_option_bias(snap, score, market_data.get('vix_level'))
    st.markdown(extended_bias)

    # 6. Erweiterter Chart mit Fibonacci
    st.markdown("**6. Erweiterter Chart mit Fibonacci**")
    fig = get_extended_chart(df, ticker, timeframe_str)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Erweiterter Chart konnte nicht erstellt werden.")
