import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta

# Indikator-Funktionen
from logic.indicators import ema, rsi, atr
from logic.additional_indicators import rsi_divergence, macd_info

# ────────────────────────────────────────────────────────────────────────────────
# Markt- & Makro-Kontext (maximal robust)
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=900)  # 15 Min Cache
def get_market_context():
    market_data = {
        "vix_level": None,
        "vix_trend": None,
        "vix_category": None,
        "sp500_trend": "N/A",
        "nasdaq_trend": "N/A",
        "adv_dec_proxy": "N/A",
        "new_highs_lows": "N/A",
        "macro_events": [
            "Fed-Zinsentscheid: 4.–5. Februar 2026",
            "CPI/PPI: 11. Februar 2026",
            "Non-Farm Payrolls: 6. Februar 2026"
        ]
    }

    try:
        vix = yf.download("^VIX", period="5d", progress=False, timeout=15)
        if vix.empty:
            vix = yf.download("VIX", period="5d", progress=False, timeout=15)
        if not vix.empty and len(vix) >= 2:
            vix_level = float(vix['Close'].iloc[-1])
            vix_change = vix['Close'].pct_change().iloc[-1]
            market_data["vix_level"] = vix_level
            market_data["vix_trend"] = "steigend" if vix_change > 0 else "fallend" if vix_change < 0 else "seitwärts"
            market_data["vix_category"] = "<15 (niedrig)" if vix_level < 15 else "15-20 (mittel)" if vix_level <= 20 else ">20 (hoch)"
    except Exception as e:
        st.caption(f"VIX-Laden fehlgeschlagen: {str(e)}")

    try:
        sp500 = yf.download("^GSPC", period="6mo", progress=False, timeout=15)
        if not sp500.empty:
            ema20 = sp500['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = sp500['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
            market_data["sp500_trend"] = "über EMA20/50" if sp500['Close'].iloc[-1] > ema20 and sp500['Close'].iloc[-1] > ema50 else "unter EMA20/50"
    except:
        pass

    try:
        nasdaq = yf.download("^IXIC", period="6mo", progress=False, timeout=15)
        if not nasdaq.empty:
            ema20 = nasdaq['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = nasdaq['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
            market_data["nasdaq_trend"] = "über EMA20/50" if nasdaq['Close'].iloc[-1] > ema20 and nasdaq['Close'].iloc[-1] > ema50 else "unter EMA20/50"
    except:
        pass

    # Advance/Decline Proxy – NaN-sicher & robust
    adv_dec_proxy = "N/A"
    if 'sp500' in locals() and not sp500.empty and len(sp500) >= 2:
        try:
            pct_changes = sp500['Close'].pct_change().dropna()
            if not pct_changes.empty:
                mean_pct = float(pct_changes.mean())  # explizit float erzwingen
                if pd.isna(mean_pct):
                    adv_dec_proxy = "neutral (Daten unklar)"
                else:
                    adv_dec_proxy = "positiv" if mean_pct > 0 else "negativ" if mean_pct < 0 else "neutral"
        except Exception as e:
            adv_dec_proxy = f"Fehler ({str(e)})"

    market_data["adv_dec_proxy"] = adv_dec_proxy

    # New Highs vs. Lows Proxy
    new_highs_lows = "N/A"
    if 'sp500' in locals() and not sp500.empty and len(sp500) >= 252:
        try:
            rolling_max = sp500['Close'].rolling(252).max()
            rolling_min = sp500['Close'].rolling(252).min()
            highs_count = (sp500['Close'] == rolling_max).sum()
            lows_count = (sp500['Close'] == rolling_min).sum()
            new_highs_lows = "mehr Highs" if highs_count > lows_count else "mehr Lows" if lows_count > highs_count else "ausgeglichen"
        except:
            pass

    market_data["new_highs_lows"] = new_highs_lows

    return market_data

# ────────────────────────────────────────────────────────────────────────────────
# Fundamental-Daten
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def get_stock_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = round(info.get('marketCap', 0) / 1e9, 2) if info.get('marketCap') else None
        beta = info.get('beta', None)
        sector = info.get('sector', 'N/A')
        short_interest = round(info.get('shortPercentOfFloat', 0) * 100, 2) if info.get('shortPercentOfFloat') else None
        free_float = info.get('floatShares', None)
        days_to_cover = info.get('shortRatio', None)
        return market_cap, beta, sector, short_interest, free_float, days_to_cover
    except:
        return None, None, 'N/A', None, None, None

# ────────────────────────────────────────────────────────────────────────────────
# Earnings-Info
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_earnings_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.earnings_dates
        if earnings is not None and not earnings.empty:
            next_earnings = earnings.index[0].strftime('%Y-%m-%d') if len(earnings) > 0 else 'N/A'
            surprises = earnings['Surprise(%)'].dropna()
            avg_move = round(surprises.mean(), 2) if not surprises.empty else 'N/A'
        else:
            q_earnings = stock.quarterly_earnings
            if not q_earnings.empty:
                avg_move = round(q_earnings['Earnings'].pct_change().mean() * 100, 2)
            else:
                avg_move = 'N/A'
            next_earnings = 'N/A (kein Datum gefunden)'

        guidance = "positiv" if isinstance(avg_move, (int, float)) and avg_move > 0 else "negativ" if isinstance(avg_move, (int, float)) and avg_move < 0 else "neutral"
        return next_earnings, avg_move, guidance
    except:
        return 'N/A', 'N/A', 'N/A'

# ────────────────────────────────────────────────────────────────────────────────
# Volumen & Struktur
# ────────────────────────────────────────────────────────────────────────────────
def get_volume_structure(df):
    if df.empty:
        return None, None, None
    avg_volume = df['volume'].mean()
    latest_volume = df['volume'].iloc[-1]
    rvol = round(latest_volume / avg_volume, 2) if avg_volume > 0 else None

    df['breakout'] = (df['close'] > df['high'].shift(1)) & (df['volume'] > avg_volume * 1.5)
    breakout_vol = "hoch" if df['breakout'].any() else "normal/niedrig"

    pullback_mask = df['close'] < df['close'].shift(1)
    pullback_vol = "niedrig" if pullback_mask.any() and df['volume'][pullback_mask].mean() < avg_volume else "hoch/normal"

    return rvol, breakout_vol, pullback_vol

def get_vwap(df):
    if df.empty:
        return None
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    pv = typical_price * df['volume']
    vwap = pv.cumsum() / df['volume'].cumsum()
    return round(vwap.iloc[-1], 2)

def get_gap_levels(df):
    if len(df) < 2:
        return None
    gaps = df['open'] - df['close'].shift(1)
    if 'ATR' not in df.columns:
        df['ATR'] = atr(df)
    atr_mean = df['ATR'].mean() if not df['ATR'].empty else 0
    significant_gaps = gaps[gaps.abs() > atr_mean]
    return round(significant_gaps.iloc[-1], 2) if not significant_gaps.empty else None

# ────────────────────────────────────────────────────────────────────────────────
# Sektor-Stärke
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def get_sector_strength(sector):
    try:
        sector_etf_map = {
            'Technology': '^IXIC',
            'Consumer Cyclical': 'XLY',
            'Consumer Defensive': 'XLP',
            'Financial Services': 'XLF',
        }
        etf = sector_etf_map.get(sector, '^GSPC')
        sector_data = yf.download(etf, period="1mo", progress=False, timeout=10)
        sector_return = sector_data['Close'].pct_change().mean() * 100 if not sector_data.empty else None

        sp500_data = yf.download("^GSPC", period="1mo", progress=False, timeout=10)
        sp500_return = sp500_data['Close'].pct_change().mean() * 100 if not sp500_data.empty else None

        if sector_return is None or sp500_return is None:
            return "N/A"
        if sector_return > sp500_return + 0.5:
            return "stark (outperforms S&P)"
        elif sector_return < sp500_return - 0.5:
            return "schwach (underperforms S&P)"
        else:
            return "gleichlaufend"
    except:
        return "N/A"

# ────────────────────────────────────────────────────────────────────────────────
# Risiko & Positionsgröße
# ────────────────────────────────────────────────────────────────────────────────
def get_risk_management(snap, capital=100000, risk_per_trade=0.01):
    if snap.atr == 0 or snap.atr is None:
        return None, None
    position_size = (capital * risk_per_trade) / snap.atr
    rr = 2.0
    return round(position_size, 0), rr

# ────────────────────────────────────────────────────────────────────────────────
# Erweiterte Option-Bias
# ────────────────────────────────────────────────────────────────────────────────
def get_extended_option_bias(snap, score, vix_level):
    if score >= 70:
        vix_info = f" (VIX {vix_level:.1f} – ruhiger Markt, Calls bevorzugt)" if vix_level is not None else ""
        return f"Stark bullish – Priorisiere Calls. Suche Strikes über EMA50. RSI niedrig: Guter Einstieg. Hohes Volume bestätigt Trend{vix_info}."
    elif score >= 40:
        return "Neutral – Beobachte. Calls wenn RSI <50, Puts wenn RSI >70. Warte auf MACD-Crossover für Richtung."
    else:
        vix_info = f" (VIX {vix_level:.1f} – höhere Volatilität, Puts bevorzugt)" if vix_level is not None else ""
        return f"Bearish – Priorisiere Puts. Suche Strikes unter EMA20. Hoher RSI: Potenzieller Abverkauf. Niedriges Volume: Schwäche{vix_info}."

# ────────────────────────────────────────────────────────────────────────────────
# Erweiterter Chart – fest über ~4 Wochen Daily
# ────────────────────────────────────────────────────────────────────────────────
def get_extended_chart(ticker):
    try:
        end = datetime.now()
        start = end - timedelta(days=40)  # Puffer für Wochenenden
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
        
        if df.empty or len(df) < 5:
            st.warning(f"Keine ausreichenden 4-Wochen-Daten für {ticker} gefunden.")
            return None
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index = pd.to_datetime(df.index)

        df["ema20"] = ema(df["close"], 20)
        df["ema50"] = ema(df["close"], 50)
        df["RSI"] = rsi(df["close"])
        df["ATR"] = atr(df)

        df['bb_mid'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)

        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        div = rsi_divergence(df)
        low_points = []
        if "Bullish" in div or "Bearish" in div:
            recent_low_idx = df['low'].argmin()
            prev_low_idx = df['low'][:-30].argmin() if len(df) > 30 else None
            low_points = [recent_low_idx] if recent_low_idx is not None else []
            if prev_low_idx is not None:
                low_points.append(prev_low_idx)

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
            title=f"{ticker} – Erweiterter Chart (letzte ~4 Wochen, Daily)",
            hovermode="x unified",
            xaxis_rangeslider_visible=True,
            xaxis=dict(autorange=True),
            yaxis=dict(autorange=True)
        )

        return fig

    except Exception as e:
        st.caption(f"4-Wochen-Chart konnte nicht geladen werden: {str(e)}")
        return None

# ────────────────────────────────────────────────────────────────────────────────
# Hauptfunktion – Tab 5
# ────────────────────────────────────────────────────────────────────────────────
def show_extended_analysis(ticker, snap, score, timeframe_str=None, df=None):
    st.subheader("🧠 Erweiterte Analyse – Vollständige Datenbasis")

    if ticker is None or snap is None:
        st.warning("Nicht alle benötigten Daten verfügbar. Wähle einen Ticker.")
        return

    # 1. Markt- & Makro-Kontext
    st.markdown("**1. Markt- & Makro-Kontext**")
    market_data = get_market_context()

    vix_level = market_data.get('vix_level')
    if vix_level is not None:
        st.write(f"VIX-Level: **{vix_level:.2f}** ({market_data.get('vix_trend', 'N/A')}, {market_data.get('vix_category', 'N/A')})")
    else:
        st.write("VIX-Level: **Nicht verfügbar**")

    st.write(f"S&P 500 Trend: **{market_data.get('sp500_trend', 'N/A')}**")
    st.write(f"Nasdaq Trend: **{market_data.get('nasdaq_trend', 'N/A')}**")
    st.write(f"Advance/Decline Proxy: **{market_data.get('adv_dec_proxy', 'N/A')}**")
    st.write(f"New Highs vs. New Lows Proxy: **{market_data.get('new_highs_lows', 'N/A')}**")

    st.write("Nächste Makro-Termine:")
    for event in market_data.get('macro_events', []):
        st.write(f"- {event}")

    # 2. Fundamental-Daten
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

    # 3. Volumen & Struktur
    st.markdown("**3. Volumen & Marktstruktur**")
    if df is not None and not df.empty:
        rvol, breakout_vol, pullback_vol = get_volume_structure(df)
        vwap = get_vwap(df)
        gap_level = get_gap_levels(df)

        st.write(f"Relatives Volumen (RVOL): **{rvol:.2f}**" if rvol else "RVOL: N/A")
        st.write(f"Volumen bei Breakouts: **{breakout_vol}**" if breakout_vol else "Breakout-Volumen: N/A")
        st.write(f"Volumen bei Pullbacks: **{pullback_vol}**" if pullback_vol else "Pullback-Volumen: N/A")
        st.write(f"Daily VWAP: **{vwap:.2f}**" if vwap else "VWAP: N/A")
        st.write(f"Gap-Level: **{gap_level:.2f}**" if gap_level else "Gap-Level: N/A")
    else:
        st.write("Volumen-Daten: **Nicht verfügbar** (df fehlt)")

    # 4. Risiko- & Trade-Planung
    st.markdown("**4. Risiko- & Trade-Planung**")
    position_size, rr = get_risk_management(snap)
    if position_size:
        st.write(f"Vorgeschlagene Positionsgröße (1% Risiko, 100k Kapital): **{position_size} Aktien**")
        st.write(f"Minimum R:R: **{rr}:1**")
    else:
        st.write("Positionsgröße: **Nicht berechenbar** (ATR = 0)")

    # 5. Erweiterte Options-Empfehlung
    st.markdown("**5. Erweiterte Options-Empfehlung**")
    extended_bias = get_extended_option_bias(snap, score, market_data.get('vix_level'))
    st.markdown(extended_bias)

    # 6. Erweiterter Chart – fest über ~4 Wochen Daily
    st.markdown("**6. Erweiterter Chart (letzte ~4 Wochen, Daily)**")
    fig = get_extended_chart(ticker)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Erweiterter Chart konnte nicht erstellt werden.")
