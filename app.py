import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime, timedelta
import requests

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from data.sp500_symbols import SP500_SYMBOLS
from logic.indicators import ema, rsi, atr
from logic.snapshot import MarketSnapshot
from logic.trend_score import calculate_trend_score
from logic.option_bias import get_option_bias
from logic.trade_plan import generate_trade_plan
from logic.decision_daytrade import decide_daytrade
from logic.decision_swing import decide_swing
from logic.premarket_scanner import scan_early_movers
from logic.decision_base import score_to_ampel

st.set_page_config(page_title="Momentum Dashboard", layout="wide")

client = StockHistoricalDataClient(
    st.secrets["ALPACA_API_KEY"],
    st.secrets["ALPACA_SECRET_KEY"]
)

alpha_vantage_key = st.secrets.get("ALPHA_VANTAGE_KEY", None)
if not alpha_vantage_key:
    st.warning("Alpha Vantage Key fehlt → News deaktiviert (füge in Secrets hinzu)")

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"

ny_tz = pytz.timezone("America/New_York")
now_ny = datetime.now(ny_tz)
market_state = "PRE" if now_ny.hour < 9 else "OPEN" if now_ny.hour < 16 else "CLOSED"

st.title("📊 Smart Momentum Trading Dashboard")
st.write(f"Marktstatus: {market_state} | Zeit: {now_ny.strftime('%Y-%m-%d %H:%M')}")

if market_state == "PRE":
    st.warning("Pre-Market: Viele Scores & Ampele sind eingeschränkt – warte auf Open (9:30 ET)")

if st.button("Daten aktualisieren (Cache leeren)"):
    st.cache_data.clear()
    st.rerun()

@st.cache_data(ttl=60)
def load_daily_data(symbols):
    data = {}
    batch_size = 80
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=now_ny - timedelta(days=150),
                end=now_ny + timedelta(days=1),
                feed="iex"
            )
            bars = client.get_stock_bars(req).df
            for sym in batch:
                try:
                    df_sym = bars[bars.index.get_level_values('symbol') == sym].copy()
                    if not df_sym.empty:
                        data[sym] = df_sym
                except:
                    pass
        except Exception as e:
            st.caption(f"Batch-Fehler: {str(e)}")
    return data

@st.cache_data(ttl=60)
def load_intraday(ticker):
    try:
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=now_ny - timedelta(days=2),
            end=now_ny + timedelta(minutes=30),
            feed="iex",
            limit=10000
        )
        bars = client.get_stock_bars(req).df

        if bars.empty:
            return pd.DataFrame()

        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index(level=1, drop=True)

        bars.index = bars.index.tz_convert(ny_tz)

        return bars

    except Exception as e:
        st.caption(f"Intraday-Fehler {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_stock_news(ticker, api_key, limit=5):
    if not api_key:
        return []
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit={limit}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("feed", [])[:limit]
        else:
            return []
    except:
        return []

daily_data = load_daily_data(SP500_SYMBOLS)

st.caption(f"Geladene Symbole: {len(daily_data)} / {len(SP500_SYMBOLS)}")

tabs = st.tabs([
    "🔥 Early Movers",
    "🧠 S&P 500 Scanner",
    "📈 Chart Analyse",
    "🟢 Trading-Entscheidung"
])

# ── Early Movers ────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("🔥 Early Movers – verbessert")
    st.caption("Gap + Volumen + Score + News-Sentiment (nur Top-Kandidaten)")

    enhanced_movers = []
    for sym, df in daily_data.items():
        if len(df) < 2:
            continue

        df = df.sort_index()
        prev_close = df["close"].iloc[-2]
        current_open = df["open"].iloc[-1]
        current_close = df["close"].iloc[-1]

        gap_pct = (current_open - prev_close) / prev_close * 100
        abs_gap = abs(gap_pct)
        volume = df["volume"].iloc[-1]
        vol_avg = df["volume"].mean()
        vol_ratio = volume / vol_avg if vol_avg > 0 else 1.0
        one_day_perf = (current_close - prev_close) / prev_close * 100

        # Snapshot für Score
        df_ind = df.copy()
        df_ind["ema9"] = ema(df_ind["close"], 9)
        df_ind["ema20"] = ema(df_ind["close"], 20)
        df_ind["ema50"] = ema(df_ind["close"], 50)
        df_ind["rsi"] = rsi(df_ind["close"])
        df_ind["atr"] = atr(df_ind)
        df_ind.dropna(inplace=True)

        score = 0
        if not df_ind.empty:
            latest = df_ind.iloc[-1]
            vol_ratio_ind = latest["volume"] / df_ind["volume"].mean() if df_ind["volume"].mean() > 0 else 1.0
            snap = MarketSnapshot(sym, latest["close"], latest["rsi"], latest["ema9"], latest["ema20"], latest["ema50"], latest["atr"], vol_ratio_ind, market_state)
            score = calculate_trend_score(snap)

        # News nur bei starken Kandidaten laden (Score ≥ 50 oder Gap > 3%)
        news_sentiment = "N/A"
        if score >= 50 or abs_gap > 3:
            news = get_stock_news(sym, alpha_vantage_key, limit=3)
            if news:
                sentiments = [item.get("overall_sentiment_label", "Neutral") for item in news]
                if "Bullish" in sentiments:
                    news_sentiment = "Bullish"
                elif "Bearish" in sentiments:
                    news_sentiment = "Bearish"
                else:
                    news_sentiment = "Neutral"

        enhanced_movers.append({
            "Symbol": sym,
            "Gap %": round(gap_pct, 2),
            "Abs Gap": abs_gap,
            "Vol Ratio": round(vol_ratio, 2),
            "Score": score,
            "Sentiment": news_sentiment,
            "Last": round(current_close, 2)
        })

    if enhanced_movers:
        df_movers = pd.DataFrame(enhanced_movers)
        df_movers = df_movers.sort_values("Abs Gap", ascending=False).head(20)

        # Empfehlung hinzufügen
        def get_recommendation(row):
            gap = row["Gap %"]
            score = row["Score"]
            sent = row["Sentiment"]
            if score >= 70 and (sent == "Bullish" or gap > 3):
                return "Kaufen / Long priorisieren"
            elif score >= 50 or gap > 2:
                return "Beobachten / Watchlist"
            else:
                return "Vermeiden"

        df_movers["Empfehlung"] = df_movers.apply(get_recommendation, axis=1)

        # Styling
        def highlight(row):
            rec = row["Empfehlung"]
            if "Kaufen" in rec:
                return ['background-color: #d4edda'] * len(row)
            elif "Beobachten" in rec:
                return ['background-color: #fff3cd'] * len(row)
            else:
                return ['background-color: #f8d7da'] * len(row)

        styled = df_movers.style.apply(highlight, axis=1)

        st.dataframe(styled, width='stretch', hide_index=True)

        selected = st.selectbox("Zu Detail springen:", ["—"] + df_movers["Symbol"].tolist())
        if selected != "—":
            st.session_state.selected_ticker = selected
            st.rerun()
    else:
        st.info("Keine Early Movers gefunden")

# ── S&P 500 Scanner ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("🧠 S&P 500 Scanner – mit Empfehlung")

    rows = []
    for sym, df in daily_data.items():
        if len(df) < 20:
            continue

        df_ind = df.copy()
        df_ind["ema9"] = ema(df_ind["close"], 9)
        df_ind["ema20"] = ema(df_ind["close"], 20)
        df_ind["ema50"] = ema(df_ind["close"], 50)
        df_ind["rsi"] = rsi(df_ind["close"])
        df_ind["atr"] = atr(df_ind)
        df_ind.dropna(inplace=True)

        if df_ind.empty:
            continue

        latest = df_ind.iloc[-1]
        vol_ratio = latest["volume"] / df_ind["volume"].mean() if df_ind["volume"].mean() > 0 else 1.0

        snap = MarketSnapshot(sym, latest["close"], latest["rsi"], latest["ema9"], latest["ema20"], latest["ema50"], latest["atr"], vol_ratio, market_state)
        score = calculate_trend_score(snap)
        bias = get_option_bias(snap, score)

        # Empfehlung
        rec = "Vermeiden"
        if score >= 70:
            rec = "Kaufen / Long priorisieren"
        elif score >= 50:
            rec = "Beobachten / Watchlist"

        rows.append({
            "Symbol": sym,
            "Score": score,
            "Bias": bias,
            "Empfehlung": rec
        })

    if rows:
        df_scores = pd.DataFrame(rows).sort_values("Score", ascending=False).head(30)

        def highlight_scanner(row):
            rec = row["Empfehlung"]
            if "Kaufen" in rec:
                return ['background-color: #d4edda'] * len(row)
            elif "Beobachten" in rec:
                return ['background-color: #fff3cd'] * len(row)
            else:
                return ['background-color: #f8d7da'] * len(row)

        styled_scanner = df_scores.style.apply(highlight_scanner, axis=1)

        st.dataframe(styled_scanner, width='stretch', hide_index=True)
    else:
        st.warning("Keine gültigen Scores berechnet")

# Chart und Trading-Tab bleiben gleich (kopiere aus deiner aktuellen Datei, falls nötig)
with tabs[2]:
    st.subheader("📈 Chart Analyse")
    # ... dein aktueller Chart-Code ...

with tabs[3]:
    st.subheader("🟢 Trading-Entscheidung")
    # ... dein aktueller Trading-Code mit News ...
