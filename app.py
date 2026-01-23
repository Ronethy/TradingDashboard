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
    st.warning("Alpha Vantage Key fehlt → News-Funktion deaktiviert")

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
            st.caption(f"News-Fehler {response.status_code}")
            return []
    except Exception as e:
        st.caption(f"News-Request-Fehler: {str(e)}")
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
    st.subheader("🔥 Early Movers – verbesserte Version")
    st.caption("Gaps seit letztem Close + Volumen + Trend-Score + News-Sentiment")

    # Erweiterte Early-Movers-Logik
    enhanced_movers = []
    for sym, df in daily_data.items():
        if len(df) < 2:
            continue
        prev_close = df["close"].iloc[-2]
        current_open = df["open"].iloc[-1]
        current_close = df["close"].iloc[-1]
        gap_pct = (current_open - prev_close) / prev_close * 100
        abs_gap = abs(gap_pct)
        volume = df["volume"].iloc[-1]
        vol_avg = df["volume"].mean()
        vol_ratio = volume / vol_avg if vol_avg > 0 else 1.0
        one_day_perf = (current_close - prev_close) / prev_close * 100

        # Trend-Score berechnen
        try:
            df_ind = df.copy()
            df_ind["ema9"] = ema(df_ind["close"], 9)
            df_ind["ema20"] = ema(df_ind["close"], 20)
            df_ind["ema50"] = ema(df_ind["close"], 50)
            df_ind["rsi"] = rsi(df_ind["close"])
            df_ind["atr"] = atr(df_ind)
            df_ind.dropna(inplace=True)
            if not df_ind.empty:
                latest = df_ind.iloc[-1]
                snap = MarketSnapshot(sym, latest["close"], latest["rsi"], latest["ema9"], latest["ema20"], latest["ema50"], latest["atr"], vol_ratio, market_state)
                score = calculate_trend_score(snap)
            else:
                score = 0
        except:
            score = 0

        # News-Sentiment (letzte 24h)
        news = get_stock_news(sym, alpha_vantage_key, limit=3)
        sentiment = "Neutral"
        if news:
            sentiments = [item.get("overall_sentiment_label", "Neutral") for item in news]
            if "Bullish" in sentiments:
                sentiment = "Bullish"
            elif "Bearish" in sentiments:
                sentiment = "Bearish"

        enhanced_movers.append({
            "Symbol": sym,
            "Gap %": round(gap_pct, 2),
            "Abs Gap": abs_gap,
            "Vol Ratio": round(vol_ratio, 2),
            "1D Perf %": round(one_day_perf, 2),
            "Score": score,
            "Sentiment": sentiment,
            "Last": round(current_close, 2)
        })

    if enhanced_movers:
        df_movers = pd.DataFrame(enhanced_movers)
        df_movers = df_movers.sort_values("Abs Gap", ascending=False).head(20)

        # Farbliche Hervorhebung
        def highlight_row(row):
            gap = row["Gap %"]
            score = row["Score"]
            sent = row["Sentiment"]
            color = ""
            if gap > 3 and score >= 60 and sent == "Bullish":
                color = "background-color: #d4edda; color: black;"
            elif gap > 2 and score >= 50:
                color = "background-color: #fff3cd; color: black;"
            elif gap < -2:
                color = "background-color: #f8d7da; color: black;"
            return [color] * len(row)

        styled = df_movers.style.apply(highlight_row, axis=1)

        st.dataframe(styled, width='stretch', hide_index=True)

        # Klick-Funktion
        selected = st.selectbox("Zu Detail springen:", ["—"] + df_movers["Symbol"].tolist())
        if selected != "—":
            st.session_state.selected_ticker = selected
            st.rerun()
    else:
        st.info("Keine Early Movers mit ausreichend Daten gefunden")

# Die anderen Tabs bleiben gleich (kopiere aus deiner aktuellen app.py, falls nötig)
with tabs[1]:
    # Dein bestehender Scanner-Code hier einfügen...
    st.subheader("🧠 S&P 500 Scanner")
    # ... (dein Code)

with tabs[2]:
    # Dein Chart-Code mit Minute-Bars hier...
    st.subheader("📈 Chart Analyse")
    # ... (dein Code)

with tabs[3]:
    # Dein Trading-Entscheidung-Tab mit News hier...
    st.subheader("🟢 Trading-Entscheidung")
    # ... (dein Code)
