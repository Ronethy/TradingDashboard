import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime, timedelta
import requests

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

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
from logic.additional_indicators import rsi_divergence, macd_info

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

# Globale Symbol-Auswahl oben
st.subheader("Ticker auswählen (für Chart & Trading)")
ticker = st.selectbox(
    "Aktie wählen",
    options=SP500_SYMBOLS,
    index=SP500_SYMBOLS.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in SP500_SYMBOLS else 0,
    key="global_ticker_select"
)

if ticker != st.session_state.selected_ticker:
    st.session_state.selected_ticker = ticker
    st.rerun()

# Zeitrahmen-Auswahl für Chart
timeframe_options = {
    "Minute": TimeFrame.Minute,
    "5 Minuten": TimeFrame(5, TimeFrameUnit.Minute),
    "Day": TimeFrame.Day,
    "Week": TimeFrame.Week
}

timeframe_str = st.selectbox("Zeitrahmen für Chart", list(timeframe_options.keys()), index=2)  # Default Day
timeframe = timeframe_options[timeframe_str]

if st.button("Daten aktualisieren (Cache leeren)"):
    st.cache_data.clear()
    st.rerun()

@st.cache_data(ttl=60)
def load_bars(ticker, timeframe, start, end):
    try:
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=timeframe,
            start=start,
            end=end,
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
        st.caption(f"Bars-Fehler {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_stock_news(ticker, api_key, limit=3):
    if not api_key:
        return []
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit={limit}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("feed", [])[:limit]
        return []
    except:
        return []

# Daten laden – pro Symbol einzeln, um UnhashableParamError zu vermeiden
daily_data = {}
for sym in SP500_SYMBOLS:
    df = load_bars(sym, TimeFrame.Day, now_ny - timedelta(days=150), now_ny + timedelta(days=1))
    if not df.empty:
        daily_data[sym] = df

st.caption(f"Geladene Symbole: {len(daily_data)} / {len(SP500_SYMBOLS)}")

tabs = st.tabs([
    "🔥 Early Movers",
    "🧠 S&P 500 Scanner",
    "📈 Chart Analyse",
    "🟢 Trading-Entscheidung"
])

with tabs[0]:
    st.subheader("🔥 Early Movers – mit Farben & News für Top-Kandidaten")
    enhanced_movers = []
    for sym, df in daily_data.items():
        if len(df) < 2:
            continue
        prev_close = df["close"].iloc[-2]
        current_open = df["open"].iloc[-1]
        gap_pct = (current_open - prev_close) / prev_close * 100
        abs_gap = abs(gap_pct)
        volume = df["volume"].iloc[-1]
        vol_avg = df["volume"].mean()
        vol_ratio = volume / vol_avg if vol_avg > 0 else 1.0
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
        enhanced_movers.append({
            "Symbol": sym,
            "Gap %": round(gap_pct, 2),
            "Abs Gap": abs_gap,
            "Vol Ratio": round(vol_ratio, 2),
            "Score": score
        })
    if enhanced_movers:
        df_movers = pd.DataFrame(enhanced_movers)
        df_movers = df_movers.sort_values("Abs Gap", ascending=False).head(20)
        def get_recommendation(row):
            gap = row["Gap %"]
            score = row["Score"]
            if score >= 70 or gap > 3:
                return "Kaufen / Long priorisieren"
            elif score >= 50 or gap > 2:
                return "Beobachten / Watchlist"
            else:
                return "Vermeiden"
        df_movers["Empfehlung"] = df_movers.apply(get_recommendation, axis=1)
        def highlight_row(row):
            rec = row["Empfehlung"]
            if "Kaufen" in rec:
                return ['background-color: #d4edda; color: black'] * len(row)
            elif "Beobachten" in rec:
                return ['background-color: #fff3cd; color: black'] * len(row)
            else:
                return ['background-color: #f8d7da; color: black'] * len(row)
        styled = df_movers.style.apply(highlight_row, axis=1)
        st.dataframe(styled, width='stretch', hide_index=True)
        st.subheader("News zu Top Early Movers")
        top_symbols = df_movers.head(5)["Symbol"].tolist()
        for sym in top_symbols:
            news = get_stock_news(sym, alpha_vantage_key, limit=2)
            if news:
                st.markdown(f"**{sym}** (Score {df_movers[df_movers['Symbol'] == sym]['Score'].values[0]})")
                for item in news:
                    title = item.get("title", "No title")
                    url = item.get("url", "#")
                    sentiment = item.get("overall_sentiment_label", "Neutral")
                    st.markdown(f"- [{title}]({url}) – Sentiment: **{sentiment}**")
                st.markdown("---")
            else:
                st.caption(f"Keine News für {sym}")
    else:
        st.info("Keine Early Movers gefunden")

# Rest der Datei bleibt exakt wie in deiner letzten funktionierenden Version
# (S&P Scanner, Chart, Trading-Entscheidung)
# ...
