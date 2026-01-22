import streamlit as st
import pandas as pd
import pytz
from datetime import datetime, timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from data.sp500_symbols import SP500_SYMBOLS
from logic.indicators import rsi
from logic.trend_score import calculate_trend_score
from logic.premarket_scanner import scan_early_movers

st.set_page_config(layout="wide")

client = StockHistoricalDataClient(
    st.secrets["ALPACA_API_KEY"],
    st.secrets["ALPACA_SECRET_KEY"]
)

st.title("📊 Smart Momentum Trading Dashboard")

ny = pytz.timezone("US/Eastern")
now = datetime.now(ny)

st.write(f"**Marktstatus:** {'🟢 Open' if 9 <= now.hour < 16 else '🔴 Closed'}")

@st.cache_data(ttl=300)
def load_data(symbols):
    data = {}
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=now - timedelta(days=60)
    )
    bars = client.get_stock_bars(req)

    for sym in symbols:
        df = bars.df[bars.df["symbol"] == sym].copy()
        if not df.empty:
            data[sym] = df

    return data

market_data = load_data(SP500_SYMBOLS)

tabs = st.tabs([
    "🔥 Early Movers",
    "🧠 S&P 500 Scanner",
    "📈 Chart Analyse"
])

# ================= EARLY MOVERS =================
with tabs[0]:
    st.subheader("🔥 Early Movers")
    movers = scan_early_movers(market_data)
    if movers.empty:
        st.info("Keine Early Movers gefunden – Markt ruhig")
    else:
        st.dataframe(movers, width="stretch")

# ================= S&P500 =================
with tabs[1]:
    st.subheader("🧠 S&P 500 Scanner")

    rows = []
    for sym, df in market_data.items():
        score = calculate_trend_score(df)
        rows.append({"Symbol": sym, "Trend-Score": score})

    df_scores = pd.DataFrame(rows).sort_values("Trend-Score", ascending=False)
    st.dataframe(df_scores, width="stretch")

# ================= CHART =================
with tabs[2]:
    st.subheader("📈 Chart Analyse")

    symbol = st.selectbox("Aktie auswählen", SP500_SYMBOLS)

    if symbol in market_data:
        df = market_data[symbol]

        df["RSI"] = rsi(df["close"])

        st.line_chart(df["close"], width="stretch")
        st.line_chart(df["RSI"], width="stretch")
    else:
        st.warning("Keine Daten verfügbar")
