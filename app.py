import streamlit as st
import pandas as pd
from datetime import datetime
import pytz

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from logic.indicators import ema, rsi, atr
from logic.snapshot import MarketSnapshot
from logic.decision_daytrade import decide_daytrade
from logic.decision_swing import decide_swing

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Smart Momentum Trading Dashboard", layout="wide")

SYMBOLS = ["AAPL", "NVDA", "AMD", "TSLA", "META", "MSFT", "AMZN", "COIN", "PLTR", "NFLX"]

client = StockHistoricalDataClient(
    api_key=st.secrets["ALPACA_API_KEY"],
    secret_key=st.secrets["ALPACA_SECRET_KEY"],
)

# ===============================
# HEADER
# ===============================
st.title("📊 Smart Momentum Trading Dashboard")
st.write("Keys geladen:", True)

ny_time = datetime.now(pytz.timezone("US/Eastern"))
st.write("Aktuelle Uhrzeit (NYSE):", ny_time)

market_state = "PRE" if ny_time.hour < 9 or (ny_time.hour == 9 and ny_time.minute < 30) else "OPEN"
st.write("Marktstatus:", "🟡 Pre-Market" if market_state == "PRE" else "🟢 Open")

# ===============================
# DATA LOADING
# ===============================
market_data = {}

for symbol in SYMBOLS:
    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            limit=100
        )
        bars = client.get_stock_bars(req).df
        if not bars.empty:
            market_data[symbol] = bars
    except:
        pass

# ===============================
# DETAIL VIEW
# ===============================
st.subheader("📈 Detailansicht")
selected = st.selectbox("Aktie auswählen", SYMBOLS)

if selected not in market_data:
    st.warning("Keine Intraday-Daten verfügbar (Pre-Market normal)")
    st.stop()

df = market_data[selected].copy()

df["ema9"] = ema(df["close"], 9)
df["ema20"] = ema(df["close"], 20)
df["ema50"] = ema(df["close"], 50)
df["rsi"] = rsi(df["close"])
df["atr"] = atr(df)

latest = df.iloc[-1]

snapshot = MarketSnapshot(
    symbol=selected,
    price=latest["close"],
    rsi=latest["rsi"],
    ema9=latest["ema9"],
    ema20=latest["ema20"],
    ema50=latest["ema50"],
    atr=latest["atr"],
    volume_ratio=latest["volume"] / df["volume"].mean(),
    market_state=market_state
)

# ===============================
# TABS
# ===============================
tab_day, tab_swing = st.tabs(["⚡ Daytrade", "🧭 Swing"])

with tab_day:
    ampel, reasons = decide_daytrade(snapshot)
    st.subheader(ampel)
    for r in reasons:
        st.write("•", r)

with tab_swing:
    ampel, reasons = decide_swing(snapshot)
    st.subheader(ampel)
    for r in reasons:
        st.write("•", r)
