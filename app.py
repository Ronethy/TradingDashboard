import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import pytz

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from logic.indicators import ema, rsi, atr
from logic.snapshot import MarketSnapshot
from logic.decision_daytrade import decide_daytrade
from logic.decision_swing import decide_swing
from logic.premarket_scanner import scan_early_movers

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Momentum Trading Dashboard",
    layout="wide"
)

# =====================================================
# NASDAQ 100 (Momentum-tauglich)
# =====================================================
SYMBOLS = [
    "AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","GOOG","AMD","NFLX",
    "INTC","PEP","AVGO","COST","CSCO","ADBE","QCOM","TXN","INTU","AMAT",
    "BKNG","MDLZ","ISRG","GILD","ADI","MU","LRCX","REGN","VRTX","PANW",
    "SNPS","KLAC","CDNS","MAR","ORLY","ADP","NXPI","FTNT","MELI","CTAS",
    "ASML","ABNB","TEAM","BIIB","KDP","PAYX","ODFL","PCAR","ROST","SIRI"
]

# =====================================================
# ALPACA CLIENT
# =====================================================
client = StockHistoricalDataClient(
    api_key=st.secrets["ALPACA_API_KEY"],
    secret_key=st.secrets["ALPACA_SECRET_KEY"]
)

# =====================================================
# HEADER
# =====================================================
st.title("📊 Smart Momentum Trading Dashboard")

ny_time = datetime.now(pytz.timezone("US/Eastern"))
st.caption(f"Aktuelle Uhrzeit (NYSE): {ny_time}")

if ny_time.hour < 9 or (ny_time.hour == 9 and ny_time.minute < 30):
    market_state = "PRE"
    st.write("Marktstatus: 🟡 Pre-Market")
elif ny_time.hour < 16:
    market_state = "OPEN"
    st.write("Marktstatus: 🟢 Open")
else:
    market_state = "CLOSED"
    st.write("Marktstatus: 🔴 Closed")

st.divider()

# =====================================================
# EARLY MOVERS SCAN (TOP 20)
# =====================================================
scan_df = scan_early_movers(
    SYMBOLS,
    client,
    max_results=20
)

if scan_df.empty:
    st.warning("Keine Early Movers gefunden")
    st.stop()

# Persist selection
if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = scan_df.iloc[0]["Symbol"]

# =====================================================
# TABS
# =====================================================
tab_scan, tab_day, tab_swing = st.tabs([
    "🔥 Early Movers",
    "⚡ Daytrade",
    "🧭 Swing"
])

# =====================================================
# 🔥 EARLY MOVERS TAB (CLICKABLE)
# =====================================================
with tab_scan:
    st.subheader("🔥 Early Movers – Top 20")

    st.caption(
        "Ranking nach stärkster Abweichung vom letzten Close "
        "(Free-Daten, konservativ)"
    )

    st.dataframe(
        scan_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    selected_rows = st.session_state.get("dataframe_selection", None)
    if selected_rows:
        row = selected_rows["rows"][0]
        st.session_state.selected_symbol = scan_df.iloc[row]["Symbol"]

# =====================================================
# LOAD INTRADAY DATA
# =====================================================
symbol = st.session_state.selected_symbol

req = StockBarsRequest(
    symbol_or_symbols=symbol,
    timeframe=TimeFrame.Minute,
    limit=300
)

bars = client.get_stock_bars(req).df

if bars.empty:
    st.warning("Keine Intraday-Daten verfügbar")
    st.stop()

df = bars.copy()

# =====================================================
# INDICATORS
# =====================================================
df["ema9"] = ema(df["close"], 9)
df["ema20"] = ema(df["close"], 20)
df["ema50"] = ema(df["close"], 50)
df["rsi"] = rsi(df["close"])
df["atr"] = atr(df)

df.dropna(inplace=True)
latest = df.iloc[-1]

snapshot = MarketSnapshot(
    symbol=symbol,
    price=float(latest["close"]),
    rsi=float(latest["rsi"]),
    ema9=float(latest["ema9"]),
    ema20=float(latest["ema20"]),
    ema50=float(latest["ema50"]),
    atr=float(latest["atr"]),
    volume_ratio=float(latest["volume"] / df["volume"].mean()),
    market_state=market_state
)

# =====================================================
# 📈 CANDLESTICK CHART
# =====================================================
st.subheader(f"📈 {symbol} – Intraday Chart")

fig = go.Figure()

fig.add_candlestick(
    x=df.index,
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    name="Price"
)

fig.add_trace(go.Scatter(x=df.index, y=df["ema9"], name="EMA 9"))
fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], name="EMA 20"))
fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], name="EMA 50"))

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ⚡ DAYTRADE
# =====================================================
with tab_day:
    st.subheader("⚡ Daytrade – konservativ")

    ampel, reasons = decide_daytrade(snapshot)
    st.markdown(f"## {ampel}")

    for r in reasons:
        st.write("•", r)

# =====================================================
# 🧭 SWING
# =====================================================
with tab_swing:
    st.subheader("🧭 Swing – konservativ")

    ampel, reasons = decide_swing(snapshot)
    st.markdown(f"## {ampel}")

    for r in reasons:
        st.write("•", r)
