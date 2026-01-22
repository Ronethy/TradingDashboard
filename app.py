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
from logic.premarket_scanner import scan_early_movers

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Momentum Trading Dashboard",
    layout="wide"
)

# =====================================================
# SYMBOL UNIVERSE (kannst du später erweitern)
# =====================================================
SYMBOLS = [
    "AAPL", "NVDA", "AMD", "TSLA", "META",
    "MSFT", "AMZN", "COIN", "PLTR", "NFLX"
]

# =====================================================
# ALPACA CLIENT (FREE)
# =====================================================
client = StockHistoricalDataClient(
    api_key=st.secrets["ALPACA_API_KEY"],
    secret_key=st.secrets["ALPACA_SECRET_KEY"],
)

# =====================================================
# HEADER
# =====================================================
st.title("📊 Smart Momentum Trading Dashboard")
st.write("Keys geladen:", True)

ny_time = datetime.now(pytz.timezone("US/Eastern"))
st.write("Aktuelle Uhrzeit (NYSE):", ny_time)

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
# LOAD INTRADAY DATA (FREE, BEST EFFORT)
# =====================================================
market_data = {}

for symbol in SYMBOLS:
    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            limit=120
        )
        bars = client.get_stock_bars(req).df
        if not bars.empty:
            market_data[symbol] = bars
    except Exception:
        pass

# =====================================================
# TABS
# =====================================================
tab_scan, tab_day, tab_swing = st.tabs([
    "🔥 Early Movers",
    "⚡ Daytrade",
    "🧭 Swing"
])

# =====================================================
# 🔥 EARLY MOVERS TAB
# =====================================================
with tab_scan:
    st.subheader("🔥 Early Movers – Top 20 (Free Data)")
    st.caption(
        "Frühe Kursabweichungen vom letzten Close "
        "(keine echten Pre-Market-Gaps, Free-Daten)"
    )

    scan_df = scan_early_movers(SYMBOLS, client, max_results=20)

    if scan_df.empty:
        st.warning("Keine relevanten Early Movers gefunden")
    else:
        gap_up = scan_df[scan_df["Gap %"] > 0]
        gap_down = scan_df[scan_df["Gap %"] < 0]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 Gap-Up Movers")
            st.dataframe(
                gap_up.drop(columns=["Abs Gap"]),
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.markdown("### 📉 Gap-Down Movers")
            st.dataframe(
                gap_down.drop(columns=["Abs Gap"]),
                use_container_width=True,
                hide_index=True
            )

# =====================================================
# DETAIL SELECTION (für Daytrade & Swing)
# =====================================================
st.divider()
st.subheader("📈 Detailansicht")

selected = st.selectbox(
    "Aktie auswählen",
    SYMBOLS
)

if selected not in market_data:
    st.warning(
        "Keine Intraday-Daten verfügbar "
        "(Pre-Market bei Free-Daten normal)"
    )
    st.stop()

df = market_data[selected].copy()

# =====================================================
# INDICATORS
# =====================================================
df["ema9"] = ema(df["close"], 9)
df["ema20"] = ema(df["close"], 20)
df["ema50"] = ema(df["close"], 50)
df["rsi"] = rsi(df["close"])
df["atr"] = atr(df)

df.dropna(inplace=True)

if df.empty:
    st.warning("Nicht genug Daten für Analyse")
    st.stop()

latest = df.iloc[-1]

snapshot = MarketSnapshot(
    symbol=selected,
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
# ⚡ DAYTRADE TAB
# =====================================================
with tab_day:
    st.subheader("⚡ Daytrade – konservativ")

    ampel, reasons = decide_daytrade(snapshot)
    st.markdown(f"## {ampel}")

    if reasons:
        st.markdown("**Begründung:**")
        for r in reasons:
            st.write("•", r)

# =====================================================
# 🧭 SWING TAB
# =====================================================
with tab_swing:
    st.subheader("🧭 Swing – konservativ")

    ampel, reasons = decide_swing(snapshot)
    st.markdown(f"## {ampel}")

    if reasons:
        st.markdown("**Begründung:**")
        for r in reasons:
            st.write("•", r)
