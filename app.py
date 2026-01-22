import streamlit as st
import pandas as pd
import pytz
from datetime import datetime

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from data.sp500_symbols import SP500_SYMBOLS
from logic.indicators import ema, rsi, atr
from logic.snapshot import MarketSnapshot
from logic.trend_score import calculate_trend_score
from logic.option_bias import option_bias
from logic.trade_plan import trade_plan
from logic.premarket_scanner import scan_early_movers

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Momentum Trading Dashboard",
    layout="wide"
)

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
# TABS (ALLE!)
# =====================================================
tab_early, tab_sp500, tab_day, tab_swing = st.tabs([
    "🔥 Early Movers",
    "🧠 S&P 500 Scanner",
    "⚡ Daytrade",
    "🧭 Swing"
])

# =====================================================
# 🔥 EARLY MOVERS
# =====================================================
with tab_early:
    st.subheader("🔥 Early Movers")
    st.caption("Daily Open vs. Vortages-Close (Free Alpaca)")

    early_df = scan_early_movers(
        symbols=SP500_SYMBOLS,
        client=client,
        max_results=20
    )

    if early_df.empty:
        st.info("Keine Early Movers gefunden – Markt ruhig")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 Gap Up")
            st.dataframe(
                early_df[early_df["Gap %"] > 0].drop(columns=["Abs Gap"]),
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.markdown("### 📉 Gap Down")
            st.dataframe(
                early_df[early_df["Gap %"] < 0].drop(columns=["Abs Gap"]),
                use_container_width=True,
                hide_index=True
            )

# =====================================================
# 🧠 S&P 500 SCANNER (DAILY)
# =====================================================
with tab_sp500:
    st.subheader("🧠 S&P 500 Trend Scanner (Daily)")
    st.caption("Trend-Score • Option-Bias • Trade-Plan (Top 20)")

    results = []

    for symbol in SP500_SYMBOLS:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                limit=60
            )
            df = client.get_stock_bars(req).df
            if df is None or df.empty:
                continue

            df["ema9"] = ema(df["close"], 9)
            df["ema20"] = ema(df["close"], 20)
            df["ema50"] = ema(df["close"], 50)
            df["rsi"] = rsi(df["close"])
            df["atr"] = atr(df)
            df.dropna(inplace=True)
            if df.empty:
                continue

            last = df.iloc[-1]

            snap = MarketSnapshot(
                symbol=symbol,
                price=float(last["close"]),
                rsi=float(last["rsi"]),
                ema9=float(last["ema9"]),
                ema20=float(last["ema20"]),
                ema50=float(last["ema50"]),
                atr=float(last["atr"]),
                volume_ratio=float(last["volume"] / df["volume"].mean()),
                market_state=market_state
            )

            score = calculate_trend_score(snap)
            bias = option_bias(snap, score)
            plan = trade_plan(snap, score)

            results.append({
                "Symbol": symbol,
                "Score": score,
                "Bias": bias,
                "Entry": plan["Entry"] if plan else None,
                "Stop": plan["Stop"] if plan else None,
                "Target": plan["Target"] if plan else None
            })

        except Exception:
            continue

    if not results:
        st.info("Keine verwertbaren S&P 500 Daten verfügbar")
    else:
        res = pd.DataFrame(results)
        res = res.sort_values("Score", ascending=False).head(20)
        st.dataframe(res, use_container_width=True, hide_index=True)

# =====================================================
# ⚡ DAYTRADE (DETAIL)
# =====================================================
with tab_day:
    st.subheader("⚡ Daytrade – Detailanalyse")

    symbol = st.selectbox(
        "Aktie auswählen",
        SP500_SYMBOLS,
        key="daytrade_symbol"
    )

    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            limit=120
        )
        df = client.get_stock_bars(req).df

        if df is None or df.empty:
            st.warning("Keine Intraday-Daten verfügbar")
        else:
            df["ema9"] = ema(df["close"], 9)
            df["ema20"] = ema(df["close"], 20)
            df["ema50"] = ema(df["close"], 50)
            df["rsi"] = rsi(df["close"])
            df["atr"] = atr(df)
            df.dropna(inplace=True)

            last = df.iloc[-1]

            snap = MarketSnapshot(
                symbol=symbol,
                price=float(last["close"]),
                rsi=float(last["rsi"]),
                ema9=float(last["ema9"]),
                ema20=float(last["ema20"]),
                ema50=float(last["ema50"]),
                atr=float(last["atr"]),
                volume_ratio=float(last["volume"] / df["volume"].mean()),
                market_state=market_state
            )

            score = calculate_trend_score(snap)
            bias = option_bias(snap, score)
            plan = trade_plan(snap, score)

            st.metric("Trend-Score", score)
            st.metric("Option-Bias", bias)

            if plan:
                st.markdown("### 📦 Trade-Plan")
                st.json(plan)
            else:
                st.info("Kein valider Trade-Plan (konservativ gefiltert)")

    except Exception:
        st.error("Fehler beim Laden der Intraday-Daten")

# =====================================================
# 🧭 SWING (DETAIL)
# =====================================================
with tab_swing:
    st.subheader("🧭 Swing – Mehrtägige Analyse")

    symbol = st.selectbox(
        "Aktie auswählen",
        SP500_SYMBOLS,
        key="swing_symbol"
    )

    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            limit=120
        )
        df = client.get_stock_bars(req).df

        if df is None or df.empty:
            st.warning("Keine Daily-Daten verfügbar")
        else:
            df["ema9"] = ema(df["close"], 9)
            df["ema20"] = ema(df["close"], 20)
            df["ema50"] = ema(df["close"], 50)
            df["rsi"] = rsi(df["close"])
            df["atr"] = atr(df)
            df.dropna(inplace=True)

            last = df.iloc[-1]

            snap = MarketSnapshot(
                symbol=symbol,
                price=float(last["close"]),
                rsi=float(last["rsi"]),
                ema9=float(last["ema9"]),
                ema20=float(last["ema20"]),
                ema50=float(last["ema50"]),
                atr=float(last["atr"]),
                volume_ratio=float(last["volume"] / df["volume"].mean()),
                market_state=market_state
            )

            score = calculate_trend_score(snap)
            bias = option_bias(snap, score)
            plan = trade_plan(snap, score)

            st.metric("Trend-Score", score)
            st.metric("Option-Bias", bias)

            if plan:
                st.markdown("### 📦 Swing-Trade-Plan")
                st.json(plan)
            else:
                st.info("Kein valider Swing-Trade (konservativ)")

    except Exception:
        st.error("Fehler beim Laden der Swing-Daten")
