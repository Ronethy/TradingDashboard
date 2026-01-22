import streamlit as st
import pandas as pd
import pytz
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# =========================
# EXTERNE MODULE
# =========================
from data.sp500_symbols import SP500_SYMBOLS
from logic.indicators import ema, rsi, atr
from logic.snapshot import MarketSnapshot
from logic.trend_score import calculate_trend_score
from logic.option_bias import option_bias
from logic.trade_plan import trade_plan
from logic.premarket_scanner import scan_early_movers

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Smart Momentum Trading Dashboard",
    layout="wide"
)

# =========================
# ALPACA CLIENT
# =========================
client = StockHistoricalDataClient(
    api_key=st.secrets["ALPACA_API_KEY"],
    secret_key=st.secrets["ALPACA_SECRET_KEY"]
)

# =========================
# HEADER
# =========================
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

# =========================
# HILFSFUNKTIONEN
# =========================
def traffic_light(score: int):
    if score >= 70:
        return "🟢 BUY"
    elif score >= 45:
        return "🟡 NEUTRAL"
    else:
        return "🔴 AVOID"


def render_trading_chart(df, symbol, timeframe, score_history):
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=[
            f"{symbol} – {timeframe}",
            "Volumen",
            "RSI",
            "Trend-Score Verlauf"
        ]
    )

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price"
        ),
        row=1, col=1
    )

    # EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df["ema9"], name="EMA 9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], name="EMA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], name="EMA 50"), row=1, col=1)

    # Volumen
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume"),
        row=2, col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df["rsi"], name="RSI"),
        row=3, col=1
    )
    fig.add_hline(y=70, line_dash="dot", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", row=3, col=1)

    # Trend-Score Verlauf
    fig.add_trace(
        go.Scatter(
            y=score_history,
            mode="lines+markers",
            name="Trend Score"
        ),
        row=4, col=1
    )

    fig.update_layout(
        height=900,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, width="stretch")


# =========================
# TABS
# =========================
tab_early, tab_sp500, tab_day, tab_swing = st.tabs([
    "🔥 Early Movers",
    "🧠 S&P 500 Scanner",
    "⚡ Daytrade",
    "🧭 Swing"
])

# =========================
# 🔥 EARLY MOVERS
# =========================
with tab_early:
    st.subheader("🔥 Early Movers (Gap Scanner)")

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
                early_df[early_df["Gap %"] > 0],
                width="stretch",
                hide_index=True
            )

        with col2:
            st.markdown("### 📉 Gap Down")
            st.dataframe(
                early_df[early_df["Gap %"] < 0],
                width="stretch",
                hide_index=True
            )

# =========================
# 🧠 S&P 500 SCANNER
# =========================
with tab_sp500:
    st.subheader("🧠 S&P 500 Trend Scanner (Daily)")

    results = []

    for symbol in SP500_SYMBOLS:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                limit=80
            )
            bars = client.get_stock_bars(req).df
            if bars is None or bars.empty:
                continue

            if isinstance(bars.index, pd.MultiIndex):
                df = bars.xs(symbol)
            else:
                df = bars.copy()

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

            results.append({
                "Symbol": symbol,
                "Score": score,
                "Ampel": traffic_light(score),
                "Bias": option_bias(snap, score)
            })

        except Exception:
            continue

    if not results:
        st.info("Keine verwertbaren S&P 500 Daten verfügbar")
    else:
        res = pd.DataFrame(results).sort_values("Score", ascending=False).head(20)
        st.dataframe(res, width="stretch", hide_index=True)

# =========================
# ⚡ DAYTRADE
# =========================
with tab_day:
    st.subheader("⚡ Daytrade Analyse")

    symbol = st.selectbox("Aktie auswählen", SP500_SYMBOLS, key="daytrade")

    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            limit=200
        )
        bars = client.get_stock_bars(req).df
        if bars is None or bars.empty:
            st.warning("Keine Intraday-Daten verfügbar")
        else:
            if isinstance(bars.index, pd.MultiIndex):
                df = bars.xs(symbol)
            else:
                df = bars.copy()

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
            st.markdown(f"## {traffic_light(score)}")

            score_history = []
            for i in range(20, len(df)):
                snap_hist = MarketSnapshot(
                    symbol=symbol,
                    price=float(df.iloc[i]["close"]),
                    rsi=float(df.iloc[i]["rsi"]),
                    ema9=float(df.iloc[i]["ema9"]),
                    ema20=float(df.iloc[i]["ema20"]),
                    ema50=float(df.iloc[i]["ema50"]),
                    atr=float(df.iloc[i]["atr"]),
                    volume_ratio=1.0,
                    market_state=market_state
                )
                score_history.append(calculate_trend_score(snap_hist))

            render_trading_chart(df, symbol, "1-Min", score_history)

    except Exception:
        st.error("Fehler beim Laden der Daytrade-Daten")

# =========================
# 🧭 SWING
# =========================
with tab_swing:
    st.subheader("🧭 Swing Analyse (Daily)")

    symbol = st.selectbox("Aktie auswählen", SP500_SYMBOLS, key="swing")

    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            limit=200
        )
        bars = client.get_stock_bars(req).df
        if bars is None or bars.empty:
            st.warning("Keine Daily-Daten verfügbar")
        else:
            if isinstance(bars.index, pd.MultiIndex):
                df = bars.xs(symbol)
            else:
                df = bars.copy()

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
            st.markdown(f"## {traffic_light(score)}")

            score_history = []
            for i in range(20, len(df)):
                snap_hist = MarketSnapshot(
                    symbol=symbol,
                    price=float(df.iloc[i]["close"]),
                    rsi=float(df.iloc[i]["rsi"]),
                    ema9=float(df.iloc[i]["ema9"]),
                    ema20=float(df.iloc[i]["ema20"]),
                    ema50=float(df.iloc[i]["ema50"]),
                    atr=float(df.iloc[i]["atr"]),
                    volume_ratio=1.0,
                    market_state=market_state
                )
                score_history.append(calculate_trend_score(snap_hist))

            render_trading_chart(df, symbol, "Daily", score_history)

    except Exception:
        st.error("Fehler beim Laden der Swing-Daten")
