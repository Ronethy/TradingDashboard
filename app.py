import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime, timedelta

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

st.set_page_config(page_title="Momentum Trading Dashboard", layout="wide")

# ── Alpaca Client ───────────────────────────────────────────
client = StockHistoricalDataClient(
    st.secrets["ALPACA_API_KEY"],
    st.secrets["ALPACA_SECRET_KEY"]
)

# ── Session State ───────────────────────────────────────────
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"

# ── Marktstatus ─────────────────────────────────────────────
ny_tz = pytz.timezone("America/New_York")
now_ny = datetime.now(ny_tz)
market_state = "PRE" if now_ny.hour < 9 else "OPEN" if now_ny.hour < 16 else "CLOSED"

st.title("📊 Smart Momentum Trading Dashboard")
st.write(f"**NYSE-Zeit:** {now_ny.strftime('%Y-%m-%d %H:%M')} | **Status:** {market_state}")

# ── Daten-Caching ───────────────────────────────────────────
@st.cache_data(ttl=600)
def load_daily_data(symbols):
    data = {}
    batch_size = 80
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=now_ny - timedelta(days=120),
                limit=120
            )
            bars = client.get_stock_bars(req).df
            for sym in batch:
                try:
                    df_sym = bars.xs(sym, level="symbol").copy()
                    if not df_sym.empty:
                        data[sym] = df_sym
                except:
                    pass
        except:
            pass
    return data

daily_data = load_daily_data(SP500_SYMBOLS)

# ── Tabs ────────────────────────────────────────────────────
tab_early, tab_scanner, tab_day, tab_swing = st.tabs([
    "🔥 Early Movers",
    "🧠 S&P 500 Scanner",
    "⚡ Daytrade",
    "🧭 Swing"
])

# ── Early Movers ────────────────────────────────────────────
with tab_early:
    st.subheader("Early Movers – Gap ≥ 0.8%")
    movers = scan_early_movers(SP500_SYMBOLS, client)
    if movers.empty:
        st.info("Keine signifikanten Gaps gefunden")
    else:
        st.dataframe(movers, use_container_width=True, hide_index=True)

        jump = st.selectbox("Zu Detail springen:", ["—"] + movers["Symbol"].tolist())
        if jump != "—":
            st.session_state.selected_ticker = jump
            st.rerun()

# ── S&P 500 Scanner ─────────────────────────────────────────
with tab_scanner:
    st.subheader("S&P 500 – Top Trend-Score")

    results = []
    for sym, df in daily_data.items():
        if len(df) < 40:
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

        snap = MarketSnapshot(
            symbol=sym,
            price=float(latest["close"]),
            rsi=float(latest["rsi"]),
            ema9=float(latest["ema9"]),
            ema20=float(latest["ema20"]),
            ema50=float(latest["ema50"]),
            atr=float(latest["atr"]),
            volume_ratio=vol_ratio,
            market_state=market_state
        )

        score = calculate_trend_score(snap)
        bias = get_option_bias(snap, score)
        plan = generate_trade_plan(snap, score)

        row = {"Symbol": sym, "Score": score, "Bias": bias}
        if plan:
            row.update(plan)
        results.append(row)

    if results:
        df_res = pd.DataFrame(results).sort_values("Score", ascending=False).head(30).reset_index(drop=True)
        st.dataframe(df_res, use_container_width=True)
    else:
        st.warning("Keine auswertbaren Daten")

# ── Daytrade & Swing ────────────────────────────────────────
selected = st.session_state.selected_ticker
st.subheader(f"Detailansicht: {selected}")

if selected in daily_data and len(daily_data[selected]) >= 30:
    df = daily_data[selected].copy()
    df["ema9"] = ema(df["close"], 9)
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr(df)
    df.dropna(inplace=True)

    if not df.empty:
        latest = df.iloc[-1]
        vol_ratio = latest["volume"] / df["volume"].mean()

        snap = MarketSnapshot(
            symbol=selected,
            price=float(latest["close"]),
            rsi=float(latest["rsi"]),
            ema9=float(latest["ema9"]),
            ema20=float(latest["ema20"]),
            ema50=float(latest["ema50"]),
            atr=float(latest["atr"]),
            volume_ratio=vol_ratio,
            market_state=market_state
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⚡ Daytrade")
            ampel_day, reasons_day = decide_daytrade(snap)
            st.markdown(f"### {ampel_day}")
            for r in reasons_day:
                st.write(f"• {r}")

        with col2:
            st.subheader("🧭 Swing")
            ampel_swing, reasons_swing = decide_swing(snap)
            st.markdown(f"### {ampel_swing}")
            for r in reasons_swing:
                st.write(f"• {r}")

        score = calculate_trend_score(snap)
        bias = get_option_bias(snap, score)
        plan = generate_trade_plan(snap, score)

        st.markdown(f"**Trend-Score:** {score}/100 | **Bias:** {bias}")

        if plan:
            st.markdown("**Trade-Plan (1:2 RR)**")
            st.json(plan)
        else:
            st.info("Kein valider Trade-Plan (Score zu niedrig oder RR < 1.8)")
