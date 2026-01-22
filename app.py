import streamlit as st
import pandas as pd
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
from logic.decision_base import score_to_ampel

st.set_page_config(page_title="Momentum Dashboard", layout="wide")

client = StockHistoricalDataClient(
    st.secrets["ALPACA_API_KEY"],
    st.secrets["ALPACA_SECRET_KEY"]
)

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"

ny_tz = pytz.timezone("America/New_York")
now_ny = datetime.now(ny_tz)
market_state = "PRE" if now_ny.hour < 9 else "OPEN" if now_ny.hour < 16 else "CLOSED"

st.title("Smart Momentum Trading Dashboard")
st.caption(f"NYSE-Zeit: {now_ny.strftime('%Y-%m-%d %H:%M')} | {market_state}")

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
                end=now_ny + timedelta(hours=6),          # aktuelle Kerze holen
                adjustment="all"
            )
            bars = client.get_stock_bars(req).df
            for sym in batch:
                try:
                    df_sym = bars[bars.index.get_level_values('symbol') == sym].copy()
                    if not df_sym.empty:
                        data[sym] = df_sym
                except:
                    pass
        except:
            pass
    return data

daily_data = load_daily_data(SP500_SYMBOLS)

st.caption(f"Geladene Symbole: {len(daily_data)} / {len(SP500_SYMBOLS)}")

tabs = st.tabs(["Early Movers", "S&P Scanner", "Detail & Entscheidung"])

with tabs[0]:
    st.subheader("Early Movers – Gap ≥ 0.8%")
    movers = scan_early_movers(SP500_SYMBOLS, client)
    if movers.empty:
        st.info("Keine signifikanten Gaps")
    else:
        st.dataframe(movers, width='stretch', hide_index=True)
        jump = st.selectbox("Zu Detail:", ["—"] + movers["Symbol"].tolist())
        if jump != "—":
            st.session_state.selected_ticker = jump
            st.rerun()

with tabs[1]:
    st.subheader("S&P 500 – Top Trend-Scores")
    results = []
    for sym, df_raw in daily_data.items():
        try:
            if len(df_raw) < 30:
                continue
            df = df_raw.copy()
            df["ema9"] = ema(df["close"], 9)
            df["ema20"] = ema(df["close"], 20)
            df["ema50"] = ema(df["close"], 50)
            df["rsi"] = rsi(df["close"])
            df["atr"] = atr(df)
            df.dropna(inplace=True)
            if df.empty:
                continue
            latest = df.iloc[-1]
            vol_ratio = latest["volume"] / df["volume"].mean() if df["volume"].mean() > 0 else 1.0
            snap = MarketSnapshot(
                sym, float(latest["close"]), float(latest["rsi"]),
                float(latest["ema9"]), float(latest["ema20"]), float(latest["ema50"]),
                float(latest["atr"]), vol_ratio, market_state
            )
            score = calculate_trend_score(snap)
            bias = get_option_bias(snap, score)
            plan = generate_trade_plan(snap, score)
            row = {"Symbol": sym, "Score": score, "Ampel": score_to_ampel(score), "Bias": bias}
            if plan:
                row.update(plan)
            results.append(row)
        except:
            pass

    if results:
        df_res = pd.DataFrame(results).sort_values("Score", ascending=False).head(30).reset_index(drop=True)
        st.dataframe(df_res, width='stretch')
    else:
        st.warning("Keine auswertbaren Daten")

with tabs[2]:
    ticker = st.selectbox("Ticker", options=list(daily_data.keys()) or ["AAPL"], index=0)
    st.session_state.selected_ticker = ticker

    if ticker in daily_data and len(daily_data[ticker]) >= 30:
        df = daily_data[ticker].copy()
        df["ema9"] = ema(df["close"], 9)
        df["ema20"] = ema(df["close"], 20)
        df["ema50"] = ema(df["close"], 50)
        df["rsi"] = rsi(df["close"])
        df["atr"] = atr(df)
        df.dropna(inplace=True)

        if not df.empty:
            latest = df.iloc[-1]
            vol_ratio = latest["volume"] / df["volume"].mean() if df["volume"].mean() > 0 else 1.0
            snap = MarketSnapshot(
                ticker, float(latest["close"]), float(latest["rsi"]),
                float(latest["ema9"]), float(latest["ema20"]), float(latest["ema50"]),
                float(latest["atr"]), vol_ratio, market_state
            )

            score = calculate_trend_score(snap)
            st.markdown(f"**Trend-Score:** {score} → {score_to_ampel(score)}")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Daytrade")
                ampel_d, reasons_d = decide_daytrade(snap)
                st.markdown(f"### {ampel_d}")
                for r in reasons_d: st.write("• " + r)

            with col2:
                st.subheader("Swing")
                ampel_s, reasons_s = decide_swing(snap)
                st.markdown(f"### {ampel_s}")
                for r in reasons_s: st.write("• " + r)

            st.markdown(f"**Bias:** {get_option_bias(snap, score)}")

            plan = generate_trade_plan(snap, score)
            if plan:
                st.json(plan)
            else:
                st.info("Kein valider Trade-Plan")
        else:
            st.warning("Nach Berechnung keine Daten mehr übrig")
    else:
        st.info("Nicht genug Daten für diesen Ticker")
