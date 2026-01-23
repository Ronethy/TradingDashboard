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

st.title("📊 Smart Momentum Trading Dashboard")
st.write(f"Marktstatus: {market_state} | Zeit: {now_ny.strftime('%Y-%m-%d %H:%M')}")

if st.button("Daten neu laden (Cache leeren)"):
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
                feed="iex",                  # ← FIX: Free-Tier-kompatibel
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
        except Exception as e:
            st.caption(f"Batch-Fehler ({len(batch)} Symbole): {str(e)}")
    return data

daily_data = load_daily_data(SP500_SYMBOLS)

st.caption(f"Geladene Symbole: {len(daily_data)} / {len(SP500_SYMBOLS)}")

tabs = st.tabs([
    "🔥 Early Movers",
    "🧠 S&P 500 Scanner",
    "📈 Chart Analyse",
    "🟢 Trading-Entscheidung"
])

# Early Movers
with tabs[0]:
    st.subheader("🔥 Early Movers")
    movers = scan_early_movers(SP500_SYMBOLS, client)
    if movers.empty:
        st.info("Keine Early Movers gefunden")
    else:
        st.dataframe(movers, width='stretch')

# S&P Scanner
with tabs[1]:
    st.subheader("🧠 S&P 500 Scanner")
    results = []
    for sym, df in daily_data.items():
        try:
            if len(df) < 30:
                continue
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
            snap = MarketSnapshot(sym, latest["close"], latest["rsi"], latest["ema9"], latest["ema20"], latest["ema50"], latest["atr"], vol_ratio, market_state)
            score = calculate_trend_score(snap)
            bias = get_option_bias(snap, score)
            plan = generate_trade_plan(snap, score)
            row = {"Symbol": sym, "Score": score, "Bias": bias}
            if plan:
                row.update(plan)
            results.append(row)
        except:
            pass

    if results:
        df_res = pd.DataFrame(results).sort_values("Score", ascending=False).head(30)
        st.dataframe(df_res, width='stretch')
    else:
        st.warning("Keine Daten geladen – prüfe API-Keys oder Symbol-Liste")

# Chart
with tabs[2]:
    st.subheader("📈 Chart Analyse")
    ticker = st.session_state.selected_ticker
    if ticker in daily_data and not daily_data[ticker].empty:
        df = daily_data[ticker]
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="OHLC"), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name="Volume"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=rsi(df['close']), name="RSI"), row=3, col=1)
        fig.update_layout(height=800, title=f"{ticker} Daily Chart")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Letzte Kerze: {df.index[-1]}")
    else:
        st.info("Keine Chart-Daten für diesen Ticker")

# Trading-Entscheidung
with tabs[3]:
    st.subheader("🟢 Trading-Entscheidung")
    ticker = st.session_state.selected_ticker
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
            snap = MarketSnapshot(ticker, latest["close"], latest["rsi"], latest["ema9"], latest["ema20"], latest["ema50"], latest["atr"], vol_ratio, market_state)
            score = calculate_trend_score(snap)
            st.markdown(f"**Trend-Score:** {score} → {score_to_ampel(score)}")
            bias = get_option_bias(snap, score)
            st.markdown(f"**Option Bias:** {bias}")
            plan = generate_trade_plan(snap, score)
            if plan:
                st.json(plan)
            else:
                st.info("Kein valider Trade-Plan")
            col1, col2 = st.columns(2)
            with col1:
                ampel_d, reasons_d = decide_daytrade(snap)
                st.markdown(f"### Daytrade: {ampel_d}")
                for r in reasons_d: st.write("• " + r)
            with col2:
                ampel_s, reasons_s = decide_swing(snap)
                st.markdown(f"### Swing: {ampel_s}")
                for r in reasons_s: st.write("• " + r)
        else:
            st.warning("Keine Daten nach Indikator-Berechnung")
    else:
        st.info("Wähle einen Ticker mit ausreichend Daten")
