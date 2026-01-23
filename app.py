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

if market_state == "PRE":
    st.warning("Pre-Market: Viele Scores & Ampele sind eingeschränkt – warte auf Open (9:30 ET)")

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
                feed="iex"
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
            st.caption(f"Batch-Fehler: {str(e)}")
    return data

daily_data = load_daily_data(SP500_SYMBOLS)

st.caption(f"Geladene Symbole: {len(daily_data)} / {len(SP500_SYMBOLS)}")

tabs = st.tabs([
    "🔥 Early Movers",
    "🧠 S&P 500 Scanner",
    "📈 Chart Analyse",
    "🟢 Trading-Entscheidung"
])

# ── Early Movers ────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("🔥 Early Movers")

    movers = scan_early_movers(daily_data)
    if movers.empty:
        st.info("Keine Early Movers gefunden")
    else:
        st.dataframe(movers, width='stretch', hide_index=True)

        # Symbol-Auswahl mit explizitem State-Update
        selected_early = st.selectbox(
            "Zu Detail springen:",
            options=["—"] + movers["Symbol"].tolist(),
            key="early_select"
        )

        if selected_early != "—" and selected_early != st.session_state.selected_ticker:
            st.session_state.selected_ticker = selected_early
            st.rerun()

# ── S&P 500 Scanner ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("🧠 S&P 500 Scanner")

    rows = []
    for sym, df in daily_data.items():
        if len(df) < 20:
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
        rows.append({"Symbol": sym, "Trend-Score": score})

    if rows:
        df_scores = pd.DataFrame(rows).sort_values("Trend-Score", ascending=False).reset_index(drop=True)
        st.dataframe(df_scores, width='stretch', hide_index=True)

        # Symbol-Auswahl für Scanner
        selected_scanner = st.selectbox(
            "Zu Detail springen (aus Scanner):",
            options=["—"] + df_scores["Symbol"].head(20).tolist(),
            key="scanner_select"
        )

        if selected_scanner != "—" and selected_scanner != st.session_state.selected_ticker:
            st.session_state.selected_ticker = selected_scanner
            st.rerun()
    else:
        st.warning("Keine gültigen Scores berechnet")

# ── Chart Analyse ───────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("📈 Chart Analyse")

    # Ticker-Auswahl mit aktueller State
    available = list(daily_data.keys()) if daily_data else ["AAPL"]
    ticker = st.selectbox(
        "Ticker auswählen",
        options=available,
        index=available.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in available else 0,
        key="chart_select"
    )

    # State synchronisieren (wichtig!)
    if ticker != st.session_state.selected_ticker:
        st.session_state.selected_ticker = ticker
        st.rerun()

    if ticker in daily_data and not daily_data[ticker].empty:
        df = daily_data[ticker].copy()

        df["ema20"] = ema(df["close"], 20)
        df["ema50"] = ema(df["close"], 50)
        df["RSI"] = rsi(df["close"])
        df["ATR"] = atr(df)

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.55, 0.15, 0.30])
        fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"]), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], name="EMA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], name="EMA50"), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df["volume"]), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(height=800, title=f"{ticker} – Daily")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Letzte Kerze: {df.index[-1]}")
    else:
        st.info("Keine Daten für diesen Ticker")

# ── Trading-Entscheidung ────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("🟢 Trading-Entscheidung")

    ticker = st.session_state.selected_ticker
    st.write(f"Ausgewählte Aktie: **{ticker}**")

    if ticker in daily_data and len(daily_data[ticker]) >= 20:
        df = daily_data[ticker].copy()
        df_ind = df.copy()
        df_ind["ema9"] = ema(df_ind["close"], 9)
        df_ind["ema20"] = ema(df_ind["close"], 20)
        df_ind["ema50"] = ema(df_ind["close"], 50)
        df_ind["rsi"] = rsi(df_ind["close"])
        df_ind["atr"] = atr(df_ind)
        df_ind.dropna(inplace=True)

        if not df_ind.empty:
            latest = df_ind.iloc[-1]
            vol_ratio = latest["volume"] / df_ind["volume"].mean() if df_ind["volume"].mean() > 0 else 1.0

            snap = MarketSnapshot(
                symbol=ticker,
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

            if score >= 70:
                st.success(f"🟢 Stark Bullish (Score {score})")
            elif score >= 40:
                st.warning(f"🟡 Neutral / vorsichtig (Score {score})")
            else:
                st.error(f"🔴 Bearish / meiden (Score {score})")

            st.markdown(f"**Option Bias:** {bias}")

            if plan:
                st.markdown("**Trade-Plan**")
                st.json(plan)
            else:
                st.info("Kein valider Trade-Plan")

            col1, col2 = st.columns(2)
            with col1:
                ampel_d, reasons_d = decide_daytrade(snap)
                st.markdown(f"**Daytrade:** {ampel_d}")
                for r in reasons_d:
                    st.write("• " + r)

            with col2:
                ampel_s, reasons_s = decide_swing(snap)
                st.markdown(f"**Swing:** {ampel_s}")
                for r in reasons_s:
                    st.write("• " + r)

        else:
            st.warning("Keine Daten nach Berechnung")
    else:
        st.info("Wähle eine Aktie mit ausreichend Historie")
