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

# ── Client ────────────────
client = StockHistoricalDataClient(
    st.secrets["ALPACA_API_KEY"],
    st.secrets["ALPACA_SECRET_KEY"]
)

# ── State ─────────────────
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"

# ── Zeit ──────────────────
ny_tz = pytz.timezone("America/New_York")
now_ny = datetime.now(ny_tz)
market_state = "PRE" if now_ny.hour < 9 else "OPEN" if now_ny.hour < 16 else "CLOSED"

st.title("📊 Smart Momentum Trading Dashboard")
st.caption(f"NYSE-Zeit: {now_ny.strftime('%H:%M')} | Markt: {'🟢 Open' if market_state == 'OPEN' else '🔴 Closed / Pre / After'}")

# ── Refresh ───────────────
if st.button("Daten neu laden (Cache leeren)"):
    st.cache_data.clear()
    st.rerun()

# ── Daten laden ───────────
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
                start=now_ny - timedelta(days=120),
                end=now_ny + timedelta(days=1),
                feed="iex",
                adjustment="all"
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

# ── Tabs ──────────────────
tabs = st.tabs([
    "🔥 Early Movers",
    "🧠 S&P Scanner",
    "📈 Chart",
    "🟢 Entscheidung"
])

# ── Early Movers ──────────
with tabs[0]:
    st.subheader("🔥 Early Movers (Gap ≥ 1%)")

    movers = scan_early_movers(daily_data)
    if movers.empty:
        st.info("Aktuell keine signifikanten Gaps → Markt ist ruhig")
    else:
        # Klickbar machen → Ticker auswählen
        st.dataframe(
            movers.style.format(precision=2),
            width='stretch',
            hide_index=True
        )
        if st.button("→ Zum ausgewählten Symbol springen", key="jump_early"):
            st.session_state.selected_ticker = movers.iloc[0]["Symbol"]
            st.rerun()

# ── S&P Scanner ───────────
with tabs[1]:
    st.subheader("🧠 S&P 500 Trend-Score Ranking")

    rows = []
    for sym, df in daily_data.items():
        if len(df) < 20:
            continue
        score = calculate_trend_score(df)
        rows.append({"Symbol": sym, "Trend-Score": score})

    df_scores = pd.DataFrame(rows).sort_values("Trend-Score", ascending=False).reset_index(drop=True)

    st.dataframe(
        df_scores.style.format({"Trend-Score": "{:.0f}"}),
        width='stretch',
        hide_index=True
    )

    # Optional: Klick auf Zeile → Ticker setzen (Streamlit Dataframe hat derzeit kein natives on_click)
    st.info("Tipp: Kopiere das Symbol und wähle es unten im Chart-Tab aus.")

# ── Chart ─────────────────
with tabs[2]:
    st.subheader("📈 Chart & Indikatoren")

    ticker = st.selectbox(
        "Aktie auswählen",
        options=SP500_SYMBOLS,
        index=SP500_SYMBOLS.index(st.session_state.selected_ticker)
        if st.session_state.selected_ticker in SP500_SYMBOLS else 0,
        key="chart_ticker_select"
    )

    # Sync session state
    st.session_state.selected_ticker = ticker

    if ticker in daily_data and not daily_data[ticker].empty:
        df = daily_data[ticker].copy()

        # Indikatoren berechnen
        df["ema20"] = ema(df["close"], 20)
        df["ema50"] = ema(df["close"], 50)
        df["RSI"] = rsi(df["close"])
        df["ATR"] = atr(df)

        # Plotly Figure mit Subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.55, 0.15, 0.30],
            subplot_titles=("Candlestick + EMAs", "Volume", "RSI")
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"], high=df["high"],
                low=df["low"], close=df["close"],
                name="OHLC",
                increasing_line_color="green", decreasing_line_color="red"
            ),
            row=1, col=1
        )

        # EMAs
        fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], name="EMA 20", line=dict(color="#00BFFF")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], name="EMA 50", line=dict(color="#FF8C00")), row=1, col=1)

        # Volume
        fig.add_trace(
            go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color="#4682B4"),
            row=2, col=1
        )

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#9932CC")), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(
            height=800,
            title=f"{ticker} – Daily",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, width='stretch')

    else:
        st.warning(f"Keine ausreichenden Daten für {ticker}")

# ── Trading-Entscheidung ──
with tabs[3]:
    st.subheader("🟢 Trading-Entscheidung")

    ticker = st.session_state.selected_ticker
    st.write(f"Ausgewählte Aktie: **{ticker}**")

    if ticker in daily_data and len(daily_data[ticker]) >= 20:
        df = daily_data[ticker]
        score = calculate_trend_score(df)
        bias = get_option_bias(score)
        last_price = df["close"].iloc[-1]
        last_atr = atr(df).iloc[-1] if "ATR" in df else 0

        plan = trade_plan(last_price, last_atr)

        # Ampel-Logik visuell
        if score >= 70:
            ampelfarbe = "🟢 Stark Bullish"
            st.success(ampelfarbe)
        elif score >= 40:
            ampelfarbe = "🟡 Neutral / vorsichtig"
            st.warning(ampelfarbe)
        else:
            ampelfarbe = "🔴 Bearish / meiden"
            st.error(ampelfarbe)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Trend-Score Breakdown**")
            st.write(f"Gesamt-Score: **{score}** / 100")
            st.write("• EMA20 > EMA50 → +40")
            st.write("• Close > EMA20 → +30")
            st.write("• RSI 50–70 → +30")

        with col2:
            st.markdown("**Option Bias & Strategie**")
            st.info(f"**{bias}** empfohlen")

        st.markdown("**Einfacher Trade-Plan (ATR-basiert)**")
        st.json(plan)

        st.caption("Hinweis: Das ist KEINE Handelsempfehlung – nur technisches Scoring. Risikomanagement selbst verantworten.")

    else:
        st.info("Wähle zuerst eine Aktie im Chart-Tab aus oder warte auf Daten.")
