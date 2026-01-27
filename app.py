import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime, timedelta
import requests

# yfinance optional (Fallback für längere 15-Min-Historie)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.warning("yfinance nicht installiert → 15-Min-Charts haben nur sehr kurze Historie. "
               "Installiere mit 'pip install yfinance' für bessere Daten.")

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

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
from logic.additional_indicators import rsi_divergence, macd_info

st.set_page_config(page_title="Momentum Dashboard", layout="wide")

client = StockHistoricalDataClient(
    st.secrets["ALPACA_API_KEY"],
    st.secrets["ALPACA_SECRET_KEY"]
)

alpha_vantage_key = st.secrets.get("ALPHA_VANTAGE_KEY", None)
if not alpha_vantage_key:
    st.warning("Alpha Vantage Key fehlt → News-Funktion deaktiviert")

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"

ny_tz = pytz.timezone("America/New_York")
now_ny = datetime.now(ny_tz)
market_state = "PRE" if now_ny.hour < 9 else "OPEN" if now_ny.hour < 16 else "CLOSED"

st.title("📊 Smart Momentum Trading Dashboard")
st.write(f"Marktstatus: {market_state} | Zeit: {now_ny.strftime('%Y-%m-%d %H:%M')}")

if market_state == "PRE":
    st.warning("Pre-Market: Viele Scores & Ampele sind eingeschränkt – warte auf Open (9:30 ET)")

# Globale Symbol-Auswahl oben
st.subheader("Ticker auswählen (für Chart & Trading)")
ticker = st.selectbox(
    "Aktie wählen",
    options=SP500_SYMBOLS,
    index=SP500_SYMBOLS.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in SP500_SYMBOLS else 0,
    key="global_ticker_select"
)

if ticker != st.session_state.selected_ticker:
    st.session_state.selected_ticker = ticker
    st.rerun()

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

@st.cache_data(ttl=60)
def load_bars(ticker, _timeframe, start, end):
    # Maximal 6 Monate zurück – hart begrenzen
    max_start = now_ny - timedelta(days=180)
    start = max(start, max_start)

    # yfinance für 15-Minuten (längere Historie)
    if _timeframe == TimeFrame(15, TimeFrameUnit.Minute) and YFINANCE_AVAILABLE:
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval="15m",
                prepost=False,
                progress=False
            )
            if not df.empty:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC').tz_convert(ny_tz)
                else:
                    df.index = df.index.tz_convert(ny_tz)
                df = df.sort_index()
                return df
        except Exception as e:
            st.caption(f"yfinance-Fehler {ticker}: {str(e)}")

    # Alpaca-Fallback für andere Intervalle
    try:
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=_timeframe,
            start=start,
            end=end,
            feed="iex",
            limit=10000
        )
        bars = client.get_stock_bars(req).df
        if bars.empty:
            return pd.DataFrame()

        # MultiIndex bereinigen
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(ticker, level=1, drop_level=True) if ticker in bars.index.levels[1] else bars.reset_index(level=1, drop=True)

        if 'symbol' in bars.columns:
            bars = bars.drop(columns=['symbol'])
        if bars.index.name == 'symbol' or bars.index.name == ticker:
            bars = bars.reset_index(drop=True)

        bars = bars.reset_index(drop=False)
        timestamp_col = next((col for col in bars.columns if 'time' in col.lower() or 'date' in col.lower()), bars.columns[0])
        bars = bars.set_index(timestamp_col)

        # Unix-Timestamp parsen (ms oder s)
        if not isinstance(bars.index, pd.DatetimeIndex):
            # Versuche als ms, dann s
            try:
                bars.index = pd.to_datetime(bars.index, unit='ms', errors='coerce')
            except ValueError:
                bars.index = pd.to_datetime(bars.index, unit='s', errors='coerce')

        bars = bars[bars.index.notnull()]

        if bars.index.tz is None:
            bars.index = bars.index.tz_localize('UTC').tz_convert(ny_tz)
        else:
            bars.index = bars.index.tz_convert(ny_tz)

        bars = bars.sort_index()
        return bars
    except Exception as e:
        st.caption(f"Alpaca-Fehler {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_stock_news(ticker, api_key, limit=3):
    if not api_key:
        return []
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit={limit}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("feed", [])[:limit]
        else:
            return []
    except:
        return []

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
    st.subheader("🔥 Early Movers – mit Farben & News für Top-Kandidaten")

    enhanced_movers = []
    for sym, df in daily_data.items():
        if len(df) < 2:
            continue

        prev_close = df["close"].iloc[-2]
        current_open = df["open"].iloc[-1]
        gap_pct = (current_open - prev_close) / prev_close * 100
        abs_gap = abs(gap_pct)

        volume = df["volume"].iloc[-1]
        vol_avg = df["volume"].mean()
        vol_ratio = volume / vol_avg if vol_avg > 0 else 1.0

        # Snapshot für Score
        df_ind = df.copy()
        df_ind["ema9"] = ema(df_ind["close"], 9)
        df_ind["ema20"] = ema(df_ind["close"], 20)
        df_ind["ema50"] = ema(df_ind["close"], 50)
        df_ind["rsi"] = rsi(df_ind["close"])
        df_ind["atr"] = atr(df_ind)
        df_ind.dropna(inplace=True)

        score = 0
        if not df_ind.empty:
            latest = df_ind.iloc[-1]
            vol_ratio_ind = latest["volume"] / df_ind["volume"].mean() if df_ind["volume"].mean() > 0 else 1.0
            snap = MarketSnapshot(sym, latest["close"], latest["rsi"], latest["ema9"], latest["ema20"], latest["ema50"], latest["atr"], vol_ratio_ind, market_state)
            score = calculate_trend_score(snap)

        enhanced_movers.append({
            "Symbol": sym,
            "Gap %": round(gap_pct, 2),
            "Abs Gap": abs_gap,
            "Vol Ratio": round(vol_ratio, 2),
            "Score": score
        })

    if enhanced_movers:
        df_movers = pd.DataFrame(enhanced_movers)
        df_movers = df_movers.sort_values("Abs Gap", ascending=False).head(20)

        # Empfehlung
        def get_recommendation(row):
            gap = row["Gap %"]
            score = row["Score"]
            if score >= 70 or gap > 3:
                return "Kaufen / Long priorisieren"
            elif score >= 50 or gap > 2:
                return "Beobachten / Watchlist"
            else:
                return "Vermeiden"

        df_movers["Empfehlung"] = df_movers.apply(get_recommendation, axis=1)

        # Farbliche Hervorhebung
        def highlight_row(row):
            rec = row["Empfehlung"]
            if "Kaufen" in rec:
                return ['background-color: #d4edda; color: black'] * len(row)
            elif "Beobachten" in rec:
                return ['background-color: #fff3cd; color: black'] * len(row)
            else:
                return ['background-color: #f8d7da; color: black'] * len(row)

        styled = df_movers.style.apply(highlight_row, axis=1)

        st.dataframe(styled, width='stretch', hide_index=True)

        # News nur für Top 5 laden
        st.subheader("News zu Top Early Movers")
        top_symbols = df_movers.head(5)["Symbol"].tolist()
        for sym in top_symbols:
            news = get_stock_news(sym, alpha_vantage_key, limit=2)
            if news:
                st.markdown(f"**{sym}** (Score {df_movers[df_movers['Symbol'] == sym]['Score'].values[0]})")
                for item in news:
                    title = item.get("title", "No title")
                    url = item.get("url", "#")
                    sentiment = item.get("overall_sentiment_label", "Neutral")
                    st.markdown(f"- [{title}]({url}) – Sentiment: **{sentiment}**")
                st.markdown("---")
            else:
                st.caption(f"Keine News für {sym}")
    else:
        st.info("Keine Early Movers gefunden")

# ── S&P 500 Scanner ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("🧠 S&P 500 Scanner – mit Farben & News für Top-Kandidaten")

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
        bias = get_option_bias(snap, score)

        rec = "Vermeiden"
        if score >= 70:
            rec = "Kaufen / Long priorisieren"
        elif score >= 50:
            rec = "Beobachten / Watchlist"

        rows.append({
            "Symbol": sym,
            "Score": score,
            "Bias": bias,
            "Empfehlung": rec
        })

    if rows:
        df_scores = pd.DataFrame(rows).sort_values("Score", ascending=False).head(30)

        def highlight_scanner(row):
            rec = row["Empfehlung"]
            if "Kaufen" in rec:
                return ['background-color: #d4edda; color: black'] * len(row)
            elif "Beobachten" in rec:
                return ['background-color: #fff3cd; color: black'] * len(row)
            else:
                return ['background-color: #f8d7da; color: black'] * len(row)

        styled_scanner = df_scores.style.apply(highlight_scanner, axis=1)

        st.dataframe(styled_scanner, width='stretch', hide_index=True)

        # News nur für Top 5 laden
        st.subheader("News zu Top S&P 500 Kandidaten")
        top_symbols_scanner = df_scores.head(5)["Symbol"].tolist()
        for sym in top_symbols_scanner:
            news = get_stock_news(sym, alpha_vantage_key, limit=2)
            if news:
                st.markdown(f"**{sym}** (Score {df_scores[df_scores['Symbol'] == sym]['Score'].values[0]})")
                for item in news:
                    title = item.get("title", "No title")
                    url = item.get("url", "#")
                    sentiment = item.get("overall_sentiment_label", "Neutral")
                    st.markdown(f"- [{title}]({url}) – Sentiment: **{sentiment}**")
                st.markdown("---")
            else:
                st.caption(f"Keine News für {sym}")
    else:
        st.warning("Keine gültigen Scores berechnet")

# ── Chart Analyse ───────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("📈 Chart Analyse")

    timeframe_options = {
        "15 Minuten": TimeFrame(15, TimeFrameUnit.Minute),
        "Täglich": TimeFrame.Day,
        "Wöchentlich": TimeFrame.Week
    }

    timeframe_str = st.selectbox("Zeitrahmen wählen", list(timeframe_options.keys()), index=1)  # Default Täglich
    timeframe = timeframe_options[timeframe_str]

    ticker = st.session_state.selected_ticker
    if ticker in daily_data and not daily_data[ticker].empty:
        # Dynamische Startzeit je nach Zeitrahmen (größerer Bereich)
        if timeframe_str == "15 Minuten":
            start = now_ny - timedelta(days=10)   # 10 Tage → viele 15-Min-Kerzen
        elif timeframe_str == "Täglich":
            start = now_ny - timedelta(days=730)  # 2 Jahre
        else:  # Wöchentlich
            start = now_ny - timedelta(days=365*5)  # 5 Jahre

        df = load_bars(ticker, timeframe, start, now_ny + timedelta(days=1))
        if df.empty:
            st.warning("Keine Daten für diesen Zeitrahmen")
        else:
            df["ema20"] = ema(df["close"], 20)
            df["ema50"] = ema(df["close"], 50)
            df["RSI"] = rsi(df["close"])
            df["ATR"] = atr(df)

            # MACD berechnen
            ema_fast = df['close'].ewm(span=12, adjust=False).mean()
            ema_slow = df['close'].ewm(span=26, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line

            # Divergenz-Punkte finden
            div = rsi_divergence(df)
            low_points = []
            if "Bullish" in div or "Bearish" in div:
                recent_low_idx = df['low'].argmin()
                prev_low_idx = df['low'][:-30].argmin() if len(df) > 30 else None
                low_points = [recent_low_idx] if recent_low_idx is not None else []
                if prev_low_idx is not None:
                    low_points.append(prev_low_idx)

            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                row_heights=[0.55, 0.15, 0.30, 0.2]
            )

            fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="OHLC"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], name="EMA 20", line=dict(color="#00BFFF")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], name="EMA 50", line=dict(color="#FF8C00")), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.update_layout(height=800, title=f"{ticker} – Daily", hovermode="x unified")
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

            # Ampel-Logik visuell
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

            # Zusatz-Indikatoren (neu)
            st.subheader("Zusatz-Indikatoren")
            col_div, col_macd = st.columns(2)

            with col_div:
                div = rsi_divergence(df_ind)
                st.markdown("**RSI-Divergenz** (letzte 30 Bars)")
                if "Bullish" in div:
                    st.success(div)
                elif "Bearish" in div:
                    st.error(div)
                else:
                    st.info(div)

            with col_macd:
                macd = macd_info(df_ind)
                st.markdown("**MACD (12,26,9)**")
                if macd["MACD"] is not None:
                    st.write(f"MACD: {macd['MACD']} | Signal: {macd['Signal']}")
                    st.write(f"Histogramm: {macd['Histogramm']}")
                    if "Bullish" in macd["Interpretation"]:
                        st.success(macd["Interpretation"])
                    elif "Bearish" in macd["Interpretation"]:
                        st.error(macd["Interpretation"])
                    else:
                        st.info(macd["Interpretation"])
                else:
                    st.info(macd["text"])

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

            # News für ausgewähltes Symbol laden
            st.subheader("News zu dieser Aktie")
            news = get_stock_news(ticker, alpha_vantage_key, limit=3)
            if news:
                for item in news:
                    title = item.get("title", "No title")
                    url = item.get("url", "#")
                    sentiment = item.get("overall_sentiment_label", "Neutral")
                    st.markdown(f"- [{title}]({url}) – Sentiment: **{sentiment}**")
                st.markdown("---")
            else:
                st.info("Keine News verfügbar")

        else:
            st.warning("Keine Daten nach Berechnung")
    else:
        st.info("Wähle einen Ticker mit ausreichend Historie")
