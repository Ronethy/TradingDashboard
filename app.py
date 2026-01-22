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

st.set_page_config(layout="wide")

client = StockHistoricalDataClient(
    st.secrets["ALPACA_API_KEY"],
    st.secrets["ALPACA_SECRET_KEY"]
)

st.title("📊 Smart Momentum Trading Dashboard")

ny = datetime.now(pytz.timezone("US/Eastern"))
market_state = "PRE" if ny.hour < 9 else "OPEN" if ny.hour < 16 else "CLOSED"
st.write("Marktstatus:", market_state)

tab1, tab2 = st.tabs(["🔥 Early Movers", "🧠 S&P 500 Scanner"])

# -------- EARLY MOVERS --------
with tab1:
    df = scan_early_movers(SP500_SYMBOLS, client)
    st.dataframe(df, use_container_width=True)

# -------- S&P 500 SCANNER --------
with tab2:
    results = []

    for s in SP500_SYMBOLS:
        try:
            req = StockBarsRequest(s, TimeFrame.Day, limit=60)
            df = client.get_stock_bars(req).df
            if df.empty:
                continue

            df["ema9"] = ema(df["close"], 9)
            df["ema20"] = ema(df["close"], 20)
            df["ema50"] = ema(df["close"], 50)
            df["rsi"] = rsi(df["close"])
            df["atr"] = atr(df)
            df.dropna(inplace=True)

            last = df.iloc[-1]

            snap = MarketSnapshot(
                s,
                last["close"],
                last["rsi"],
                last["ema9"],
                last["ema20"],
                last["ema50"],
                last["atr"],
                last["volume"] / df["volume"].mean(),
                market_state
            )

            score = calculate_trend_score(snap)
            bias = option_bias(snap, score)
            plan = trade_plan(snap, score)

            results.append({
                "Symbol": s,
                "Score": score,
                "Bias": bias,
                "Entry": plan["Entry"] if plan else None,
                "Stop": plan["Stop"] if plan else None,
                "Target": plan["Target"] if plan else None
            })

        except Exception:
            pass

    res = pd.DataFrame(results).sort_values("Score", ascending=False).head(20)
    st.dataframe(res, use_container_width=True)
