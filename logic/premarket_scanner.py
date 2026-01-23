from __future__ import annotations

import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

def scan_early_movers(symbols, client, max_results=20, min_gap_pct=0.8):
    results = []
    for symbol in symbols:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                limit=3
            )
            bars = client.get_stock_bars(req).df
            if len(bars) < 2:
                continue

            prev_close = bars["close"].iloc[-2]
            current_open = bars["open"].iloc[-1]
            gap = (current_open - prev_close) / prev_close * 100

            if abs(gap) >= min_gap_pct:
                results.append({
                    "Symbol": symbol,
                    "Gap %": round(gap, 2),
                    "Abs Gap": abs(gap),
                    "Last": round(bars["close"].iloc[-1], 2),
                    "Volume": int(bars["volume"].iloc[-1])
                })
        except:
            pass

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df.sort_values("Abs Gap", ascending=False).head(max_results)
