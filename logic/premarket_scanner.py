import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

def scan_early_movers(symbols, client, max_results=20, min_gap_pct=0.8):
    rows = []
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
            gap_pct = (current_open - prev_close) / prev_close * 100
            if abs(gap_pct) >= min_gap_pct:
                rows.append({
                    "Symbol": symbol,
                    "Gap %": round(gap_pct, 2),
                    "Abs Gap": abs(gap_pct),
                    "Last": round(bars["close"].iloc[-1], 2),
                    "Volume": int(bars["volume"].iloc[-1])
                })
        except:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("Abs Gap", ascending=False).head(max_results)
