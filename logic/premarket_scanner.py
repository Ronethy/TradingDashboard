import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

def scan_early_movers(symbols, client, max_results=20):
    rows = []

    for s in symbols:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=s,
                timeframe=TimeFrame.Day,
                limit=2
            )
            df = client.get_stock_bars(req).df
            if len(df) < 2:
                continue

            prev, curr = df.iloc[-2], df.iloc[-1]
            gap = (curr["open"] - prev["close"]) / prev["close"] * 100

            rows.append({
                "Symbol": s,
                "Gap %": round(gap, 2),
                "Abs Gap": abs(gap)
            })
        except Exception:
            pass

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values("Abs Gap", ascending=False).head(max_results)
