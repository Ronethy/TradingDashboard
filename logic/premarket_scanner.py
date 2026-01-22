import pandas as pd
from logic.trend_score import calculate_trend_score

def scan_early_movers(data_dict):
    results = []

    for symbol, df in data_dict.items():
        if len(df) < 2:
            continue

        prev_close = df["close"].iloc[-2]
        open_price = df["open"].iloc[-1]

        gap = (open_price - prev_close) / prev_close * 100

        if abs(gap) >= 1:
            score = calculate_trend_score(df)
            results.append({
                "Symbol": symbol,
                "Gap %": round(gap, 2),
                "Trend-Score": score
            })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("Gap %", ascending=False).head(20)
