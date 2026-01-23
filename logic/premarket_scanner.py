import pandas as pd

def scan_early_movers(daily_data, max_results=20, min_gap_pct=0.8):
    results = []
    for symbol, df in daily_data.items():
        if len(df) < 2:
            continue

        df = df.sort_index()

        prev_close = df["close"].iloc[-2]
        current_open = df["open"].iloc[-1]

        gap_pct = (current_open - prev_close) / prev_close * 100
        abs_gap = abs(gap_pct)

        if abs_gap >= min_gap_pct:
            results.append({
                "Symbol": symbol,
                "Gap %": round(gap_pct, 2),
                "Abs Gap": abs_gap,
                "Last": round(df["close"].iloc[-1], 2),
                "Volume": int(df["volume"].iloc[-1])
            })

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    return df_res.sort_values("Abs Gap", ascending=False).head(max_results)
