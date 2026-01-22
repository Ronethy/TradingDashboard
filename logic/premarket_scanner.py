import pandas as pd

def scan_early_movers(symbols, client, max_results=20):
    movers = []

    for symbol in symbols:
        try:
            # Vortages-Close
            daily = client.get_stock_bars(
                symbol_or_symbols=symbol,
                timeframe="1Day",
                limit=2
            ).df

            if daily.empty or len(daily) < 2:
                continue

            prev_close = daily.iloc[-2]["close"]

            # Letzte verfügbare Minuten (Free Data)
            intraday = client.get_stock_bars(
                symbol_or_symbols=symbol,
                timeframe="1Min",
                limit=30
            ).df

            if intraday.empty:
                continue

            last_price = intraday.iloc[-1]["close"]
            volume = intraday["volume"].sum()

            gap_pct = (last_price - prev_close) / prev_close * 100

            # Gatekeeper (konservativ)
            if abs(gap_pct) < 2:
                continue
            if last_price < 5:
                continue
            if volume < 10_000:
                continue

            movers.append({
                "Symbol": symbol,
                "Preis": round(last_price, 2),
                "Gap %": round(gap_pct, 2),
                "Richtung": "📈 Gap-Up" if gap_pct > 0 else "📉 Gap-Down",
                "Pre-Market Volumen": int(volume)
            })

        except:
            continue

    if not movers:
        return pd.DataFrame()

    df = pd.DataFrame(movers)

    # Ranking nach absoluter Bewegung
    df["Abs Gap"] = df["Gap %"].abs()
    df = df.sort_values("Abs Gap", ascending=False)

    return df.head(max_results)
