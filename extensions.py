    # Proxy nur wenn genug Daten
    adv_dec_proxy = "N/A"
    if 'sp500' in locals() and not sp500.empty and len(sp500) >= 2:
        pct_changes = sp500['Close'].pct_change().dropna()
        if not pct_changes.empty:
            mean_pct = pct_changes.mean()
            if pd.isna(mean_pct):
                adv_dec_proxy = "neutral (Daten unklar)"
            else:
                adv_dec_proxy = "positiv" if mean_pct > 0 else "negativ" if mean_pct < 0 else "neutral"

    new_highs_lows = "N/A"
    if 'sp500' in locals() and not sp500.empty and len(sp500) >= 252:
        rolling_max = sp500['Close'].rolling(252).max()
        rolling_min = sp500['Close'].rolling(252).min()
        highs_count = (sp500['Close'] == rolling_max).sum()
        lows_count = (sp500['Close'] == rolling_min).sum()
        if highs_count > lows_count:
            new_highs_lows = "mehr Highs"
        elif lows_count > highs_count:
            new_highs_lows = "mehr Lows"
        else:
            new_highs_lows = "ausgeglichen"
