def trade_plan(price, atr):
    return {
        "Entry": round(price, 2),
        "Stop": round(price - atr, 2),
        "Target": round(price + (2 * atr), 2)
    }
