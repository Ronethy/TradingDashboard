def trade_plan(snapshot, score):
    if score < 65:
        return None

    entry = snapshot.price
    stop = entry - 1.2 * snapshot.atr
    target = entry + 2 * (entry - stop)

    rr = (target - entry) / (entry - stop)

    if rr < 1.8:
        return None

    return {
        "Entry": round(entry, 2),
        "Stop": round(stop, 2),
        "Target": round(target, 2),
        "RR": round(rr, 2)
    }
