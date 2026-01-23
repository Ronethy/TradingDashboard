from logic.snapshot import MarketSnapshot

def generate_trade_plan(snapshot: MarketSnapshot, score: int) -> dict | None:
    if score < 65:
        return None

    entry = snapshot.price
    stop = entry - 1.2 * snapshot.atr
    risk = entry - stop
    if risk <= 0:
        return None

    target = entry + 2.0 * risk
    rr = (target - entry) / risk

    if rr < 1.8:
        return None

    return {
        "Entry": round(entry, 2),
        "Stop": round(stop, 2),
        "Target": round(target, 2),
        "R:R": round(rr, 1)
    }
