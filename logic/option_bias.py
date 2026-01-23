from __future__ import annotations

from logic.snapshot import MarketSnapshot

def get_option_bias(snapshot: MarketSnapshot, score: int) -> str:
    if score < 60:
        return "NEUTRAL / kein Options-Trade"

    if snapshot.ema9 > snapshot.ema20 > snapshot.ema50:
        return "BULLISH → CALLs / Long-Bias"

    if snapshot.ema9 < snapshot.ema20 < snapshot.ema50:
        return "BEARISH → PUTs / Short-Bias"

    return "NEUTRAL / Spreads oder warten"
