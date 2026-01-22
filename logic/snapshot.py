from dataclasses import dataclass

@dataclass
class MarketSnapshot:
    symbol: str
    price: float
    rsi: float
    ema9: float
    ema20: float
    ema50: float
    atr: float
    volume_ratio: float
    market_state: str  # PRE / OPEN / CLOSED
