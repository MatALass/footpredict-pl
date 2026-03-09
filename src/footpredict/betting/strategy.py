from __future__ import annotations

def should_bet(p: float, odds_decimal: float, margin: float = 0.0) -> bool:
    """
    Value bet si p > (1/odds) + margin
    margin peut être 0.01 par ex pour être plus strict.
    """
    implied = 1.0 / odds_decimal
    return p > (implied + margin)