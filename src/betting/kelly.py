from __future__ import annotations

def kelly_fraction(p: float, odds_decimal: float) -> float:
    b = odds_decimal - 1.0
    q = 1.0 - p
    return (p * b - q) / b

def stake(bankroll: float, p: float, odds_decimal: float, kelly_mult: float = 0.25, max_frac: float = 0.05) -> float:
    f = kelly_fraction(p, odds_decimal)
    if f <= 0:
        return 0.0
    f = min(f, max_frac)
    return bankroll * (kelly_mult * f)