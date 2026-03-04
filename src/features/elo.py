from __future__ import annotations

import math
from collections import defaultdict
import pandas as pd

def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

def build_elo_features(
    df_finished: pd.DataFrame,
    k: float = 20.0,
    home_adv: float = 60.0,
    start_rating: float = 1500.0,
) -> pd.DataFrame:
    """Elo calculé séquentiellement (pas de fuite du futur)."""
    rating = defaultdict(lambda: start_rating)
    rows = []

    for _, r in df_finished.iterrows():
        home, away = r["home"], r["away"]
        hg, ag = int(r["home_goals"]), int(r["away_goals"])

        rh = rating[home]
        ra = rating[away]

        # avantage domicile
        eh = _expected(rh + home_adv, ra)
        ea = 1.0 - eh

        # outcome
        if hg > ag:
            sh, sa = 1.0, 0.0
        elif hg < ag:
            sh, sa = 0.0, 1.0
        else:
            sh, sa = 0.5, 0.5

        rows.append({
            "elo_home": rh,
            "elo_away": ra,
            "elo_diff": (rh + home_adv) - ra,
            "elo_exp_home": eh,
        })

        # update
        rating[home] = rh + k * (sh - eh)
        rating[away] = ra + k * (sa - ea)

    return pd.DataFrame(rows)