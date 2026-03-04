from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

def build_form_features(df_finished: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    last_points = defaultdict(lambda: deque(maxlen=window))
    last_gf = defaultdict(lambda: deque(maxlen=window))
    last_ga = defaultdict(lambda: deque(maxlen=window))

    rows = []
    for _, r in df_finished.iterrows():
        home, away = r["home"], r["away"]
        hg, ag = int(r["home_goals"]), int(r["away_goals"])

        def mean0(dq): return float(np.mean(dq)) if len(dq) else 0.0

        rows.append({
            "home_pts_mean": mean0(last_points[home]),
            "away_pts_mean": mean0(last_points[away]),
            "home_gf_mean": mean0(last_gf[home]),
            "home_ga_mean": mean0(last_ga[home]),
            "away_gf_mean": mean0(last_gf[away]),
            "away_ga_mean": mean0(last_ga[away]),
            "home_hist_n": len(last_points[home]),
            "away_hist_n": len(last_points[away]),
        })

        if hg > ag: hp, ap = 3, 0
        elif hg < ag: hp, ap = 0, 3
        else: hp, ap = 1, 1

        last_points[home].append(hp); last_points[away].append(ap)
        last_gf[home].append(hg); last_ga[home].append(ag)
        last_gf[away].append(ag); last_ga[away].append(hg)

    return pd.DataFrame(rows)

def build_goal_diff_features(df_finished, window=5):

    import numpy as np
    from collections import defaultdict, deque

    last_diff = defaultdict(lambda: deque(maxlen=window))

    rows = []

    for _, r in df_finished.iterrows():

        home = r["home"]
        away = r["away"]

        def mean0(dq):
            return float(np.mean(dq)) if len(dq) else 0.0

        rows.append({
            "home_goal_diff_mean": mean0(last_diff[home]),
            "away_goal_diff_mean": mean0(last_diff[away]),
        })

        diff = r["home_goals"] - r["away_goals"]

        last_diff[home].append(diff)
        last_diff[away].append(-diff)

    return pd.DataFrame(rows)