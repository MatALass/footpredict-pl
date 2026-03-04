from __future__ import annotations

import pandas as pd

def check_matches_df(df: pd.DataFrame) -> None:
    required = {"event_id","date","home","away","status","home_goals","away_goals"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    if df["event_id"].isna().any():
        raise ValueError("event_id contient des NA")

    dup = df["event_id"].duplicated().sum()
    if dup:
        raise ValueError(f"event_id dupliqués: {dup}")

    # finished -> scores présents
    fin = df[df["status"] == "finished"]
    if fin["home_goals"].isna().any() or fin["away_goals"].isna().any():
        raise ValueError("Matchs finished sans score (home_goals/away_goals NA)")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("Les dates ne sont pas triées")