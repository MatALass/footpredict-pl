import pandas as pd
import numpy as np

from features.form import build_form_features, build_goal_diff_features
from features.elo import build_elo_features
from features.rest_days import build_rest_days


def prepare_training_frame(df_matches: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    df = df_matches.copy()

    # garder matchs terminés
    df = df[df["status"] == "finished"].dropna(subset=["home_goals", "away_goals"]).copy()

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    # target
    df["y_homewin"] = (df["home_goals"] > df["away_goals"]).astype(int)

    # features
    form = build_form_features(df, window=form_window)
    elo = build_elo_features(df)
    goal_diff = build_goal_diff_features(df, window=form_window)
    rest = build_rest_days(df)

    out = pd.concat([df.reset_index(drop=True), form, elo, goal_diff, rest], axis=1)
    return out


def prepare_next_matches_frame(df_hist: pd.DataFrame, df_next: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    """
    Construit les features pour df_next en utilisant l'historique df_hist.
    df_hist: matchs finished avec scores
    df_next: matchs à venir (notstarted) sans scores
    """
    hist = df_hist.copy()
    hist = hist[hist["status"] == "finished"].dropna(subset=["home_goals", "away_goals"]).copy()
    hist["home_goals"] = hist["home_goals"].astype(int)
    hist["away_goals"] = hist["away_goals"].astype(int)

    # calcul features sur historique
    form_hist = build_form_features(hist, window=form_window)
    goal_hist = build_goal_diff_features(hist, window=form_window)
    elo_hist = build_elo_features(hist)
    rest_hist = build_rest_days(hist)

    hist_feat = pd.concat([hist.reset_index(drop=True), form_hist, goal_hist, elo_hist, rest_hist], axis=1)

    # dernier état connu par équipe (home/away)
    last_home = hist_feat.sort_values("date").groupby("home").tail(1).set_index("home")
    last_away = hist_feat.sort_values("date").groupby("away").tail(1).set_index("away")

    def get_team_snapshot(team: str):
        if team in last_home.index:
            return last_home.loc[team]
        if team in last_away.index:
            return last_away.loc[team]
        return None

    rows = []
    for _, r in df_next.iterrows():
        home = r["home"]
        away = r["away"]

        sh = get_team_snapshot(home)
        sa = get_team_snapshot(away)

        def val(snapshot, col, default=0.0):
            if snapshot is None:
                return default
            v = snapshot.get(col, default)
            if pd.isna(v):
                return default
            return float(v)

        elo_home = val(sh, "elo_home")
        elo_away = val(sa, "elo_away")
        home_adv = 60.0

        rows.append({
            "home_pts_mean": val(sh, "home_pts_mean"),
            "away_pts_mean": val(sa, "away_pts_mean"),
            "home_gf_mean": val(sh, "home_gf_mean"),
            "home_ga_mean": val(sh, "home_ga_mean"),
            "away_gf_mean": val(sa, "away_gf_mean"),
            "away_ga_mean": val(sa, "away_ga_mean"),
            "home_hist_n": val(sh, "home_hist_n"),
            "away_hist_n": val(sa, "away_hist_n"),

            "home_goal_diff_mean": val(sh, "home_goal_diff_mean"),
            "away_goal_diff_mean": val(sa, "away_goal_diff_mean"),

            # TODO: calcul rest_days "propre" (il faut le dernier match date de chaque équipe)
            "home_rest_days": 0.0,
            "away_rest_days": 0.0,

            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": (elo_home + home_adv) - elo_away,
            "elo_exp_home": 1.0 / (1.0 + 10 ** ((elo_away - (elo_home + home_adv)) / 400.0)),
        })

    feat_next = pd.DataFrame(rows)
    out = pd.concat([df_next.reset_index(drop=True), feat_next], axis=1)
    return out