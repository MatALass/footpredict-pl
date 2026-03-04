from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES = [
    "home_pts_mean","away_pts_mean",
    "home_gf_mean","home_ga_mean",
    "away_gf_mean","away_ga_mean",
    "home_hist_n","away_hist_n",

    "home_goal_diff_mean",
    "away_goal_diff_mean",

    "home_rest_days",
    "away_rest_days",

    "elo_home","elo_away","elo_diff","elo_exp_home",
]

def train_model(df_train: pd.DataFrame) -> Pipeline:
    X = df_train[FEATURES].fillna(0.0)
    y = df_train["y_homewin"].astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500)),
    ])
    model.fit(X, y)
    return model