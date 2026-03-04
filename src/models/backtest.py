from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from models.train import train_model, FEATURES

def walk_forward_backtest(df_ml: pd.DataFrame, n_splits: int = 5) -> dict:
    X = df_ml[FEATURES].fillna(0.0)
    y = df_ml["y_homewin"].astype(int)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    accs, lls, briers = [], [], []
    oof = np.zeros(len(df_ml))

    for tr, te in tscv.split(X):
        model = train_model(df_ml.iloc[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        pred = (proba >= 0.5).astype(int)

        accs.append(accuracy_score(y.iloc[te], pred))
        lls.append(log_loss(y.iloc[te], np.c_[1-proba, proba]))
        briers.append(brier_score_loss(y.iloc[te], proba))
        oof[te] = proba

    return {
        "accuracy_mean": float(np.mean(accs)),
        "logloss_mean": float(np.mean(lls)),
        "brier_mean": float(np.mean(briers)),
        "oof_proba": oof,
    }