import pandas as pd
from pathlib import Path
import joblib

from config import get_settings
from api.sofascore import SofaScoreAPI
from features.pipeline import prepare_next_matches_frame
from models.train import FEATURES


def main():
    s = get_settings()
    api = SofaScoreAPI(s)

    tournament_id = s.premier_league_id
    season_id = api.get_latest_season_id(tournament_id)

    print("Season:", season_id)

    data = api.get_next_matches(tournament_id, season_id)
    events = data["events"]

    rows = []
    for e in events:
        rows.append({
            "event_id": e["id"],
            "timestamp": e["startTimestamp"],
            "date": pd.to_datetime(e["startTimestamp"], unit="s"),
            "home": e["homeTeam"]["name"],
            "away": e["awayTeam"]["name"],
            "status": "notstarted",
            "home_goals": None,
            "away_goals": None
        })

    df_next = pd.DataFrame(rows)
    print("Next matches:", len(df_next))

    # historique (dernier parquet processed)
    processed = list((s.data_dir / "processed").glob("matches_t*.parquet"))
    if not processed:
        raise RuntimeError("Aucun dataset processed trouvé. Lance scripts/build_dataset.py d'abord.")
    hist_path = sorted(processed)[-1]
    df_hist = pd.read_parquet(hist_path)

    df_next_features = prepare_next_matches_frame(df_hist=df_hist, df_next=df_next, form_window=5)

    model = joblib.load("data/reports/model.joblib")

    X = df_next_features[FEATURES].fillna(0.0)
    proba = model.predict_proba(X)[:, 1]
    df_next_features["home_win_proba"] = proba

    out_path = Path("data/reports/predictions_next.parquet")
    df_next_features.to_parquet(out_path, index=False)

    print("✅ Predictions saved:", out_path)

    show = df_next_features[["date", "home", "away", "home_win_proba"]].sort_values("date")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()