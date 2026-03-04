import pandas as pd


def build_rest_days(df):

    last_match = {}

    rows = []

    for _, r in df.iterrows():

        home = r["home"]
        away = r["away"]

        date = pd.to_datetime(r["date"])

        home_rest = None
        away_rest = None

        if home in last_match:
            home_rest = (date - last_match[home]).days

        if away in last_match:
            away_rest = (date - last_match[away]).days

        rows.append({
            "home_rest_days": home_rest,
            "away_rest_days": away_rest
        })

        last_match[home] = date
        last_match[away] = date

    return pd.DataFrame(rows)