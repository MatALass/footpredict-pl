from config import get_settings
from api.sofascore import SofaScoreAPI
from data.build_dataset import build_matches_dataset
from data.storage import read_parquet
from validation.checks import check_matches_df

def main():
    s = get_settings()
    api = SofaScoreAPI(s)

    # IMPORTANT: remplace si ton RapidAPI utilise un autre endpoint saison:
    EVENTS_ENDPOINT = "tournaments/get-matches"

    out_path = build_matches_dataset(api, s, tournament_id=s.premier_league_id, events_endpoint=EVENTS_ENDPOINT)
    df = read_parquet(out_path)
    check_matches_df(df)
    print("✅ dataset:", out_path, "rows:", len(df))
    

if __name__ == "__main__":
    main()