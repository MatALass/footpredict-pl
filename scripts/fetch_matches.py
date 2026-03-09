from footpredict.config import get_settings
from footpredict.api.sofascore import SofaScoreAPI

def main():
    s = get_settings()
    api = SofaScoreAPI(s)

    season_id = api.get_latest_season_id(s.premier_league_id)
    data = api.get_next_matches(s.premier_league_id, season_id, page_index=0)

    print("season_id:", season_id)
    print("events:", len(data.get("events", [])))
    print("hasNextPage:", data.get("hasNextPage"))

if __name__ == "__main__":
    main()