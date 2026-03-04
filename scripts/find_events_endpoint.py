from config import get_settings
from api.sofascore import SofaScoreAPI

CANDIDATES = [
    "tournaments/get-events",
    "tournaments/get-matches",
    "unique-tournaments/get-events",
    "unique-tournaments/get-matches",
    "tournaments/get-events-by-round",
    "tournaments/get-round-events",
]

def main():
    s = get_settings()
    api = SofaScoreAPI(s)

    tid = s.premier_league_id
    sid = api.get_latest_season_id(tid)

    print("tournament:", tid, "season:", sid)
    print("Testing endpoints...")

    for ep in CANDIDATES:
        try:
            events = api.paginate_events(ep, tournament_id=tid, season_id=sid, max_pages=3)
            if not events:
                print(ep, "-> 0 events")
                continue

            statuses = {}
            for e in events:
                st = ((e.get("status") or {}).get("type") or "").lower()
                statuses[st] = statuses.get(st, 0) + 1

            print(ep, "->", len(events), "events | statuses:", statuses)
        except Exception as e:
            print(ep, "-> ERROR:", str(e)[:120])

if __name__ == "__main__":
    main()