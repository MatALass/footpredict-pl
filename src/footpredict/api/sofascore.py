from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from footpredict.api.client import ApiClient
from footpredict.config import Settings

class SofaScoreAPI:
    def __init__(self, settings: Settings):
        self.client = ApiClient(
            base_url=f"https://{settings.sofascore_host}",
            headers={
                "x-rapidapi-host": settings.sofascore_host,
                "x-rapidapi-key": settings.rapidapi_key,
            },
        )

    def get_seasons(self, tournament_id: int) -> List[Dict[str, Any]]:
        data = self.client.get_json("tournaments/get-seasons", {"tournamentId": str(tournament_id)})
        seasons = (data or {}).get("seasons", [])
        return seasons

    def get_latest_season_id(self, tournament_id: int) -> int:
        seasons = self.get_seasons(tournament_id)
        if not seasons:
            raise RuntimeError("Aucune saison trouvée.")
        return max(int(s["id"]) for s in seasons if "id" in s)

    def get_next_matches(self, tournament_id: int, season_id: int, page_index: int = 0) -> Dict[str, Any]:
        return self.client.get_json(
            "tournaments/get-next-matches",
            {"tournamentId": str(tournament_id), "seasonId": str(season_id), "pageIndex": str(page_index)},
        ) or {"events": [], "hasNextPage": False}

    def paginate_events(
        self,
        endpoint: str,
        tournament_id: int,
        season_id: int,
        max_pages: int = 200,
    ) -> List[Dict[str, Any]]:
        """endpoint attendu: ex 'tournaments/get-events' (varie selon RapidAPI)."""
        all_events: List[Dict[str, Any]] = []
        page = 0

        while page < max_pages:
            params = {
                "tournamentId": str(tournament_id),
                "seasonId": str(season_id),
                "pageIndex": str(page),
            }
            data = self.client.get_json(endpoint, params) or {}
            events = data.get("events", []) or []
            all_events.extend(events)

            has_next = data.get("hasNextPage")
            if has_next is False or not events:
                break

            page += 1

        return all_events