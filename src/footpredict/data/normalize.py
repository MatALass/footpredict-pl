from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

def simplify_event(e: Dict[str, Any]) -> Dict[str, Any]:
    ts = e.get("startTimestamp")
    dt = datetime.fromtimestamp(ts) if ts else None

    return {
        "event_id": e.get("id"),
        "timestamp": ts,
        "date": dt.date().isoformat() if dt else None,
        "time": dt.strftime("%H:%M") if dt else None,
        "home": (e.get("homeTeam") or {}).get("name"),
        "away": (e.get("awayTeam") or {}).get("name"),
        "status": (e.get("status") or {}).get("type"),
        "home_goals": (e.get("homeScore") or {}).get("current"),
        "away_goals": (e.get("awayScore") or {}).get("current"),
        "round": (e.get("roundInfo") or {}).get("round"),
    }