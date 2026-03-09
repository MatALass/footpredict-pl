from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeasonInfo:
    season_id: int
    name: str
    year_start: int | None
    year_end: int | None


def parse_season_label(season_name: str) -> tuple[int | None, int | None]:
    """
    Essaie de convertir un nom de saison type:
    - '2024/2025'
    - '2023-2024'
    - '2022'
    en (year_start, year_end)
    """
    text = season_name.strip().replace("-", "/")
    parts = text.split("/")

    if len(parts) == 2:
        try:
            y1 = int(parts[0])
            y2 = int(parts[1])
            return y1, y2
        except ValueError:
            return None, None

    if len(parts) == 1:
        try:
            y = int(parts[0])
            return y, y + 1
        except ValueError:
            return None, None

    return None, None


def normalize_seasons(raw_seasons: list[dict]) -> list[SeasonInfo]:
    seasons: list[SeasonInfo] = []

    for item in raw_seasons:
        season_id = int(item["id"])
        name = str(item.get("name") or item.get("year") or item.get("season") or season_id)
        y1, y2 = parse_season_label(name)
        seasons.append(
            SeasonInfo(
                season_id=season_id,
                name=name,
                year_start=y1,
                year_end=y2,
            )
        )

    seasons.sort(
        key=lambda s: (
            s.year_start if s.year_start is not None else -1,
            s.season_id,
        )
    )
    return seasons


def select_last_n_seasons(seasons: list[SeasonInfo], n: int) -> list[SeasonInfo]:
    if n <= 0:
        raise ValueError("n must be > 0")
    return seasons[-n:]