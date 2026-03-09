from __future__ import annotations

from asyncio import events
from pathlib import Path
from typing import List

import pandas as pd

from footpredict.api.sofascore import SofaScoreAPI
from footpredict.config import Settings
from footpredict.data.normalize import simplify_event
from footpredict.data.storage import write_jsonl, write_parquet

def build_matches_dataset(
    api: SofaScoreAPI,
    settings: Settings,
    tournament_id: int,
    events_endpoint: str,
) -> Path:
    season_id = api.get_latest_season_id(tournament_id)

    # raw events
    events = api.paginate_events(events_endpoint, tournament_id=tournament_id, season_id=season_id)
    raw_path = settings.data_dir / "raw" / f"events_t{tournament_id}_s{season_id}.jsonl"
    write_jsonl(raw_path, events)

        # normalized df
    rows = [simplify_event(e) for e in events]
    df = pd.DataFrame(rows)

    # --- CLEAN / DEDUP ---
    # si jamais df est vide (sécurité)
    if df.empty:
        raise RuntimeError("Aucun event normalisé. Vérifie l'endpoint et simplify_event().")

    # enlever lignes sans event_id
    df = df.dropna(subset=["event_id"]).copy()

    # trier pour garder la version la plus récente d'un event_id
    # (on utilise timestamp si dispo, sinon date)
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df = df.sort_values(["event_id", "timestamp"])
    else:
        df = df.sort_values(["event_id", "date"])

    # garder 1 seule ligne par match
    before = len(df)
    df = df.drop_duplicates(subset=["event_id"], keep="last")
    after = len(df)

    if before != after:
        print(f"⚠️ Dedup: {before-after} doublons supprimés (event_id).")

    # datetime + tri global
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    out_path = settings.data_dir / "processed" / f"matches_t{tournament_id}_s{season_id}.parquet"
    write_parquet(out_path, df)
    return out_path