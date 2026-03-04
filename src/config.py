from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    rapidapi_key: str
    sofascore_host: str = os.getenv("SOFASCORE_HOST", "sofascore.p.rapidapi.com")
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))

    # ligues (IDs SofaScore)
    premier_league_id: int = 17

def get_settings() -> Settings:
    key = os.getenv("RAPIDAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("RAPIDAPI_KEY manquant. Mets-le dans .env ou variable d'environnement.")
    return Settings(rapidapi_key=key)