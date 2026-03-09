from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    api_base_url: str
    api_key: str
    premier_league_id: int
    seasons_back: int
    events_endpoint: str


def get_settings() -> Settings:
    return Settings(
        data_dir=Path(os.getenv("DATA_DIR", "data")),
        api_base_url=os.getenv("API_BASE_URL", "").strip(),
        api_key=os.getenv("API_KEY", "").strip(),
        premier_league_id=int(os.getenv("PREMIER_LEAGUE_ID", "17")),
        seasons_back=int(os.getenv("SEASONS_BACK", "4")),
        events_endpoint=os.getenv("EVENTS_ENDPOINT", "tournaments/get-matches").strip(),
    )