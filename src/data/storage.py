from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(pd.Series(r).to_json(force_ascii=False))
            f.write("\n")

def write_parquet(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)

def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)