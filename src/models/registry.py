from __future__ import annotations

from pathlib import Path
import joblib

def save_model(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_model(path: Path):
    return joblib.load(path)