from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

@dataclass
class ApiClient:
    base_url: str
    headers: Dict[str, str]
    timeout_sec: int = 25
    sleep_between_calls_sec: float = 0.12
    max_retries: int = 3

    def get_json(self, path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        last_err = None

        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.get(url, headers=self.headers, params=params, timeout=self.timeout_sec)

                # 204 = pas de contenu (ok)
                if r.status_code == 204:
                    return None

                # parfois 200 mais body = {"error": ...}
                data = r.json()
                if isinstance(data, dict) and "error" in data:
                    raise RuntimeError(f"API error: {data['error']} url={r.url}")

                return data

            except Exception as e:
                last_err = e
                # petit backoff
                time.sleep(self.sleep_between_calls_sec * attempt)

        raise RuntimeError(f"GET failed after retries. path={path} params={params} err={last_err}")