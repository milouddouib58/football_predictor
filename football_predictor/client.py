# file: football_predictor/client.py
import requests
import time
import random
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Optional, Any

from football_predictor.settings import BASE_URL, VERSION
from football_predictor.utils import log

class AnalysisCache:
    def __init__(self):
        self.cache = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self.cache.get(key)
            if not item: return None
            value, expires_at = item
            if time.time() > expires_at:
                self.cache.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: int):
        with self._lock:
            self.cache[key] = (value, time.time() + ttl_seconds)

class FootballDataClient:
    def __init__(self, api_key: str, min_interval: float):
        self.headers = {"X-Auth-Token": api_key, "User-Agent": f"FD-Predictor/{VERSION} (Streamlit)"}
        self.min_interval = min_interval
        self._last_call_ts = 0.0
        self._lock = threading.Lock()
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    def make_request(self, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        url = f"{BASE_URL}{path}"
        with self._lock:
            delta = time.time() - self._last_call_ts
            if delta < self.min_interval:
                time.sleep((self.min_interval - delta) + random.uniform(0, 0.5))
            self._last_call_ts = time.time()
        try:
            resp = self.session.get(url, headers=self.headers, params=params, timeout=25)
            if resp.status_code == 429:
                wait_sec = int(resp.headers.get("Retry-After", 60))
                log(f"Rate limit hit. Waiting {wait_sec}s...")
                time.sleep(wait_sec)
                return self.make_request(path, params)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            log(f"Request error for {url}: {e}")
            return None
