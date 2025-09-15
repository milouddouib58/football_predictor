import random
import threading
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .settings import BASE_URL, VERSION
from .utils import log


class AnalysisCache:
    """
    كاش خفيف مع TTL وآمن على الخيوط.
    """
    def __init__(self) -> None:
        self.cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self.cache.get(key)
            if not item:
                return None
            value, expires_at = item
            if time.time() > expires_at:
                self.cache.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        with self._lock:
            self.cache[key] = (value, time.time() + ttl_seconds)

    def clear(self) -> None:
        with self._lock:
            self.cache.clear()


class FootballDataClient:
    """
    عميل HTTP مع:
    - كاش داخلي قابل للتهيئة.
    - احترام حدود المعدّل.
    - إعادة المحاولة عند أخطاء الشبكة المؤقتة.
    """
    def __init__(self, api_key: str, min_interval: float, cache: Optional[AnalysisCache] = None, default_ttl: int = 3600):
        self.headers = {
            "X-Auth-Token": api_key,
            "User-Agent": f"FD-Predictor/{VERSION} (Streamlit)",
        }
        self.min_interval = min_interval
        self._last_call_ts = 0.0
        self._lock = threading.Lock()
        self.session = self._create_session()
        self.cache = cache
        self.default_ttl = default_ttl

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504, 429],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))
        return session

    def _cache_key(self, path: str, params: Optional[Dict]) -> str:
        if not params:
            return path
        return f"{path}?{urlencode(sorted(params.items()))}"

    def make_request(self, path: str, params: Optional[Dict] = None, ttl: Optional[int] = None) -> Dict:
        url = f"{BASE_URL}{path}"
        cache_key = self._cache_key(path, params)

        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        with self._lock:
            delta = time.time() - self._last_call_ts
            if delta < self.min_interval:
                time.sleep((self.min_interval - delta) + random.uniform(0, 0.5))
            self._last_call_ts = time.time()

        try:
            resp = self.session.get(url, headers=self.headers, params=params, timeout=25)
            
            # سيقوم هذا السطر بإطلاق استثناء (exception) إذا كان الرد خطأ (مثل 403, 404, 500)
            resp.raise_for_status()
            
            data = resp.json()
            if self.cache:
                self.cache.set(cache_key, data, ttl_seconds=(ttl or self.default_ttl))
            return data

        except requests.exceptions.HTTPError as e:
            # هنا نلتقط أخطاء الـ API مثل (خطأ في الصلاحية، الصفحة غير موجودة)
            error_message = f"API Error: {e.response.status_code} - {e.response.reason}."
            try:
                # محاولة قراءة رسالة الخطأ التفصيلية من الـ API
                api_error_details = e.response.json().get("message")
                if api_error_details:
                    error_message += f" Details: {api_error_details}"
            except requests.exceptions.JSONDecodeError:
                # في حال لم يرسل الـ API رسالة JSON
                error_message += f" Details: {e.response.text}"
            
            log(error_message)
            # نطلق استثناء جديد بالرسالة الواضحة ليظهر في الواجهة
            raise ConnectionError(error_message) from e

        except requests.exceptions.RequestException as e:
            # هنا نلتقط أخطاء الشبكة (مثل انقطاع الإنترنت، فشل DNS)
            log(f"Network request error for {url}: {e}")
            raise ConnectionError(f"Network error while connecting to the API: {e}") from e

    def clear_cache(self) -> None:
        if self.cache:
            self.cache.clear()
