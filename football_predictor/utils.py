# file: football_predictor/utils.py
import math
from datetime import datetime, timedelta
from typing import Optional

def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def parse_date_safe(s: Optional[str]):
    if not s: return None
    try:
        return datetime.strptime(s.split("T")[0], "%Y-m-d").date()
    except (ValueError, TypeError):
        return None

def ewma_weight(match_date_iso: str, ref_date_iso: str, half_life_days: int) -> float:
    md = parse_date_safe(match_date_iso)
    rd = parse_date_safe(ref_date_iso)
    if not md or not rd or half_life_days <= 0: return 1.0
    age = max(0, (rd - md).days)
    return 0.5 ** (age / half_life_days)

def poisson_pmf(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    try:
        return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))
    except (ValueError, OverflowError):
        return 0.0
