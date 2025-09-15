# file: football_predictor/settings.py
VERSION = "11.0-Final-Full-Logic"

# --- إعدادات الاتصال ---
BASE_URL = "https://api.football-data.org/v4"
MIN_INTERVAL_SEC = 6.5

# --- إعدادات النموذج ---
H2H_LOOKBACK_DAYS = 365
PRIOR_GAMES = 12
HALF_LIFE_DAYS = 270
DC_RHO_MAX = 0.3
LAM_CLAMP_MIN = 0.1
LAM_CLAMP_MAX = 3.0
AD_CLAMP_MIN = 0.7
AD_CLAMP_MAX = 1.3
ELO_LAM_MIN = 0.88
ELO_LAM_MAX = 1.12
ELO_SCALE = 0.28
MAX_GOALS_GRID = 8

# --- أولويات المسابقات ---
COMPETITION_PRIORITY = ["CL", "PD", "PL", "SA", "BL1", "FL1"]
