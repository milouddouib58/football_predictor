VERSION = "12.0-Improved"

# إعدادات الاتصال
BASE_URL = "https://api.football-data.org/v4"
MIN_INTERVAL_SEC = 6.5  # احترام قيود المعدل

# إعدادات TTL للكاش (بالثواني)
TEAMS_TTL = 3600 * 6
MATCHES_TTL = 3600 * 2
COMP_INFO_TTL = 3600 * 24
TEAM_DETAILS_TTL = 3600 * 24
PREDICTION_TTL = 3600  # نتيجة التوقع نفسها

# إعدادات النموذج
PRIOR_GAMES = 12
LAM_CLAMP_MIN = 0.1
LAM_CLAMP_MAX = 3.0
AD_CLAMP_MIN = 0.7
AD_CLAMP_MAX = 1.3
ELO_LAM_MIN = 0.88
ELO_LAM_MAX = 1.12
ELO_SCALE = 0.28
MAX_GOALS_GRID = 8

# إعدادات متقدمة (قابلة للتطوير لاحقًا)
H2H_LOOKBACK_DAYS = 365
HALF_LIFE_DAYS = 270
DC_RHO_MAX = 0.3

# أولويات المسابقات (للاستخدام المستقبلي)
COMPETITION_PRIORITY = ["CL", "PD", "PL", "SA", "BL1", "FL1"]
