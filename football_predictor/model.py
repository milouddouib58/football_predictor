import difflib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import football_predictor.settings as settings
from football_predictor.client import AnalysisCache, FootballDataClient
from football_predictor.utils import clamp, log, poisson_pmf


class PredictionModel:
    """
    يحتوي على منطق استخراج البيانات، بناء عوامل الفرق، وحساب الاحتمالات.
    """
    def __init__(self, client: FootballDataClient, cache: AnalysisCache):
        self.client = client
        self.cache = cache

    # --- Data Fetching ---
    def _get_competition_teams(self, comp_id: int) -> List[Dict]:
        log(f"Fetching teams for competition {comp_id}")
        data = self.client.make_request(f"/competitions/{comp_id}/teams", ttl=settings.TEAMS_TTL)
        return data.get("teams", []) if data else []

    def find_team_id_by_name(self, name: str, comp_id: int) -> Optional[int]:
        teams = self._get_competition_teams(comp_id)
        if not teams:
            return None

        best_score, best_id = 0.0, None
        name_lower = name.lower()
        for team in teams:
            for team_name_variant in [team.get("name"), team.get("shortName"), team.get("tla")]:
                if not team_name_variant:
                    continue
                score = difflib.SequenceMatcher(None, name_lower, team_name_variant.lower()).ratio()
                if score > best_score:
                    best_score, best_id = score, team.get("id")

        if best_score > 0.65:
            return best_id
        return None

    def _get_competition_matches(self, comp_id: int, date_from: str, date_to: str) -> List[Dict]:
        log(f"Fetching matches for competition {comp_id}")
        params = {"competitions": comp_id, "status": "FINISHED", "dateFrom": date_from, "dateTo": date_to}
        data = self.client.make_request("/matches", params=params, ttl=settings.MATCHES_TTL)
        return data.get("matches", []) if data else []

    def _get_competition_season_dates(self, comp_id: int) -> Tuple[str, str, str]:
        info = self.client.make_request(f"/competitions/{comp_id}", ttl=settings.COMP_INFO_TTL) or {}
        season = info.get("currentSeason", {}) or {}
        start = season.get("startDate")
        end = season.get("endDate")
        # قيم افتراضية إذا تعذر جلب التواريخ
        start = start or datetime.utcnow().strftime("%Y-%m-%d")
        end = end or datetime.utcnow().strftime("%Y-%m-%d")
        comp_name = info.get("name", "Unknown Competition")
        return start, end, comp_name

    def _get_team_details(self, team_id: int) -> Dict:
        return self.client.make_request(f"/teams/{team_id}", ttl=settings.TEAM_DETAILS_TTL) or {}

    # --- League Averages ---
    def _calculate_league_averages(self, matches: List[Dict]) -> Dict:
        hg_sum, ag_sum, cnt = 0, 0, 0
        for m in matches:
            score = (m.get("score") or {}).get("fullTime", {})
            hg, ag = score.get("home"), score.get("away")
            if hg is not None and ag is not None:
                hg_sum += int(hg)
                ag_sum += int(ag)
                cnt += 1
        if cnt < 20:
            # افتراضات معقولة في حال قلة البيانات
            return {"avg_home_goals": 1.45, "avg_away_goals": 1.15, "matches_count": cnt}
        return {"avg_home_goals": hg_sum / cnt, "avg_away_goals": ag_sum / cnt, "matches_count": cnt}

    # --- Team Factors (Attack/Defense) ---
    def _build_iterative_team_factors(self, matches: List[Dict], league_avgs: Dict, prior_games: int):
        team_ids = set()
        for m in matches:
            if m.get("homeTeam") and m["homeTeam"].get("id"):
                team_ids.add(m["homeTeam"]["id"])
            if m.get("awayTeam") and m["awayTeam"].get("id"):
                team_ids.add(m["awayTeam"]["id"])

        A = {tid: 1.0 for tid in team_ids}  # Attack
        D = {tid: 1.0 for tid in team_ids}  # Defense

        # تكرارات بسيطة للوصول للتقارب
        for _ in range(10):
            new_A = A.copy()
            new_D = D.copy()

            # تحديث الهجوم
            for team_id in team_ids:
                goals_scored, expected_scored = prior_games, prior_games
                for match in matches:
                    h = (match.get("homeTeam") or {}).get("id")
                    a = (match.get("awayTeam") or {}).get("id")
                    score = (match.get("score") or {}).get("fullTime", {})
                    hg, ag = score.get("home"), score.get("away")
                    if not all([h, a, hg is not None, ag is not None]):
                        continue

                    if team_id == h:
                        goals_scored += int(hg)
                        expected_scored += league_avgs["avg_home_goals"] * new_D.get(a, 1.0)
                    elif team_id == a:
                        goals_scored += int(ag)
                        expected_scored += league_avgs["avg_away_goals"] * new_D.get(h, 1.0)

                new_A[team_id] = goals_scored / expected_scored if expected_scored > 0 else 1.0

            A = {tid: clamp(val, settings.AD_CLAMP_MIN, settings.AD_CLAMP_MAX) for tid, val in new_A.items()}

            # تحديث الدفاع
            for team_id in team_ids:
                goals_conceded, expected_conceded = prior_games, prior_games
                for match in matches:
                    h = (match.get("homeTeam") or {}).get("id")
                    a = (match.get("awayTeam") or {}).get("id")
                    score = (match.get("score") or {}).get("fullTime", {})
                    hg, ag = score.get("home"), score.get("away")
                    if not all([h, a, hg is not None, ag is not None]):
                        continue

                    if team_id == h:
                        goals_conceded += int(ag)
                        expected_conceded += league_avgs["avg_away_goals"] * A.get(a, 1.0)
                    elif team_id == a:
                        goals_conceded += int(hg)
                        expected_conceded += league_avgs["avg_home_goals"] * A.get(h, 1.0)

                new_D[team_id] = goals_conceded / expected_conceded if expected_conceded > 0 else 1.0

            D = {tid: clamp(val, settings.AD_CLAMP_MIN, settings.AD_CLAMP_MAX) for tid, val in new_D.items()}

        return A, D

    # --- ELO Table ---
    def _build_elo_table(self, matches: List[Dict]) -> Dict[int, float]:
        ratings: Dict[int, float] = {}
        K, H_adv = 20, 50
        # ترتيب زمني
        matches_sorted = sorted(matches, key=lambda x: x.get("utcDate", ""))
        for m in matches_sorted:
            h = (m.get("homeTeam") or {}).get("id")
            a = (m.get("awayTeam") or {}).get("id")
            score = (m.get("score") or {}).get("fullTime", {})
            hg, ag = score.get("home"), score.get("away")
            if not all([h, a, hg is not None, ag is not None]):
                continue

            Rh, Ra = ratings.get(h, 1500.0), ratings.get(a, 1500.0)
            Eh = 1.0 / (1.0 + 10 ** (-(Rh + H_adv - Ra) / 400.0))
            Sh = 1.0 if int(hg) > int(ag) else 0.5 if int(hg) == int(ag) else 0.0

            ratings[h] = Rh + K * (Sh - Eh)
            ratings[a] = Ra - K * (Sh - Eh)

        return ratings

    # --- Poisson Helpers ---
    def _poisson_matrix(self, lam_home: float, lam_away: float, max_goals: int):
        size = max_goals + 1
        M = [[poisson_pmf(i, lam_home) * poisson_pmf(j, lam_away) for j in range(size)] for i in range(size)]
        return M

    def _matrix_to_outcomes(self, M):
        size = len(M)
        p_home = sum(M[i][j] for i in range(size) for j in range(i))
        p_draw = sum(M[i][i] for i in range(size))
        p_away = sum(M[i][j] for i in range(size) for j in range(i + 1, size))
        total_p = p_home + p_draw + p_away
        if total_p == 0:
            return 0.33, 0.34, 0.33
        return p_home / total_p, p_draw / total_p, p_away / total_p

    # --- Main Prediction ---
    def predict(self, comp_id: int, home_team_id: int, away_team_id: int, advanced_settings: Dict) -> Dict:
        log(f"Starting prediction for competition {comp_id}...")

        elo_scale = float(advanced_settings.get("elo_scale", settings.ELO_SCALE))
        prior_games = int(advanced_settings.get("prior_games", settings.PRIOR_GAMES))

        # كاش نتيجة التوقع
        cache_key = f"pred:{settings.VERSION}:{comp_id}:{home_team_id}:{away_team_id}:{elo_scale:.3f}:{prior_games}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        start_date, _end_date, comp_name = self._get_competition_season_dates(comp_id)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        matches = self._get_competition_matches(comp_id, start_date, today)

        if len(matches) < 20:
            raise RuntimeError(f"لا توجد بيانات مباريات كافية للتحليل (وجد {len(matches)} مباراة فقط).")

        league_avgs = self._calculate_league_averages(matches)
        A, D = self._build_iterative_team_factors(matches, league_avgs, prior_games)
        elo_ratings = self._build_elo_table(matches)

        # عوامل الهجوم والدفاع
        Ah, Dh = A.get(home_team_id, 1.0), D.get(home_team_id, 1.0)
        Aa, Da = A.get(away_team_id, 1.0), D.get(away_team_id, 1.0)  # تم إصلاح العكس

        lam_home_base = league_avgs["avg_home_goals"] * Ah * Da
        lam_away_base = league_avgs["avg_away_goals"] * Aa * Dh

        # عامل ELO
        Rh, Ra = float(elo_ratings.get(home_team_id, 1500.0)), float(elo_ratings.get(away_team_id, 1500.0))
        elo_adv = (Rh + 50.0 - Ra) / 400.0
        elo_factor = 1.0 + elo_scale * (1.0 / (1.0 + 10 ** (-elo_adv)) - 0.5)

        lam_home_final = lam_home_base * clamp(elo_factor, settings.ELO_LAM_MIN, settings.ELO_LAM_MAX)
        lam_away_final = lam_away_base * clamp(2.0 - elo_factor, settings.ELO_LAM_MIN, settings.ELO_LAM_MAX)
        lam_home_final = clamp(lam_home_final, settings.LAM_CLAMP_MIN, settings.LAM_CLAMP_MAX)
        lam_away_final = clamp(lam_away_final, settings.LAM_CLAMP_MIN, settings.LAM_CLAMP_MAX)

        # مصفوفة بواسون والاحتمالات
        matrix = self._poisson_matrix(lam_home_final, lam_away_final, settings.MAX_GOALS_GRID)
        p_home, p_draw, p_away = self._matrix_to_outcomes(matrix)

        # احتمالات إضافية
        size = len(matrix)
        total_mass = sum(matrix[i][j] for i in range(size) for j in range(size)) or 1.0
        prob_over_2_5 = sum(matrix[i][j] for i in range(size) for j in range(size) if i + j >= 3) / total_mass
        prob_btts_yes = sum(matrix[i][j] for i in range(size) for j in range(size) if i > 0 and j > 0) / total_mass

        scorelines = [(f"{i}-{j}", matrix[i][j] / total_mass) for i in range(size) for j in range(size)]
        scorelines.sort(key=lambda x: x[1], reverse=True)
        top_scorelines = [{"score": s, "p": round(p * 100, 1)} for s, p in scorelines[:5]]

        # تفاصيل الفرق
        home_details = self._get_team_details(home_team_id)
        away_details = self._get_team_details(away_team_id)

        def _pick_name(team_detail: Dict) -> Optional[str]:
            return team_detail.get("shortName") or team_detail.get("name") or team_detail.get("tla")

        result = {
            "meta": {
                "version": settings.VERSION,
                "competition": f"{comp_name} ({comp_id})",
                "generated_at": datetime.utcnow().isoformat(),
            },
            "teams": {
                "home": {"id": home_team_id, "name": _pick_name(home_details)},
                "away": {"id": away_team_id, "name": _pick_name(away_details)},
            },
            "lambdas": {"home_final": round(lam_home_final, 3), "away_final": round(lam_away_final, 3)},
            "elo": {"home_rating": round(Rh), "away_rating": round(Ra)},
            "probabilities": {
                "1x2": {
                    "home": round(p_home * 100, 1),
                    "draw": round(p_draw * 100, 1),
                    "away": round(p_away * 100, 1),
                },
                "over_under": {
                    "over_2_5": round(prob_over_2_5 * 100, 1),
                    "under_2_5": round((1 - prob_over_2_5) * 100, 1),
                },
                "btts": {
                    "yes": round(prob_btts_yes * 100, 1),
                    "no": round((1 - prob_btts_yes) * 100, 1),
                },
                "top_scorelines": top_scorelines,
            },
        }

        # تخزين النتيجة في الكاش
        self.cache.set(cache_key, result, ttl_seconds=settings.PREDICTION_TTL)
        return result
