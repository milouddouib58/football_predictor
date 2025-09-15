# file: football_predictor/model.py
import difflib
import math
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import streamlit as st

from football_predictor.client import FootballDataClient, AnalysisCache
import football_predictor.settings as settings
from football_predictor.utils import log, parse_date_safe, poisson_pmf, clamp, ewma_weight

class PredictionModel:
    def __init__(self, client: FootballDataClient, cache: AnalysisCache):
        self.client = client
        self.cache = cache

    # --- دوال جلب البيانات ---
    @st.cache_data(ttl=3600 * 6) # تخزين مؤقت لقوائم الفرق لمدة 6 ساعات
    def _get_competition_teams(_self, comp_id: int) -> List[Dict]:
        data = _self.client.make_request(f"/competitions/{comp_id}/teams")
        return data.get("teams", []) if data else []

    def find_team_id_by_name(self, name: str, comp_id: int) -> Optional[int]:
        teams = self._get_competition_teams(comp_id)
        if not teams: return None
        
        best_score, best_id = 0.0, None
        name_lower = name.lower()
        for team in teams:
            for team_name_variant in [team.get('name'), team.get('shortName'), team.get('tla')]:
                if not team_name_variant: continue
                score = difflib.SequenceMatcher(None, name_lower, team_name_variant.lower()).ratio()
                if score > best_score:
                    best_score, best_id = score, team.get('id')
        
        if best_score > 0.65:
            log(f"Found team '{name}' as ID {best_id} with score {best_score:.2f}")
            return best_id
        return None

    @st.cache_data(ttl=3600 * 2) # تخزين مؤقت لبيانات المباريات لمدة ساعتين
    def _get_competition_matches(_self, comp_id: int, date_from: str, date_to: str) -> List[Dict]:
        params = {"competitions": comp_id, "status": "FINISHED", "dateFrom": date_from, "dateTo": date_to}
        data = _self.client.make_request("/matches", params=params)
        return data.get("matches", []) if data else []

    @st.cache_data(ttl=3600 * 24)
    def _get_competition_season_dates(_self, comp_id: int) -> Tuple[str, str, str]:
        info = _self.client.make_request(f"/competitions/{comp_id}") or {}
        season = info.get("currentSeason", {})
        start = season.get("startDate", (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"))
        end = season.get("endDate", (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))
        return start, end, info.get("name", "Unknown Competition")

    # --- دوال التحليل المتقدم ---
    def _calculate_league_averages(self, matches: List[Dict]) -> Dict:
        hg_sum, ag_sum, cnt = 0, 0, 0
        for m in matches:
            score = m.get("score", {}).get("fullTime", {})
            hg, ag = score.get("home"), score.get("away")
            if hg is not None and ag is not None:
                hg_sum += hg
                ag_sum += ag
                cnt += 1
        if cnt < 20: return {"avg_home_goals": 1.45, "avg_away_goals": 1.15, "matches_count": cnt}
        return {"avg_home_goals": hg_sum / cnt, "avg_away_goals": ag_sum / cnt, "matches_count": cnt}

    def _build_iterative_team_factors(self, matches: List[Dict], league_avgs: Dict) -> Tuple[Dict, Dict]:
        # ... (المنطق الكامل لحساب قوة الهجوم والدفاع A, D) ...
        # (هذا تبسيط، ولكن في المشروع الحقيقي ستضع الحلقة التكرارية هنا)
        team_ids = set(m['homeTeam']['id'] for m in matches) | set(m['awayTeam']['id'] for m in matches)
        A = {tid: 1.0 for tid in team_ids}
        D = {tid: 1.0 for tid in team_ids}
        return A, D
        
    def _build_elo_table(self, matches: List[Dict]) -> Dict:
        ratings = {}
        K, H_adv = 20, 50
        for m in sorted(matches, key=lambda x: x.get('utcDate', '')):
            h, a = (m.get("homeTeam") or {}).get("id"), (m.get("awayTeam") or {}).get("id")
            score = (m.get("score") or {}).get("fullTime", {})
            hg, ag = score.get("home"), score.get("away")
            if not all([h, a, hg is not None, ag is not None]): continue
            
            Rh, Ra = ratings.get(h, 1500), ratings.get(a, 1500)
            Eh = 1.0 / (1.0 + 10 ** (-(Rh + H_adv - Ra) / 400.0))
            Sh = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            
            ratings[h] = Rh + K * (Sh - Eh)
            ratings[a] = Ra - K * (Sh - Eh)
        return ratings

    def _poisson_matrix(self, lh, la, max_goals):
        M = [[poisson_pmf(i, lh) * poisson_pmf(j, la) for j in range(max_goals + 1)] for i in range(max_goals + 1)]
        return M

    def _matrix_to_outcomes(self, M):
        p_home = sum(M[i][j] for i in range(len(M)) for j in range(i))
        p_draw = sum(M[i][i] for i in range(len(M)))
        p_away = sum(M[i][j] for i in range(len(M)) for j in range(i + 1, len(M)))
        total_p = p_home + p_draw + p_away
        if total_p == 0: return 0.33, 0.34, 0.33
        return p_home/total_p, p_draw/total_p, p_away/total_p

    # --- الدالة الرئيسية للتوقع ---
    def predict(self, team1_name: str, team2_name: str, team1_is_home: bool) -> Dict:
        log(f"Starting full logic prediction for: {team1_name} vs {team2_name}")

        comp_id = 2021 # Premier League
        start_date, end_date, comp_name = self._get_competition_season_dates(comp_id)
        
        t1_id = self.find_team_id_by_name(team1_name, comp_id)
        t2_id = self.find_team_id_by_name(team2_name, comp_id)
        if not t1_id or not t2_id:
            raise ValueError(f"لم يتم العثور على أحد الفريقين في الدوري الإنجليزي الممتاز.")

        home_id, away_id = (t1_id, t2_id) if team1_is_home else (t2_id, t1_id)
        
        matches = self._get_competition_matches(comp_id, start_date, datetime.now().strftime("%Y-%m-%d"))
        if not matches:
            raise RuntimeError("لا توجد بيانات مباريات كافية للتحليل.")

        league_avgs = self._calculate_league_averages(matches)
        A, D = self._build_iterative_team_factors(matches, league_avgs)
        elo_ratings = self._build_elo_table(matches)

        Ah, Dh = A.get(home_id, 1.0), D.get(home_id, 1.0)
        Aa, Da = A.get(away_id, 1.0), D.get(away_id, 1.0)
        
        lam_home_base = league_avgs['avg_home_goals'] * Ah * Da
        lam_away_base = league_avgs['avg_away_goals'] * Aa * Dh

        Rh, Ra = elo_ratings.get(home_id, 1500), elo_ratings.get(away_id, 1500)
        elo_adv = (Rh + 50 - Ra) / 400.0
        elo_factor = 1.0 + settings.ELO_SCALE * (1.0 / (1.0 + 10**(-elo_adv)) - 0.5)

        lam_home_final = lam_home_base * clamp(elo_factor, settings.ELO_LAM_MIN, settings.ELO_LAM_MAX)
        lam_away_final = lam_away_base * clamp(2.0 - elo_factor, settings.ELO_LAM_MIN, settings.ELO_LAM_MAX)
        lam_home_final = clamp(lam_home_final, settings.LAM_CLAMP_MIN, settings.LAM_CLAMP_MAX)
        lam_away_final = clamp(lam_away_final, settings.LAM_CLAMP_MIN, settings.LAM_CLAMP_MAX)
        
        matrix = self._poisson_matrix(lam_home_final, lam_away_final, settings.MAX_GOALS_GRID)
        p_home, p_draw, p_away = self._matrix_to_outcomes(matrix)
        
        home_details = self.client.make_request(f"/teams/{home_id}") or {}
        away_details = self.client.make_request(f"/teams/{away_id}") or {}

        return {
            "meta": {"version": settings.VERSION, "competition": f"{comp_name} ({comp_id})"},
            "teams": {
                "home": {"name": home_details.get("shortName", team1_name)},
                "away": {"name": away_details.get("shortName", team2_name)},
            },
            "lambdas": {"home_final": round(lam_home_final, 3), "away_final": round(lam_away_final, 3)},
            "elo": {"home_rating": round(Rh), "away_rating": round(Ra)},
            "probabilities": {
                "1x2": {
                    "home": round(p_home * 100, 1), "draw": round(p_draw * 100, 1), "away": round(p_away * 100, 1)
                }
            },
        }
