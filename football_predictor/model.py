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

    # --- Data Fetching Methods ---
    @st.cache_data(ttl=3600 * 6)
    def _get_competition_teams(_self, comp_id: int) -> List[Dict]:
        log(f"Fetching teams for competition {comp_id}")
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
            return best_id
        return None

    @st.cache_data(ttl=3600 * 2)
    def _get_competition_matches(_self, comp_id: int, date_from: str, date_to: str) -> List[Dict]:
        log(f"Fetching matches for competition {comp_id}")
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

    # --- Advanced Analytical Methods ---
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

    def _build_iterative_team_factors(self, matches: List[Dict], league_avgs: Dict, prior_games: int) -> Tuple[Dict, Dict]:
        team_ids = set()
        for m in matches:
            if m.get('homeTeam') and m['homeTeam'].get('id'): team_ids.add(m['homeTeam']['id'])
            if m.get('awayTeam') and m['awayTeam'].get('id'): team_ids.add(m['awayTeam']['id'])
        
        A = {tid: 1.0 for tid in team_ids}
        D = {tid: 1.0 for tid in team_ids}

        for i in range(10): # Iterate 10 times for convergence
            new_A = A.copy()
            new_D = D.copy()

            for team_id in team_ids:
                # Update Attack Strength (A)
                goals_scored, expected_goals_scored = prior_games, prior_games
                for match in matches:
                    h, a = (match.get('homeTeam') or {}).get('id'), (match.get('awayTeam') or {}).get('id')
                    score = (match.get('score') or {}).get('fullTime', {})
                    hg, ag = score.get('home'), score.get('away')
                    if not all([h, a, hg is not None, ag is not None]): continue
                    
                    if team_id == h:
                        goals_scored += hg
                        expected_goals_scored += league_avgs['avg_home_goals'] * new_D.get(a, 1.0)
                    elif team_id == a:
                        goals_scored += ag
                        expected_goals_scored += league_avgs['avg_away_goals'] * new_D.get(h, 1.0)
                new_A[team_id] = goals_scored / expected_goals_scored if expected_goals_scored > 0 else 1.0

            A = {tid: clamp(val, settings.AD_CLAMP_MIN, settings.AD_CLAMP_MAX) for tid, val in new_A.items()}

            for team_id in team_ids:
                # Update Defense Strength (D)
                goals_conceded, expected_goals_conceded = prior_games, prior_games
                for match in matches:
                    h, a = (match.get('homeTeam') or {}).get('id'), (match.get('awayTeam') or {}).get('id')
                    score = (match.get('score') or {}).get('fullTime', {})
                    hg, ag = score.get('home'), score.get('away')
                    if not all([h, a, hg is not None, ag is not None]): continue

                    if team_id == h:
                        goals_conceded += ag
                        expected_goals_conceded += league_avgs['avg_away_goals'] * A.get(a, 1.0)
                    elif team_id == a:
                        goals_conceded += hg
                        expected_goals_conceded += league_avgs['avg_home_goals'] * A.get(h, 1.0)
                new_D[team_id] = goals_conceded / expected_goals_conceded if expected_goals_conceded > 0 else 1.0
            
            D = {tid: clamp(val, settings.AD_CLAMP_MIN, settings.AD_CLAMP_MAX) for tid, val in new_D.items()}
        
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

    # --- Main Prediction Function ---
    def predict(self, comp_id: int, home_team_id: int, away_team_id: int, advanced_settings: Dict) -> Dict:
        log(f"Starting full logic prediction for competition {comp_id}...")

        elo_scale = advanced_settings.get('elo_scale', settings.ELO_SCALE)
        prior_games = advanced_settings.get('prior_games', settings.PRIOR_GAMES)
        
        start_date, end_date, comp_name = self._get_competition_season_dates(comp_id)
        
        matches = self._get_competition_matches(comp_id, start_date, datetime.now().strftime("%Y-%m-%d"))
        if len(matches) < 20:
            raise RuntimeError(f"لا توجد بيانات مباريات كافية للتحليل (وجد {len(matches)} مباراة فقط).")

        league_avgs = self._calculate_league_averages(matches)
        A, D = self._build_iterative_team_factors(matches, league_avgs, prior_games)
        elo_ratings = self._build_elo_table(matches)

        Ah, Dh = A.get(home_team_id, 1.0), D.get(home_team_id, 1.0)
        Aa, Da = D.get(away_team_id, 1.0), A.get(away_team_id, 1.0)
        
        lam_home_base = league_avgs['avg_home_goals'] * Ah * Da
        lam_away_base = league_avgs['avg_away_goals'] * Aa * Dh

        Rh, Ra = elo_ratings.get(home_team_id, 1500), elo_ratings.get(away_team_id, 1500)
        elo_adv = (Rh + 50 - Ra) / 400.0
        elo_factor = 1.0 + elo_scale * (1.0 / (1.0 + 10**(-elo_adv)) - 0.5)

        lam_home_final = lam_home_base * clamp(elo_factor, settings.ELO_LAM_MIN, settings.ELO_LAM_MAX)
        lam_away_final = lam_away_base * clamp(2.0 - elo_factor, settings.ELO_LAM_MIN, settings.ELO_LAM_MAX)
        lam_home_final = clamp(lam_home_final, settings.LAM_CLAMP_MIN, settings.LAM_CLAMP_MAX)
        lam_away_final = clamp(lam_away_final, settings.LAM_CLAMP_MIN, settings.LAM_CLAMP_MAX)
        
        matrix = self._poisson_matrix(lam_home_final, lam_away_final, settings.MAX_GOALS_GRID)
        p_home, p_draw, p_away = self._matrix_to_outcomes(matrix)
        
        home_details = self.client.make_request(f"/teams/{home_team_id}") or {}
        away_details = self.client.make_request(f"/teams/{away_team_id}") or {}

        return {
            "meta": {"version": settings.VERSION, "competition": f"{comp_name} ({comp_id})"},
            "teams": {
                "home": {"name": home_details.get("shortName")},
                "away": {"name": away_details.get("shortName")},
            },
            "lambdas": {"home_final": round(lam_home_final, 3), "away_final": round(lam_away_final, 3)},
            "elo": {"home_rating": round(Rh), "away_rating": round(Ra)},
            "probabilities": {
                "1x2": {
                    "home": round(p_home * 100, 1), "draw": round(p_draw * 100, 1), "away": round(p_away * 100, 1)
                }
            },
        }
