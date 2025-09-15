# In file: football_predictor/model.py

# ... (imports and other class methods remain the same) ...

    # --- الدالة الرئيسية للتوقع (النسخة المحدثة) ---
    def predict(self, comp_id: int, home_team_id: int, away_team_id: int, advanced_settings: Dict) -> Dict:
        log(f"Starting prediction for competition {comp_id}...")

        # استخدام الإعدادات من الواجهة أو القيم الافتراضية
        elo_scale = advanced_settings.get('elo_scale', settings.ELO_SCALE)
        prior_games = advanced_settings.get('prior_games', settings.PRIOR_GAMES)
        
        start_date, end_date, comp_name = self._get_competition_season_dates(comp_id)
        
        matches = self._get_competition_matches(comp_id, start_date, datetime.now().strftime("%Y-%m-%d"))
        if not matches:
            raise RuntimeError("لا توجد بيانات مباريات كافية للتحليل.")

        league_avgs = self._calculate_league_averages(matches)
        A, D = self._build_iterative_team_factors(matches, league_avgs) # Note: this needs full implementation
        elo_ratings = self._build_elo_table(matches)

        Ah, Dh = A.get(home_team_id, 1.0), D.get(home_team_id, 1.0)
        Aa, Da = A.get(away_team_id, 1.0), D.get(away_team_id, 1.0)
        
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
