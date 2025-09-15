import os
import time
from datetime import datetime, timedelta, UTC
import pandas as pd

from football_predictor.client import FootballDataClient, AnalysisCache
from football_predictor.model import PredictionModel

# --- ضع مفتاحك الصحيح هنا ---
API_KEY = "df128dfc99124925a2dca05ea3a361ed"
# --------------------------------

# --- إعدادات ---
COMPETITIONS = {
    "Premier League (England)": 2021, "La Liga (Spain)": 2014,
    "Serie A (Italy)": 2019, "Bundesliga (Germany)": 2002, "Ligue 1 (France)": 2015,
}
START_DATE_STR = "2023-08-01"
END_DATE_STR = datetime.now(UTC).strftime("%Y-%m-%d")
OUTPUT_CSV_FILE = "historical_dataset.csv"

def get_team_form(team_id: int, all_matches: list, current_match_date: datetime, games_to_look_back: int = 5):
    team_matches = []
    for match in all_matches:
        match_date = datetime.strptime(match['utcDate'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        if match_date < current_match_date:
            if (match['homeTeam']['id'] == team_id) or (match['awayTeam']['id'] == team_id):
                team_matches.append(match)
    
    last_n_games = sorted(team_matches, key=lambda x: x['utcDate'], reverse=True)[:games_to_look_back]
    
    points, goals_scored, goals_conceded = 0, 0, 0
    for match in last_n_games:
        score = match['score']['fullTime']
        if match['homeTeam']['id'] == team_id:
            points += 3 if score['home'] > score['away'] else 1 if score['home'] == score['away'] else 0
            goals_scored += score['home']
            goals_conceded += score['away']
        else:
            points += 3 if score['away'] > score['home'] else 1 if score['away'] == score['home'] else 0
            goals_scored += score['away']
            goals_conceded += score['home']
            
    return {"points": points, "goals_scored": goals_scored, "goals_conceded": goals_conceded}

def main():
    if not API_KEY or API_KEY == "YOUR_CORRECT_API_KEY_HERE":
        raise ValueError("Please replace 'YOUR_CORRECT_API_KEY_HERE' with your actual API key inside the script.")

    cache = AnalysisCache()
    # --- FIXED: Slow down the requests to respect the rate limit (10 per minute) ---
    client = FootballDataClient(api_key=API_KEY, min_interval=6.5, cache=cache) # 6.5 seconds > 6 seconds (60s/10 req)
    model = PredictionModel(client=client, cache=cache)

    all_processed_matches = []
    for comp_name, comp_id in COMPETITIONS.items():
        print(f"\n--- Processing Competition: {comp_name} ---")
        start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d").date()
        end_date = datetime.strptime(END_DATE_STR, "%Y-%m-%d").date()
        
        # Note: The chunking logic is now inside model._get_historical_matches_in_chunks
        # And the client itself handles the delay between requests.
        matches = model._get_historical_matches_in_chunks(comp_id, start_date, end_date)
        
        if not matches:
            print(f"No matches found for {comp_name}. Skipping.")
            continue
            
        matches_sorted = sorted(matches, key=lambda x: x['utcDate'])
        
        elo_ratings = {}
        K, H_adv = 20, 50

        for i, match in enumerate(matches_sorted):
            home_team, away_team = match['homeTeam'], match['awayTeam']
            score = match['score']['fullTime']
            match_date_dt = datetime.strptime(match['utcDate'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
            
            if score['home'] is None or score['away'] is None: continue

            home_elo_before = elo_ratings.get(home_team['id'], 1500.0)
            away_elo_before = elo_ratings.get(away_team['id'], 1500.0)
            
            home_form = get_team_form(home_team['id'], matches_sorted[:i], match_date_dt)
            away_form = get_team_form(away_team['id'], matches_sorted[:i], match_date_dt)

            processed_row = {
                "match_id": match['id'], "date": match['utcDate'], "competition": comp_name,
                "home_team_id": home_team['id'], "home_team_name": home_team['name'],
                "away_team_id": away_team['id'], "away_team_name": away_team['name'],
                "home_team_elo": home_elo_before, "away_team_elo": away_elo_before,
                "elo_diff": home_elo_before - away_elo_before,
                "home_form_points": home_form['points'], "home_form_gs": home_form['goals_scored'], "home_form_gc": home_form['goals_conceded'],
                "away_form_points": away_form['points'], "away_form_gs": away_form['goals_scored'], "away_form_gc": away_form['goals_conceded'],
                "home_team_league_pos": 0, "away_team_league_pos": 0,
                "result_ft_home_goals": score['home'], "result_ft_away_goals": score['away'],
            }
            all_processed_matches.append(processed_row)
            
            Eh = 1.0 / (1.0 + 10 ** (-(home_elo_before + H_adv - away_elo_before) / 400.0))
            Sh = 1.0 if score['home'] > score['away'] else 0.5 if score['home'] == score['away'] else 0.0
            
            elo_ratings[home_team['id']] = home_elo_before + K * (Sh - Eh)
            elo_ratings[away_team['id']] = away_elo_before - K * (Sh - Eh)

        print(f"Processed {len(matches_sorted)} matches for {comp_name}.")
        
    if all_processed_matches:
        df = pd.DataFrame(all_processed_matches)
        df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"\n✅ Successfully built dataset with {len(df)} matches.")
        print(f"Saved to '{OUTPUT_CSV_FILE}'")
    else:
        print("\nNo data was processed.")

if __name__ == "__main__":
    main()
