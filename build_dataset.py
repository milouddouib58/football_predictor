import os
import time
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv

# استيراد الفئات التي بنيناها بالفعل لإعادة استخدامها
from football_predictor.client import FootballDataClient, AnalysisCache
from football_predictor.model import PredictionModel

# تحميل مفتاح API من ملف .env
load_dotenv()

# --- إعدادات ---
# الدوريات التي نريد بناء بيانات لها
COMPETITIONS = {
    "Premier League (England)": 2021,
    "La Liga (Spain)": 2014,
    "Serie A (Italy)": 2019,
    "Bundesliga (Germany)": 2002,
    "Ligue 1 (France)": 2015,
}
# الفترة الزمنية للبيانات التاريخية (مثلاً، من بداية موسم 2023-2024)
START_DATE_STR = "2023-08-01"
END_DATE_STR = datetime.utcnow().strftime("%Y-%m-%d")
OUTPUT_CSV_FILE = "historical_dataset.csv"


def get_team_form(team_id: int, all_matches: list, current_match_date: datetime, games_to_look_back: int = 5):
    """
    يحسب نقاط وفورمة الفريق في آخر N مباريات قبل تاريخ مباراة معينة.
    """
    team_matches = []
    # فلترة المباريات السابقة للفريق فقط
    for match in all_matches:
        match_date = datetime.strptime(match['utcDate'], "%Y-%m-%dT%H:%M:%SZ")
        if match_date < current_match_date:
            if (match['homeTeam']['id'] == team_id) or (match['awayTeam']['id'] == team_id):
                team_matches.append(match)
    
    # أخذ آخر N مباريات فقط
    last_n_games = sorted(team_matches, key=lambda x: x['utcDate'], reverse=True)[:games_to_look_back]
    
    points = 0
    goals_scored = 0
    goals_conceded = 0

    for match in last_n_games:
        score = match['score']['fullTime']
        if match['homeTeam']['id'] == team_id:
            points += 3 if score['home'] > score['away'] else 1 if score['home'] == score['away'] else 0
            goals_scored += score['home']
            goals_conceded += score['away']
        else: # الفريق هو الضيف
            points += 3 if score['away'] > score['home'] else 1 if score['away'] == score['home'] else 0
            goals_scored += score['away']
            goals_conceded += score['home']
            
    return {
        "points": points,
        "goals_scored": goals_scored,
        "goals_conceded": goals_conceded
    }


def main():
    """
    الوظيفة الرئيسية لتشغيل عملية بناء مجموعة البيانات.
    """
    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    if not api_key:
        raise ValueError("API key not found in .env file.")

    cache = AnalysisCache()
    client = FootballDataClient(api_key=api_key, min_interval=1.0, cache=cache)
    model = PredictionModel(client=client, cache=cache)

    all_processed_matches = []

    for comp_name, comp_id in COMPETITIONS.items():
        print(f"\n--- Processing Competition: {comp_name} ---")

        # 1. جلب كل المباريات التاريخية باستخدام الدالة التي تتعامل مع الأجزاء
        start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d").date()
        end_date = datetime.strptime(END_DATE_STR, "%Y-%m-%d").date()
        
        matches = model._get_historical_matches_in_chunks(comp_id, start_date, end_date)
        
        if not matches:
            print(f"No matches found for {comp_name}. Skipping.")
            continue
            
        # 2. ترتيب المباريات زمنيًا، وهذا مهم جدًا لحساب الفورمة و ELO بشكل صحيح
        matches_sorted = sorted(matches, key=lambda x: x['utcDate'])
        
        # 3. حساب الميزات لكل مباراة بالترتيب
        elo_ratings = {} # قاموس لتخزين تقييمات ELO وتحديثها باستمرار
        K, H_adv = 20, 50 # ثوابت ELO

        for i, match in enumerate(matches_sorted):
            home_team = match['homeTeam']
            away_team = match['awayTeam']
            score = match['score']['fullTime']
            match_date_dt = datetime.strptime(match['utcDate'], "%Y-%m-%dT%H:%M:%SZ")
            
            # تجاهل المباريات التي لم تنته بعد (إذا وجدت)
            if score['home'] is None or score['away'] is None:
                continue

            # الحصول على تقييم ELO **قبل** المباراة
            home_elo_before = elo_ratings.get(home_team['id'], 1500.0)
            away_elo_before = elo_ratings.get(away_team['id'], 1500.0)
            
            # حساب الفورمة **قبل** المباراة
            # نمرر قائمة المباريات حتى المباراة الحالية
            home_form = get_team_form(home_team['id'], matches_sorted[:i], match_date_dt, games_to_look_back=5)
            away_form = get_team_form(away_team['id'], matches_sorted[:i], match_date_dt, games_to_look_back=5)

            # إنشاء صف البيانات الجديد مع كل الميزات
            processed_row = {
                "match_id": match['id'],
                "date": match['utcDate'],
                "competition": comp_name,
                "home_team_id": home_team['id'],
                "home_team_name": home_team['name'],
                "away_team_id": away_team['id'],
                "away_team_name": away_team['name'],
                "home_team_elo": home_elo_before,
                "away_team_elo": away_elo_before,
                "elo_diff": home_elo_before - away_elo_before,
                "home_form_points": home_form['points'],
                "home_form_gs": home_form['goals_scored'],
                "home_form_gc": home_form['goals_conceded'],
                "away_form_points": away_form['points'],
                "away_form_gs": away_form['goals_scored'],
                "away_form_gc": away_form['goals_conceded'],
                # بيانات يدوية يمكن إضافتها لاحقًا
                "home_team_league_pos": 0,
                "away_team_league_pos": 0,
                # النتيجة النهائية (الهدف الذي سيتوقعه النموذج)
                "result_ft_home_goals": score['home'],
                "result_ft_away_goals": score['away'],
            }
            all_processed_matches.append(processed_row)
            
            # 4. تحديث تقييمات ELO **بعد** المباراة للتحضير للمباراة التالية
            Eh = 1.0 / (1.0 + 10 ** (-(home_elo_before + H_adv - away_elo_before) / 400.0))
            Sh = 1.0 if score['home'] > score['away'] else 0.5 if score['home'] == score['away'] else 0.0
            
            elo_ratings[home_team['id']] = home_elo_before + K * (Sh - Eh)
            elo_ratings[away_team['id']] = away_elo_before - K * (Sh - Eh)

        print(f"Processed {len(matches_sorted)} matches for {comp_name}.")
        
    # 5. حفظ كل البيانات في ملف CSV واحد
    if all_processed_matches:
        df = pd.DataFrame(all_processed_matches)
        df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"\n✅ Successfully built dataset with {len(df)} matches.")
        print(f"Saved to '{OUTPUT_CSV_FILE}'")
    else:
        print("\nNo data was processed.")


if __name__ == "__main__":
    main()

