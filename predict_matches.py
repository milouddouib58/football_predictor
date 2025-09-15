import os
import pandas as pd
import joblib
from datetime import datetime
from dotenv import load_dotenv

# استيراد الأدوات التي نحتاجها من مشروعنا
from football_predictor.client import FootballDataClient, AnalysisCache
from football_predictor.model import PredictionModel
from build_dataset import get_team_form # نستورد دالة حساب الفورمة من السكربت السابق

# تحميل مفتاح API
load_dotenv()

# --- إعدادات ---
# ❗ اختر الدوري الذي تريد التنبؤ بمبارياته من هنا
COMPETITION_NAME = "Premier League (England)"
COMPETITIONS = {
    "Premier League (England)": 2021, "La Liga (Spain)": 2014,
    "Serie A (Italy)": 2019, "Bundesliga (Germany)": 2002, "Ligue 1 (France)": 2015,
}
MODEL_FILE = "football_model.joblib"
HISTORICAL_DATA_FILE = "historical_dataset.csv"

def predict_upcoming():
    """
    الوظيفة الرئيسية لجلب المباريات القادمة، حساب ميزاتها، والتنبؤ بنتائجها.
    """
    # 1. تحميل النموذج المدرب والبيانات التاريخية
    print(f"Loading trained model from '{MODEL_FILE}'...")
    model = joblib.load(MODEL_FILE)
    
    print(f"Loading historical data from '{HISTORICAL_DATA_FILE}' for context...")
    historical_df = pd.read_csv(HISTORICAL_DATA_FILE)
    
    # 2. إعداد الاتصال بالـ API لجلب المباريات القادمة
    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    client = FootballDataClient(api_key=api_key, min_interval=1.0, cache=AnalysisCache())
    predictor_model = PredictionModel(client=client, cache=AnalysisCache())
    
    competition_id = COMPETITIONS.get(COMPETITION_NAME)
    if not competition_id:
        print(f"Error: Competition '{COMPETITION_NAME}' not found.")
        return

    print(f"\nFetching upcoming matches for {COMPETITION_NAME}...")
    upcoming_fixtures = predictor_model.get_upcoming_fixtures(comp_id=competition_id, horizon_days=7)

    if not upcoming_fixtures:
        print("No upcoming matches found in the next 7 days.")
        return
        
    print(f"Found {len(upcoming_fixtures)} upcoming matches. Preparing for prediction...")

    # 3. حساب أحدث تقييم ELO لكل فريق من بياناتنا التاريخية
    # هذا يعطينا أحدث تقييم ELO معروف لكل فريق
    latest_elo = {}
    df_sorted = historical_df.sort_values(by='date')
    for index, row in df_sorted.iterrows():
        latest_elo[row['home_team_id']] = row['home_team_elo']
        latest_elo[row['away_team_id']] = row['away_team_elo']

    # 4. التنبؤ بكل مباراة قادمة
    predictions = []
    for fixture in upcoming_fixtures:
        home_team = fixture['homeTeam']
        away_team = fixture['awayTeam']
        
        # حساب الميزات الحالية للمباراة القادمة
        home_elo = latest_elo.get(home_team['id'], 1500.0) # 1500 كقيمة افتراضية لفريق جديد
        away_elo = latest_elo.get(away_team['id'], 1500.0)
        elo_diff = home_elo - away_elo

        # تحويل البيانات التاريخية إلى صيغة مناسبة لدالة الفورمة
        historical_matches_list = historical_df.rename(columns={
            'home_team_id': 'homeTeam', 'away_team_id': 'awayTeam', 'result_ft_home_goals': 'home', 'result_ft_away_goals': 'away'
        }).to_dict('records')
        
        # إعداد البيانات الفرعية المطلوبة للدالة
        for record in historical_matches_list:
            record['homeTeam'] = {'id': record['homeTeam']}
            record['awayTeam'] = {'id': record['awayTeam']}
            record['score'] = {'fullTime': {'home': record['home'], 'away': record['away']}}

        today = datetime.utcnow()
        home_form = get_team_form(home_team['id'], historical_matches_list, today, 5)
        away_form = get_team_form(away_team['id'], historical_matches_list, today, 5)

        # إنشاء DataFrame بنفس هيكل بيانات التدريب
        features_df = pd.DataFrame([{
            'home_team_elo': home_elo,
            'away_team_elo': away_elo,
            'elo_diff': elo_diff,
            'home_form_points': home_form['points'],
            'home_form_gs': home_form['goals_scored'],
            'home_form_gc': home_form['goals_conceded'],
            'away_form_points': away_form['points'],
            'away_form_gs': away_form['goals_scored'],
            'away_form_gc': away_form['goals_conceded'],
        }])

        # استخدام النموذج للتنبؤ بالاحتمالات
        # .predict_proba يعطينا احتمالية كل نتيجة [P(فوز المضيف), P(تعادل), P(فوز الضيف)]
        probabilities = model.predict_proba(features_df)[0]
        
        predictions.append({
            "match": f"{home_team['name']} vs {away_team['name']}",
            "home_win_prob": probabilities[0],
            "draw_prob": probabilities[1],
            "away_win_prob": probabilities[2],
        })

    # 5. عرض النتائج النهائية
    print("\n--- Predictions ---")
    for pred in predictions:
        print(f"\n📅 Match: {pred['match']}")
        print(f"  🏠 Home Win: {pred['home_win_prob']:.1%}")
        print(f"  ⚖️ Draw:     {pred['draw_prob']:.1%}")
        print(f"  ✈️ Away Win: {pred['away_win_prob']:.1%}")
        
if __name__ == "__main__":
    predict_upcoming()
