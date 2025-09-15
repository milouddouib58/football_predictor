# file: app.py
import streamlit as st
import os
import traceback
from dotenv import load_dotenv

from football_predictor.model import PredictionModel
from football_predictor.client import FootballDataClient, AnalysisCache
from football_predictor.settings import MIN_INTERVAL_SEC
import football_predictor.settings as default_settings
from football_predictor.utils import log

load_dotenv()
st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")
st.title("⚽ Football Match Predictor")

# --- البيانات الأساسية ---
COMPETITIONS = {
    "Premier League (England)": 2021,
    "La Liga (Spain)": 2014,
    "Serie A (Italy)": 2019,
    "Bundesliga (Germany)": 2002,
    "Ligue 1 (France)": 2015,
    "UEFA Champions League": 2001,
}

@st.cache_resource(show_spinner="... تهيئة نموذج التوقع")
def load_model():
    # ... (محتوى هذه الدالة يبقى كما هو)
    pass

model = load_model()

# --- الشريط الجانبي للإعدادات المتقدمة ---
with st.sidebar:
    st.header("⚙️ إعدادات النموذج المتقدمة")
    st.write("يمكنك تعديل معاملات النموذج من هنا:")

    elo_scale = st.slider(
        "تأثير عامل ELO", 
        min_value=0.0, max_value=1.0, 
        value=default_settings.ELO_SCALE, step=0.01,
        help="يتحكم في مدى تأثير فارق تصنيف ELO على توقع الأهداف. القيمة الأعلى تعني تأثيرًا أكبر."
    )
    prior_games = st.slider(
        "وزن المباريات السابقة (Prior)",
        min_value=5, max_value=20,
        value=default_settings.PRIOR_GAMES,
        help="عدد المباريات الافتراضي الذي يضاف لتقليل تأثير العينات الصغيرة في حساب قوة الفرق."
    )
    
    advanced_settings = {
        "elo_scale": elo_scale,
        "prior_games": prior_games,
    }

# --- الواجهة الرئيسية ---
st.subheader("1. اختر المسابقة")
comp_name = st.selectbox("اختر الدوري:", options=list(COMPETITIONS.keys()))
selected_comp_id = COMPETITIONS[comp_name]

@st.cache_data(ttl=3600*6) # تخزين قائمة الفرق لمدة 6 ساعات
def get_teams_for_competition(comp_id):
    log(f"Fetching teams for competition {comp_id}")
    return model._get_competition_teams(comp_id)

teams = get_teams_for_competition(selected_comp_id)

if not teams:
    st.error(f"لم يتم العثور على فرق لهذه المسابقة. قد تكون هناك مشكلة في الاتصال بالـ API.")
else:
    team_names = sorted([team['shortName'] for team in teams if team.get('shortName')])
    team_map = {team['shortName']: team['id'] for team in teams if team.get('shortName')}
    
    st.subheader("2. اختر الفرق")
    col1, col2 = st.columns(2)
    with col1:
        home_team_name = st.selectbox("الفريق المضيف:", options=team_names, index=0)
    with col2:
        away_team_name = st.selectbox("الفريق الضيف:", options=team_names, index=1)

    st.subheader("3. ابدأ التوقع")
    if st.button("🚀 توقع النتيجة"):
        if home_team_name == away_team_name:
            st.warning("الرجاء اختيار فريقين مختلفين.")
        else:
            home_team_id = team_map[home_team_name]
            away_team_id = team_map[away_team_name]
            
            with st.spinner(f"جاري تحليل مباراة {home_team_name} ضد {away_team_name}..."):
                try:
                    result = model.predict(
                        comp_id=selected_comp_id,
                        home_team_id=home_team_id,
                        away_team_id=away_team_id,
                        advanced_settings=advanced_settings
                    )
                    st.success("تم التوقع بنجاح!")
                    
                    # عرض النتائج بشكل أفضل
                    probs = result["probabilities"]["1x2"]
                    home_name = result["teams"]["home"]["name"]
                    away_name = result["teams"]["away"]["name"]

                    st.subheader("📊 احتمالات نتيجة المباراة (1X2)")
                    p_col1, p_col2, p_col3 = st.columns(3)
                    p_col1.metric(f"فوز {home_name}", f"{probs['home']:.1f}%")
                    p_col2.metric("تعادل", f"{probs['draw']:.1f}%")
                    p_col3.metric(f"فوز {away_name}", f"{probs['away']:.1f}%")

                    st.subheader("📈 مقاييس إضافية")
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("معدل الأهداف المتوقع للمضيف", f"{result['lambdas']['home_final']:.2f}")
                    m_col2.metric("معدل الأهداف المتوقع للضيف", f"{result['lambdas']['away_final']:.2f}")

                    with st.expander("عرض التفاصيل التحليلية الكاملة (JSON)"):
                        st.json(result)
                except Exception as e:
                    st.error(f"حدث خطأ أثناء التوقع: {e}")
                    st.code(traceback.format_exc())

