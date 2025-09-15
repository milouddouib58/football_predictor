# file: app.py

import streamlit as st
import os
import traceback
from dotenv import load_dotenv
import pandas as pd

# استيراد الوحدات من حزمة المشروع
from football_predictor.model import PredictionModel
from football_predictor.client import FootballDataClient, AnalysisCache
from football_predictor.settings import MIN_INTERVAL_SEC
import football_predictor.settings as default_settings
from football_predictor.utils import log

# تحميل متغيرات البيئة من ملف .env (يعمل فقط في بيئة التطوير المحلية)
load_dotenv()

# --- إعدادات الصفحة والتصميم العام ---
st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚽ Football Match Predictor")
st.markdown("لوحة تحكم تفاعلية لتوقع نتائج مباريات كرة القدم بناءً على نموذج إحصائي متقدم.")

# --- البيانات الأساسية وقواميس الخيارات ---
COMPETITIONS = {
    "Premier League (England)": 2021,
    "La Liga (Spain)": 2014,
    "Serie A (Italy)": 2019,
    "Bundesliga (Germany)": 2002,
    "Ligue 1 (France)": 2015,
    "UEFA Champions League": 2001,
}

# --- تهيئة وتحميل النموذج (يتم تخزينه مؤقتًا) ---
@st.cache_resource(show_spinner="... تهيئة نموذج التوقع لأول مرة")
def load_model():
    """
    يقوم بتهيئة كائنات النموذج والعميل مرة واحدة فقط ويخزنها مؤقتًا.
    """
    log("Initializing model for Streamlit...")
    # الوصول إلى مفتاح API من أسرار Streamlit أولاً (للبيئة السحابية)
    api_key = st.secrets.get("FOOTBALL_DATA_API_KEY", os.getenv("FOOTBALL_DATA_API_KEY"))
    
    if not api_key:
        raise ValueError("API key not found. Please add it to Streamlit secrets or your local .env file.")
    
    client = FootballDataClient(api_key=api_key, min_interval=MIN_INTERVAL_SEC)
    cache = AnalysisCache()
    model = PredictionModel(client=client, cache=cache)
    log("Model initialized successfully.")
    return model

# تحميل النموذج ومعالجة أي خطأ في التهيئة
try:
    model = load_model()
except Exception as e:
    st.error(f"حدث خطأ فادح أثناء تهيئة النموذج: {e}")
    st.info("تأكد من إضافة FOOTBALL_DATA_API_KEY في قسم الأسرار (Secrets) في إعدادات التطبيق على منصة Streamlit أو في ملف .env المحلي.")
    st.stop() # إيقاف تشغيل التطبيق إذا فشل تحميل النموذج

# --- الشريط الجانبي للإعدادات المتقدمة ---
with st.sidebar:
    st.header("⚙️ إعدادات النموذج المتقدمة")
    st.markdown("يمكنك تعديل معاملات النموذج من هنا للتأثير على طريقة الحساب.")

    elo_scale = st.slider(
        "تأثير عامل ELO", 
        min_value=0.0, max_value=1.0, 
        value=default_settings.ELO_SCALE, step=0.01,
        help="يتحكم في مدى تأثير فارق تصنيف ELO على توقع الأهداف. القيمة الأعلى تعني تأثيرًا أكبر."
    )
    prior_games = st.slider(
        "وزن المباريات السابقة (Prior)",
        min_value=5, max_value=25,
        value=default_settings.PRIOR_GAMES,
        help="عدد المباريات الافتراضي الذي يضاف لتقليل تأثير العينات الصغيرة في حساب قوة الفرق."
    )
    
    advanced_settings = {
        "elo_scale": elo_scale,
        "prior_games": prior_games,
    }

# --- الواجهة الرئيسية للتطبيق ---
st.subheader("1. اختر المسابقة")
comp_name = st.selectbox("اختر الدوري:", options=list(COMPETITIONS.keys()), help="اختر الدوري الذي تود التوقع فيه.")
selected_comp_id = COMPETITIONS[comp_name]

@st.cache_data(ttl=3600*6, show_spinner="... تحميل قائمة الفرق") # تخزين قائمة الفرق لمدة 6 ساعات
def get_teams_for_competition(comp_id):
    """
    تجلب قائمة الفرق لمسابقة معينة وتخزنها مؤقتًا.
    """
    log(f"Fetching teams for competition {comp_id}")
    teams_data = model._get_competition_teams(comp_id)
    if not teams_data:
        return {}, {}
    # إنشاء قاموسين: واحد للأسماء للعرض، والآخر للربط بين الاسم والهوية
    team_names = sorted([team['shortName'] for team in teams_data if team.get('shortName')])
    team_map = {team['shortName']: team['id'] for team in teams_data if team.get('shortName')}
    return team_names, team_map

team_names, team_map = get_teams_for_competition(selected_comp_id)

if not team_names:
    st.error(f"لم يتم العثور على فرق لهذه المسابقة. قد تكون هناك مشكلة في الاتصال بالـ API أو أن المسابقة غير نشطة حاليًا.")
else:
    st.subheader("2. اختر الفرق")
    col1, col2 = st.columns(2)
    with col1:
        home_team_name = st.selectbox("الفريق المضيف:", options=team_names, index=0)
    with col2:
        away_team_name = st.selectbox("الفريق الضيف:", options=team_names, index=1)

    st.subheader("3. ابدأ التوقع")
    if st.button("🚀 توقع النتيجة", type="primary", use_container_width=True):
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
                    
                    # --- عرض النتائج المحسن ---
                    probs = result["probabilities"]["1x2"]
                    home_name_res = result["teams"]["home"]["name"] or home_team_name
                    away_name_res = result["teams"]["away"]["name"] or away_team_name

                    st.markdown("---")
                    st.subheader("📊 احتمالات نتيجة المباراة (1X2)")
                    
                    # عرض الاحتمالات باستخدام الأعمدة والمقاييس
                    p_col1, p_col2, p_col3 = st.columns(3)
                    p_col1.metric(f"فوز {home_name_res}", f"{probs['home']:.1f}%")
                    p_col2.metric("تعادل", f"{probs['draw']:.1f}%")
                    p_col3.metric(f"فوز {away_name_res}", f"{probs['away']:.1f}%")

                    # عرض رسم بياني للاحتمالات
                    chart_data = pd.DataFrame({
                        "النتيجة": [home_name_res, "تعادل", away_name_res],
                        "الاحتمالية (%)": [probs['home'], probs['draw'], probs['away']]
                    })
                    st.bar_chart(chart_data, x="النتيجة", y="الاحتمالية (%)")
                    
                    st.markdown("---")
                    st.subheader("📈 مقاييس إضافية")
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Rating الفريق المضيف (ELO)", f"{result['elo']['home_rating']:.0f}")
                    m_col2.metric("Rating الفريق الضيف (ELO)", f"{result['elo']['away_rating']:.0f}")
                    m_col3.metric("مجموع الأهداف المتوقع", f"{result['lambdas']['home_final'] + result['lambdas']['away_final']:.2f}")

                    with st.expander("عرض التفاصيل التحليلية الكاملة (JSON)"):
                        st.json(result)

                except Exception as e:
                    st.error(f"حدث خطأ أثناء التوقع: {e}")
                    st.code(traceback.format_exc())
