# file: app.py
import streamlit as st
import os
import traceback
from dotenv import load_dotenv

from football_predictor.model import PredictionModel
from football_predictor.client import FootballDataClient, AnalysisCache
from football_predictor.settings import MIN_INTERVAL_SEC
from football_predictor.utils import log

load_dotenv()
st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")
st.title("⚽ Football Match Predictor")

@st.cache_resource(show_spinner="... تهيئة نموذج التوقع")
def load_model():
    api_key = st.secrets.get("FOOTBALL_DATA_API_KEY", os.getenv("FOOTBALL_DATA_API_KEY"))
    if not api_key:
        raise ValueError("API key not found. Please add it to Streamlit secrets or .env file.")
    
    client = FootballDataClient(api_key=api_key, min_interval=MIN_INTERVAL_SEC)
    cache = AnalysisCache()
    model = PredictionModel(client=client, cache=cache)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"فشل تهيئة النموذج: {e}")
    st.stop()

with st.form("prediction_form"):
    st.write("اختر الفرق من الدوري الإنجليزي الممتاز")
    col1, col2 = st.columns(2)
    with col1:
        team1_name = st.text_input("الفريق المضيف", "Arsenal FC")
    with col2:
        team2_name = st.text_input("الفريق الضيف", "Manchester City FC")
    submitted = st.form_submit_button("🚀 توقع النتيجة")

if submitted:
    with st.spinner(f"جاري تحليل مباراة {team1_name} ضد {team2_name}..."):
        try:
            result = model.predict(team1_name=team1_name, team2_name=team2_name, team1_is_home=True)
            st.success("تم التوقع بنجاح!")
            
            probs = result["probabilities"]["1x2"]
            home_name = result["teams"]["home"]["name"]
            away_name = result["teams"]["away"]["name"]

            p_col1, p_col2, p_col3 = st.columns(3)
            p_col1.metric(f"فوز {home_name}", f"{probs['home']:.1f}%")
            p_col2.metric("تعادل", f"{probs['draw']:.1f}%")
            p_col3.metric(f"فوز {away_name}", f"{probs['away']:.1f}%")

            with st.expander("عرض التفاصيل التحليلية الكاملة"):
                st.json(result)
        except Exception as e:
            st.error(f"حدث خطأ أثناء التوقع: {e}")
            st.code(traceback.format_exc())
