import os
import traceback
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# حزمة المشروع
from football_predictor.client import AnalysisCache, FootballDataClient
from football_predictor.model import PredictionModel
from football_predictor.settings import MIN_INTERVAL_SEC
import football_predictor.settings as default_settings
from football_predictor.utils import log

# تحميل متغيرات البيئة (محلياً)
load_dotenv()

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚽ Football Match Predictor")
st.markdown("يتم جلب المباريات القادمة تلقائيًا، وتوقع مباراة واحدة أو جميع المباريات المعروضة مع تحكم كامل في الإعدادات المتقدمة.")

# المسابقات المدعومة
COMPETITIONS = {
    "Premier League (England)": 2021,
    "La Liga (Spain)": 2014,
    "Serie A (Italy)": 2019,
    "Bundesliga (Germany)": 2002,
    "Ligue 1 (France)": 2015,
    "UEFA Champions League": 2001,
}


@st.cache_resource(show_spinner="... تهيئة نموذج التوقع لأول مرة")
def load_model() -> PredictionModel:
    """
    تهيئة كائنات العميل والنموذج مرة واحدة وتخزينها مؤقتًا.
    """
    log("Initializing model for Streamlit...")
    api_key = st.secrets.get("FOOTBALL_DATA_API_KEY", os.getenv("FOOTBALL_DATA_API_KEY"))
    if not api_key:
        raise ValueError(
            "API key not found. Please add FOOTBALL_DATA_API_KEY to Streamlit secrets or your .env file."
        )
    cache = AnalysisCache()
    client = FootballDataClient(api_key=api_key, min_interval=MIN_INTERVAL_SEC, cache=cache)
    model = PredictionModel(client=client, cache=cache)
    log("Model initialized successfully.")
    return model


# تحميل النموذج ومعالجة الأخطاء
try:
    model = load_model()
except Exception as e:
    st.error(f"حدث خطأ فادح أثناء تهيئة النموذج: {e}")
    st.info("تأكد من إضافة FOOTBALL_DATA_API_KEY في Streamlit Secrets أو في ملف .env المحلي.")
    st.stop()

# الشريط الجانبي - إعدادات متقدمة
with st.sidebar:
    st.header("⚙️ إعدادات النموذج")
    st.markdown("تحكم في معاملات النموذج وطريقة جلب البيانات.")

    # إعدادات النموذج
    elo_scale = st.slider(
        "تأثير عامل ELO",
        min_value=0.0,
        max_value=1.0,
        value=float(default_settings.ELO_SCALE),
        step=0.01,
        help="يتحكم في مدى تأثير فارق تصنيف ELO على توقع الأهداف. كلما زاد، زاد التأثير.",
    )
    prior_games = st.slider(
        "وزن المباريات السابقة (Prior)",
        min_value=5,
        max_value=25,
        value=int(default_settings.PRIOR_GAMES),
        help="عدد المباريات الافتراضي المستخدم لتخفيف تأثير العينات الصغيرة عند تقدير قوة الفرق.",
    )
    max_goals_grid = st.slider(
        "أقصى عدد أهداف في شبكة بواسون",
        min_value=5,
        max_value=12,
        value=int(default_settings.MAX_GOALS_GRID),
        help="زيادة القيمة قد تعطي دقة أفضل قليلاً على حساب الوقت.",
    )

    st.markdown("---")
    st.header("🗓️ إعدادات البيانات")
    use_season_dates = st.checkbox(
        "استخدام تواريخ الموسم الحالي كأساس",
        value=True,
        help="إذا فُعل: يبدأ التاريخ من بداية الموسم. إذا لم يفعل: يُستخدم عدد الأيام المحدد مع التاريخ الحالي.",
    )
    lookback_days = st.slider(
        "عدد الأيام للتاريخ (Lookback)",
        min_value=30,
        max_value=720,
        value=int(default_settings.DEFAULT_LOOKBACK_DAYS),
        step=15,
        help="سيتم استخدام آخر X يوم من المباريات التاريخية لبناء النموذج (مقيد بتاريخ بداية الموسم إذا كان الخيار أعلاه مفعل).",
    )
    history_match_limit = st.slider(
        "الحد الأقصى لعدد المباريات التاريخية المستخدمة",
        min_value=50,
        max_value=2000,
        value=int(default_settings.HISTORY_MATCH_LIMIT),
        step=50,
        help="تقليص هذا العدد يسرّع البناء لكن قد يقلل الدقة.",
    )
    fixture_horizon_days = st.slider(
        "عرض المباريات القادمة خلال (أيام)",
        min_value=1,
        max_value=30,
        value=int(default_settings.DEFAULT_FIXTURE_HORIZON_DAYS),
        help="سيتم جلب المباريات القادمة خلال هذا النطاق الزمني.",
    )

    st.markdown("---")
    if st.button("🧹 مسح الكاش", use_container_width=True):
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
            if hasattr(model, "cache"):
                model.cache.clear()
            if hasattr(model, "client"):
                model.client.clear_cache()
            st.success("تم مسح جميع الكاش بنجاح.")
            st.rerun()
        except Exception as e:
            st.error(f"تعذر مسح الكاش: {e}")

    advanced_settings = {
        "elo_scale": elo_scale,
        "prior_games": prior_games,
        "max_goals_grid": max_goals_grid,
        "use_season_dates": use_season_dates,
        "lookback_days": lookback_days,
        "history_match_limit": history_match_limit,
        "fixture_horizon_days": fixture_horizon_days,
    }

# واجهة الاستخدام الرئيسية
st.subheader("1. اختر المسابقة")
comp_name = st.selectbox("اختر الدوري:", options=list(COMPETITIONS.keys()), help="اختر الدوري الذي تود التوقع فيه.")
selected_comp_id = COMPETITIONS[comp_name]

st.subheader("2. جلب المباريات القادمة")

# جلب تلقائي للمباريات عند تغيير المسابقة أو أفق الأيام
def fetch_and_store_fixtures(comp_id: int, horizon_days: int):
    try:
        fixtures = model.get_upcoming_fixtures(comp_id=comp_id, horizon_days=horizon_days)
        st.session_state["fixtures"] = fixtures
        st.session_state["fixtures_comp"] = comp_id
        st.session_state["fixtures_horizon"] = horizon_days
        st.session_state["fixtures_loaded_at"] = datetime.utcnow().isoformat()
        return fixtures
    except Exception as e:
        st.error(f"تعذر جلب المباريات القادمة: {e}")
        st.code(traceback.format_exc())
        return []

# قرر هل نحتاج إعادة الجلب تلقائياً
need_fetch = False
if "fixtures" not in st.session_state:
    need_fetch = True
else:
    if st.session_state.get("fixtures_comp") != selected_comp_id:
        need_fetch = True
    if st.session_state.get("fixtures_horizon") != advanced_settings["fixture_horizon_days"]:
        need_fetch = True

colf1, colf2, colf3 = st.columns([2, 1, 1])
with colf1:
    st.caption(f"سيتم جلب المباريات خلال {advanced_settings['fixture_horizon_days']} يومًا القادمة.")
with colf2:
    if st.button("🔄 تحديث المباريات", use_container_width=True):
        if hasattr(model, "client"):
            model.client.clear_cache()
        need_fetch = True
with colf3:
    if st.button("📅 جلب الآن", use_container_width=True):
        need_fetch = True

if need_fetch:
    fixtures = fetch_and_store_fixtures(selected_comp_id, int(advanced_settings["fixture_horizon_days"]))
else:
    fixtures = st.session_state.get("fixtures", [])

if not fixtures:
    st.info("لا توجد مباريات قادمة ضمن النطاق المحدد.")
    st.stop()

# عرض المباريات القادمة واختيار واحدة للتوقع
df_fixtures = pd.DataFrame(
    [
        {
            "date": m.get("utcDate"),
            "status": m.get("status"),
            "home": (m.get("homeTeam") or {}).get("shortName") or (m.get("homeTeam") or {}).get("name"),
            "away": (m.get("awayTeam") or {}).get("shortName") or (m.get("awayTeam") or {}).get("name"),
            "home_id": (m.get("homeTeam") or {}).get("id"),
            "away_id": (m.get("awayTeam") or {}).get("id"),
        }
        for m in fixtures
    ]
)

st.dataframe(df_fixtures[["date", "status", "home", "away"]], use_container_width=True, hide_index=True)

# اختيار مباراة من القائمة
options = list(range(len(df_fixtures)))
labels = [
    f"{df_fixtures.iloc[i]['date']} — {df_fixtures.iloc[i]['home']} vs {df_fixtures.iloc[i]['away']}"
    for i in options
]
selected_idx = st.selectbox("3) اختر مباراة من المباريات القادمة:", options=options, format_func=lambda i: labels[i])

# زر توقع المباراة المختارة
if st.button("🎯 توقع المباراة المختارة", type="primary", use_container_width=True):
    row = df_fixtures.iloc[selected_idx]
    home_team_id = int(row["home_id"])
    away_team_id = int(row["away_id"])
    home_name = str(row["home"])
    away_name = str(row["away"])

    with st.spinner(f"جاري تحليل مباراة {home_name} ضد {away_name}..."):
        try:
            result = model.predict(
                comp_id=selected_comp_id,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                advanced_settings=advanced_settings,
            )
            st.success("تم التوقع بنجاح!")

            probs = result["probabilities"]["1x2"]
            extras = result["probabilities"]
            home_name_res = result["teams"]["home"]["name"] or home_name
            away_name_res = result["teams"]["away"]["name"] or away_name

            st.markdown("---")
            st.subheader("📊 احتمالات نتيجة المباراة (1X2)")
            p_col1, p_col2, p_col3 = st.columns(3)
            p_col1.metric(f"فوز {home_name_res}", f"{probs['home']:.1f}%")
            p_col2.metric("تعادل", f"{probs['draw']:.1f}%")
            p_col3.metric(f"فوز {away_name_res}", f"{probs['away']:.1f}%")

            chart_data = pd.DataFrame(
                {"النتيجة": [home_name_res, "تعادل", away_name_res], "الاحتمالية (%)": [probs["home"], probs["draw"], probs["away"]]}
            )
            st.bar_chart(chart_data, x="النتيجة", y="الاحتمالية (%)")

            st.markdown("---")
            st.subheader("🎯 احتمالات إضافية")
            e_col1, e_col2, e_col3 = st.columns(3)
            e_col1.metric("Over 2.5", f"{extras['over_under']['over_2_5']:.1f}%")
            e_col2.metric("Under 2.5", f"{extras['over_under']['under_2_5']:.1f}%")
            e_col3.metric("BTTS (نعم)", f"{extras['btts']['yes']:.1f}%")

            st.markdown("#### أعلى النتائج احتمالاً")
            top_df = pd.DataFrame(extras["top_scorelines"])
            if not top_df.empty:
                st.table(top_df)
            else:
                st.info("لا توجد نتائج مرجحة كفاية لعرضها.")

            st.markdown("---")
            st.subheader("📈 مقاييس إضافية")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Rating الفريق المضيف (ELO)", f"{result['elo']['home_rating']:.0f}")
            m_col2.metric("Rating الفريق الضيف (ELO)", f"{result['elo']['away_rating']:.0f}")
            m_col3.metric(
                "مجموع الأهداف المتوقع",
                f"{result['lambdas']['home_final'] + result['lambdas']['away_final']:.2f}",
            )
            if extras.get("top_scorelines"):
                likely = extras["top_scorelines"][0]
                st.metric("أكثر نتيجة محتملة", f"{likely['score']}", f"{likely['p']:.1f}%")

            with st.expander("عرض التفاصيل التحليلية الكاملة (JSON)"):
                st.json(result)

        except Exception as e:
            st.error(f"حدث خطأ أثناء التوقع: {e}")
            st.code(traceback.format_exc())

# زر توقع جميع المباريات المعروضة
if st.button("🧮 توقع كل المباريات المعروضة", use_container_width=True):
    with st.spinner("جاري حساب توقعات جميع المباريات القادمة..."):
        try:
            bulk_results = model.predict_bulk_for_scheduled(
                comp_id=selected_comp_id,
                horizon_days=int(advanced_settings["fixture_horizon_days"]),
                advanced_settings=advanced_settings,
            )
            if not bulk_results:
                st.info("تعذر حساب التوقعات للمباريات القادمة.")
            else:
                df_pred = pd.DataFrame(
                    [
                        {
                            "date": r["date"],
                            "home": r["teams"]["home"]["name"],
                            "away": r["teams"]["away"]["name"],
                            "P(Home%)": r["probabilities"]["1x2"]["home"],
                            "P(Draw%)": r["probabilities"]["1x2"]["draw"],
                            "P(Away%)": r["probabilities"]["1x2"]["away"],
                            "Over2.5%": r["probabilities"]["over_under"]["over_2_5"],
                            "BTTS%": r["probabilities"]["btts"]["yes"],
                            "Home λ": r["lambdas"]["home_final"],
                            "Away λ": r["lambdas"]["away_final"],
                        }
                        for r in bulk_results
                    ]
                ).sort_values("date")
                st.dataframe(df_pred, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"حدث خطأ أثناء التوقع الجماعي: {e}")
            st.code(traceback.format_exc())
