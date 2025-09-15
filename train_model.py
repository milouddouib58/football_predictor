import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- إعدادات ---
INPUT_CSV_FILE = "historical_dataset.csv"
MODEL_OUTPUT_FILE = "football_model.joblib"

def train():
    """
    الوظيفة الرئيسية لقراءة البيانات، تدريب النموذج، وتقييم أدائه.
    """
    # 1. تحميل وإعداد البيانات
    print("Loading dataset...")
    df = pd.read_csv(INPUT_CSV_FILE)

    # تجاهل المباريات التي ليس لها بيانات كافية (من بداية السجل التاريخي)
    df.dropna(inplace=True)
    df = df[(df['home_form_points'] > 0) | (df['away_form_points'] > 0)]

    if df.empty:
        print("Dataset is empty after cleaning. Cannot train model.")
        return

    # 2. هندسة الميزات وتحديد الهدف
    # الهدف: التنبؤ بنتيجة المباراة (فوز المضيف=0, تعادل=1, فوز الضيف=2)
    def get_result(row):
        if row['result_ft_home_goals'] > row['result_ft_away_goals']:
            return 0  # فوز المضيف
        elif row['result_ft_home_goals'] == row['result_ft_away_goals']:
            return 1  # تعادل
        else:
            return 2  # فوز الضيف

    df['target'] = df.apply(get_result, axis=1)

    # تحديد "الميزات" التي سيتعلم منها النموذج
    features = [
        'home_team_elo', 'away_team_elo', 'elo_diff',
        'home_form_points', 'home_form_gs', 'home_form_gc',
        'away_form_points', 'away_form_gs', 'away_form_gc',
        # يمكنك إضافة الميزات اليدوية هنا إذا قمت بملئها
        # 'home_team_league_pos', 'away_team_league_pos',
    ]

    X = df[features]
    y = df['target']

    # 3. تقسيم البيانات إلى مجموعة تدريب ومجموعة اختبار
    # 80% للتدريب، 20% للاختبار لتقييم أداء النموذج على بيانات لم يرها من قبل
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training on {len(X_train)} matches, testing on {len(X_test)} matches.")

    # 4. تدريب نموذج XGBoost
    print("Training XGBoost model...")
    # multi:softprob سيعطينا احتمالية لكل نتيجة (فوز، تعادل، خسارة)
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. تقييم أداء النموذج
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {accuracy:.2%}")
    
    print("\nClassification Report:")
    # target_names يترجم الأرقام (0,1,2) إلى أسماء مفهومة
    print(classification_report(y_test, y_pred, target_names=['Home Win', 'Draw', 'Away Win']))
    
    # 6. حفظ النموذج المدرب
    print(f"\nSaving trained model to '{MODEL_OUTPUT_FILE}'...")
    joblib.dump(model, MODEL_OUTPUT_FILE)
    print("✅ Model saved successfully.")


if __name__ == "__main__":
    train()
