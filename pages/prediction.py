import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
import os

# -------------------- Page Config --------------------
st.set_page_config(page_title="Individual Prediction", layout="centered")

# -------------------- Page Header --------------------
st.markdown("<h2 style='color:#4B8BBE;'>🧠 Individual Burnout Risk Prediction</h2>", unsafe_allow_html=True)
st.markdown("Fill in your workplace activity details below to estimate your burnout risk.")

# -------------------- Load Model & Scaler --------------------
model = xgb.Booster()
model.load_model("burnout_model.json")
scaler = joblib.load("scaler.pkl")

# -------------------- Load Burnout Threshold --------------------
threshold_path = "burnout_threshold.txt"
burnout_threshold = 0.5
if os.path.exists(threshold_path):
    with open(threshold_path, "r") as f:
        burnout_threshold = float(f.read().strip())

# -------------------- Input Form --------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        work_hours_per_week = st.slider("Work Hours per Week", 30, 80, 45)
        after_hours_emails = st.slider("After-hours Emails per Week", 0, 50, 10)
        meeting_count = st.slider("Weekly Meetings", 0, 40, 10)

    with col2:
        negative_sentiment_score = st.slider("Negative Sentiment Score (0.0–1.0)", 0.0, 1.0, 0.3, step=0.01)
        task_completion_rate = st.slider("Task Completion Rate (0.0–1.0)", 0.0, 1.0, 0.8, step=0.01)

    submitted = st.form_submit_button("🔍 Predict Burnout Risk")

# -------------------- Prediction --------------------
if submitted:
    input_data = {
        "work_hours_per_week": work_hours_per_week,
        "after_hours_emails": after_hours_emails,
        "negative_sentiment_score": negative_sentiment_score,
        "meeting_count": meeting_count,
        "task_completion_rate": task_completion_rate
    }

    expected_columns = [
        "work_hours_per_week",
        "after_hours_emails",
        "negative_sentiment_score",
        "meeting_count",
        "task_completion_rate"
    ]

    input_df = pd.DataFrame([input_data])[expected_columns]
    X_scaled = scaler.transform(input_df)
    dtest = xgb.DMatrix(X_scaled)
    proba = model.predict(dtest)[0]
    prediction = int(proba >= burnout_threshold)

    # -------------------- Results Display --------------------
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Burnout Risk Detected\n\n**Probability: {proba:.2f}**")
    else:
        st.success(f"✅ No Burnout Risk Detected\n\n**Probability: {proba:.2f}**")

    st.markdown("<small>Note: This result is based on a machine learning model trained on synthetic data and is not a medical diagnosis.</small>", unsafe_allow_html=True)
