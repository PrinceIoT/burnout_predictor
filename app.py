import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load model and scaler ---
model = xgb.Booster()
model.load_model("burnout_model.json")
scaler = joblib.load("scaler.pkl")

# --- Load threshold ---
threshold_path = "burnout_threshold.txt"
burnout_threshold = 0.5
if os.path.exists(threshold_path):
    with open(threshold_path, "r") as f:
        burnout_threshold = float(f.read().strip())

# --- Streamlit UI Layout ---
st.set_page_config(page_title="Burnout Prediction App", layout="wide")
st.title("💼 Burnout Risk Prediction System")

tabs = st.tabs(["🏠 Individual Prediction", "📊 Organization Upload & Analysis"])

# ------------------------- TAB 1: INDIVIDUAL PREDICTION -------------------------
with tabs[0]:
    st.subheader("Enter Your Workplace Activity Information")

    work_hours_per_week = st.slider("Work Hours per Week", 30, 80, 45)
    after_hours_emails = st.slider("After-hours Emails per Week", 0, 50, 10)
    negative_sentiment_score = st.slider("Negative Sentiment Score (0.0–1.0)", 0.0, 1.0, 0.3, step=0.01)
    meeting_count = st.slider("Number of Weekly Meetings", 0, 40, 10)
    task_completion_rate = st.slider("Task Completion Rate (0.0–1.0)", 0.0, 1.0, 0.8, step=0.01)

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

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Burnout Risk Detected (Probability: {proba:.2f})")
    else:
        st.success(f"✅ No Burnout Risk Detected (Probability: {proba:.2f})")

# ------------------------- TAB 2: ORGANIZATION UPLOAD -------------------------
with tabs[1]:
    st.subheader("Upload Employee Data CSV")

    st.markdown("""
        Upload a `.csv` file with the following columns:
        - `work_hours_per_week`
        - `after_hours_emails`
        - `negative_sentiment_score`
        - `meeting_count`
        - `task_completion_rate`
    """)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = [
                "work_hours_per_week",
                "after_hours_emails",
                "negative_sentiment_score",
                "meeting_count",
                "task_completion_rate"
            ]

            if not all(col in df.columns for col in required_cols):
                st.error("❌ Uploaded CSV is missing required columns.")
            else:
                df = df[required_cols]  # reorder
                X_scaled = scaler.transform(df)
                dtest = xgb.DMatrix(X_scaled)
                df["burnout_probability"] = model.predict(dtest)
                df["burnout_prediction"] = (df["burnout_probability"] >= burnout_threshold).astype(int)

                st.success("✅ Predictions generated successfully.")
                st.write(df.head())

                # Download button
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="burnout_predictions.csv",
                    mime="text/csv"
                )

                # Visualizations
                st.markdown("---")
                st.markdown("### 📈 Burnout Overview")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Burnout Distribution**")
                    pie_data = df["burnout_prediction"].value_counts().rename(index={0: "No Burnout", 1: "Burnout"})
                    fig1, ax1 = plt.subplots()
                    ax1.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=["#4CAF50", "#F44336"])
                    ax1.axis('equal')
                    st.pyplot(fig1)

                with col2:
                    st.markdown("**Average Feature Values by Burnout**")
                    avg_features = df.groupby("burnout_prediction")[required_cols].mean()
                    avg_features.index = ["No Burnout", "Burnout"]
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    avg_features.T.plot(kind="bar", ax=ax2, colormap="coolwarm")
                    ax2.set_ylabel("Average Value")
                    ax2.set_title("Feature Comparison")
                    st.pyplot(fig2)

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ------------------------- SIDEBAR -------------------------
st.sidebar.markdown("### ℹ️ About This App")
st.sidebar.write("""
This tool helps individuals and organizations predict burnout risk 
based on workplace activity patterns.

**Trained on synthetic data**, this model estimates risk based on:
- Workload intensity
- Communication patterns
- Task performance
- Emotional sentiment
""")
