import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_extras.app_logo import add_logo
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
from streamlit_extras.mention import mention

# Load model and scaler
model = xgb.Booster()
model.load_model("burnout_model.json")
scaler = joblib.load("scaler.pkl")

# Load threshold
burnout_threshold = 0.5
if os.path.exists("burnout_threshold.txt"):
    with open("burnout_threshold.txt", "r") as f:
        burnout_threshold = float(f.read().strip())

st.title("📊 Organization Burnout Analysis")
st.markdown("Upload a CSV of employees’ data to analyze workplace burnout trends.")

with st.expander("ℹ️ Required Columns Info"):
    st.markdown("""
    - `work_hours_per_week`
    - `after_hours_emails`
    - `negative_sentiment_score`
    - `meeting_count`
    - `task_completion_rate`
    """)

uploaded_file = st.file_uploader("📎 Upload Employee Data", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = [
            "work_hours_per_week", "after_hours_emails",
            "negative_sentiment_score", "meeting_count", "task_completion_rate"
        ]

        if not all(col in df.columns for col in required_cols):
            st.error("Missing required columns.")
        else:
            df = df[required_cols]
            X_scaled = scaler.transform(df)
            dtest = xgb.DMatrix(X_scaled)
            df["burnout_probability"] = model.predict(dtest)
            df["burnout_prediction"] = (df["burnout_probability"] >= burnout_threshold).astype(int)

            st.success("✅ Predictions generated.")
            st.dataframe(df.head(10), use_container_width=True)

            st.download_button("📥 Download CSV", data=df.to_csv(index=False).encode(),
                               file_name="burnout_predictions.csv", mime="text/csv")

            # Pie chart
            pie_data = df["burnout_prediction"].value_counts().rename(index={0: "No Burnout", 1: "Burnout"})
            fig1, ax1 = plt.subplots()
            ax1.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90,
                    colors=["#4CAF50", "#F44336"])
            ax1.axis('equal')
            st.subheader("🧭 Burnout Distribution")
            st.pyplot(fig1)

            # Feature comparison
            avg_features = df.groupby("burnout_prediction")[required_cols].mean()
            avg_features.index = ["No Burnout", "Burnout"]
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            avg_features.T.plot(kind="bar", ax=ax2, colormap="coolwarm")
            ax2.set_ylabel("Average Value")
            ax2.set_title("Feature Comparison")
            st.subheader("📊 Feature Breakdown")
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error: {e}")
