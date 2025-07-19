import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Cache model loading for performance
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("burnout_model.json")  # Saved model file
    return model

model = load_model()

st.title("🧠 Burnout Detection System")
st.write("Upload employee data CSV for burnout risk prediction.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
threshold = st.slider("Select classification threshold", 0.0, 1.0, 0.5, 0.01)

expected_features = ["satisfaction", "hours_worked", "num_projects", "remote_work", "team_conflict"]

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Input Data Preview")
    st.dataframe(df.head())

    # Check if label column exists for evaluation
    has_label = "burnout" in df.columns

    # Filter only expected features for prediction
    try:
        X = df[expected_features]
    except KeyError as e:
        st.error(f"Input CSV is missing one or more required columns: {expected_features}")
        st.stop()

    if has_label:
        y = df["burnout"]

    # For now, no scaling - just convert to numpy
    X_input = X.values

    # Predict probabilities and apply threshold
    y_proba = model.predict_proba(X_input)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Show prediction results
    st.subheader("Prediction Results")
    results_df = df.copy()
    results_df["Burnout Probability"] = y_proba
    results_df["Prediction (1=Burnout)"] = y_pred
    st.dataframe(results_df)

    if has_label:
        st.subheader("Model Evaluation")

        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        st.markdown("**Classification Report**")
        st.dataframe(report_df)

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Built with XGBoost + Streamlit")
