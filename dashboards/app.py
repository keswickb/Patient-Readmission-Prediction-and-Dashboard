import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Patient Readmission Prediction", page_icon="🏥", layout="centered")
st.title("🏥 Patient Readmission Prediction")

# --- Load artifacts saved from your notebooks ---
# Expect these files to exist:
#   ../models/pipeline_model.pkl  (the trained classifier)
#   ../models/scaler.pkl          (StandardScaler fitted on training features)
#   ../models/feature_cols.pkl    (list of training feature column names after get_dummies)
try:
    clf = joblib.load("../models/pipeline_model.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    feature_cols = joblib.load("../models/feature_cols.pkl")
except Exception as e:
    st.error(f"Could not load model/scaler/feature columns. Make sure they exist in ../models.\n\n{e}")
    st.stop()

# --- Sidebar inputs: these must match your RAW training columns (before get_dummies) ---
st.sidebar.header("Enter Patient Info")

age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=50)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
primary_diagnosis = st.sidebar.selectbox(
    "Primary Diagnosis",
    ["Diabetes", "Heart Failure", "Pneumonia", "COPD", "Other"]  # adjust to your data if needed
)
num_procedures = st.sidebar.number_input("Number of Procedures", min_value=0, max_value=20, value=1)
days_in_hospital = st.sidebar.number_input("Days in Hospital", min_value=0, max_value=60, value=3)
comorbidity_score = st.sidebar.number_input("Comorbidity Score", min_value=0, max_value=20, value=2)
discharge_to = st.sidebar.selectbox(
    "Discharge To",
    ["Home", "Skilled Nursing Facility", "Rehabilitation Facility", "Home Health Care", "Other"]  # adjust if needed
)

# --- Build a single-row DataFrame in the SAME raw schema used during training ---
raw_row = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "primary_diagnosis": primary_diagnosis,
    "num_procedures": num_procedures,
    "days_in_hospital": days_in_hospital,
    "comorbidity_score": comorbidity_score,
    "discharge_to": discharge_to
}])

st.subheader("Review Input")
st.write(raw_row)

# --- Preprocess exactly like training: get_dummies -> reindex -> scale -> predict ---
if st.button("Predict Readmission"):
    try:
        # 1) One-hot encode like training
        X = pd.get_dummies(raw_row, drop_first=True)

        # 2) Force the SAME columns (names + order) as during training
        #    Any missing training columns are filled with 0; extra columns are dropped.
        X = X.reindex(columns=feature_cols, fill_value=0)

        # 3) Scale using the saved scaler.
        #    Use .values to avoid sklearn's feature-name checks and keep order consistent.
        X_scaled = scaler.transform(X.values)

        # 4) Predict
        pred = clf.predict(X_scaled)[0]
        prob = float(clf.predict_proba(X_scaled)[:, 1][0])

        label = "Readmitted" if int(pred) == 1 else "Not Readmitted"
        st.success(f"🔮 Prediction: **{label}**")
        st.metric(label="Probability of Readmission", value=f"{prob*100:.2f}%")

        # Optional: show engineered features the model actually saw
        with st.expander("Show engineered feature vector the model used"):
            st.dataframe(pd.DataFrame(X, columns=feature_cols))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
