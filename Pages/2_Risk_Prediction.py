"""
Risk Prediction page — patient input form -> readmission risk score
"""

import streamlit as st
import sys
import os
import random

# allow importing from the model/ folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))

st.set_page_config(page_title="Risk Prediction", page_icon="🔮", layout="wide")
st.title("🔮 Patient Readmission Risk Prediction")
st.caption("Enter patient details to estimate readmission risk.")

# ---- Try to import the real model. If it isn't trained yet, fall back to a
# ---- fake predictor so the UI can still be built/tested independently. ----
USE_FAKE_MODEL = False
try:
    from predict import predict_risk, get_feature_importance
except Exception:
    USE_FAKE_MODEL = True

    def predict_risk(patient_dict):
        """Temporary stand-in until model/model.pkl exists.
        Remove this fallback once train_model.py has been run."""
        prob = random.uniform(0, 1)
        if prob < 0.33:
            tier = "Low"
        elif prob < 0.66:
            tier = "Medium"
        else:
            tier = "High"
        return prob, tier

    def get_feature_importance():
        return []

if USE_FAKE_MODEL:
    st.warning(
        "⚠️ No trained model found yet — showing randomized placeholder predictions. "
        "Run `model/train_model.py` and refresh once the real model is ready."
    )

AGE_BRACKETS = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]

st.subheader("Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.selectbox("Age bracket", AGE_BRACKETS, index=6)
    time_in_hospital = st.number_input("Time in hospital (days)", min_value=1, max_value=14, value=4)
    num_lab_procedures = st.number_input("Number of lab procedures", min_value=0, max_value=150, value=40)
    num_procedures = st.number_input("Number of procedures", min_value=0, max_value=10, value=1)

with col2:
    num_medications = st.number_input("Number of medications", min_value=0, max_value=100, value=15)
    number_outpatient = st.number_input("Prior outpatient visits", min_value=0, max_value=50, value=0)
    number_emergency = st.number_input("Prior emergency visits", min_value=0, max_value=50, value=0)
    number_inpatient = st.number_input("Prior inpatient visits", min_value=0, max_value=50, value=0)

with col3:
    number_diagnoses = st.number_input("Number of diagnoses", min_value=1, max_value=20, value=7)
    insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
    change = st.selectbox("Change in medication", ["No", "Ch"])
    diabetesMed = st.selectbox("On diabetes medication", ["Yes", "No"])

st.divider()

if st.button("Predict Readmission Risk", type="primary"):
    patient_dict = {
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "insulin": insulin,
        "change": change,
        "diabetesMed": diabetesMed,
    }

    try:
        prob, tier = predict_risk(patient_dict)

        st.subheader("Result")
        r1, r2 = st.columns(2)
        r1.metric("Readmission Probability", f"{prob*100:.1f}%")

        tier_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        r2.metric("Risk Tier", f"{tier_color.get(tier, '')} {tier}")

        st.progress(min(max(prob, 0.0), 1.0))

        if not USE_FAKE_MODEL:
            importances = get_feature_importance()
            if importances:
                st.subheader("Top Contributing Factors")
                import pandas as pd
                imp_df = pd.DataFrame(importances[:8], columns=["Feature", "Importance"])
                st.bar_chart(imp_df.set_index("Feature"))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
