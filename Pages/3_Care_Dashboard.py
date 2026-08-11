"""
Care Management Dashboard page — batch view of flagged high-risk patients
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))

st.set_page_config(page_title="Care Dashboard", page_icon="📋", layout="wide")
st.title("📋 Care Management Dashboard")
st.caption("Flagged patients ranked by readmission risk, with suggested follow-up actions.")

# ---- Try to import the real model; fall back to a fake one if not trained yet ----
USE_FAKE_MODEL = False
try:
    from predict import predict_risk
except Exception:
    USE_FAKE_MODEL = True

    def predict_risk(patient_dict):
        prob = random.uniform(0, 1)
        tier = "Low" if prob < 0.33 else "Medium" if prob < 0.66 else "High"
        return prob, tier

if USE_FAKE_MODEL:
    st.warning(
        "⚠️ No trained model found yet — showing randomized placeholder risk scores. "
        "Run `model/train_model.py` and refresh once the real model is ready."
    )

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diabetic_data.csv")


@st.cache_data
def load_sample_patients(n=50):
    """Load a sample of real patients if the dataset exists, else generate dummy rows."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df = df.replace("?", np.nan)
        cols = [
            "age", "time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_outpatient", "number_emergency",
            "number_inpatient", "number_diagnoses", "insulin", "change", "diabetesMed",
        ]
        df = df[cols].dropna().sample(n=min(n, len(df)), random_state=1).reset_index(drop=True)
        return df
    else:
        # dummy fallback data so the page still works without the dataset
        ages = ["[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"]
        rows = []
        for i in range(n):
            rows.append({
                "age": random.choice(ages),
                "time_in_hospital": random.randint(1, 14),
                "num_lab_procedures": random.randint(1, 100),
                "num_procedures": random.randint(0, 6),
                "num_medications": random.randint(1, 30),
                "number_outpatient": random.randint(0, 5),
                "number_emergency": random.randint(0, 3),
                "number_inpatient": random.randint(0, 4),
                "number_diagnoses": random.randint(1, 16),
                "insulin": random.choice(["No", "Steady", "Up", "Down"]),
                "change": random.choice(["No", "Ch"]),
                "diabetesMed": random.choice(["Yes", "No"]),
            })
        return pd.DataFrame(rows)


def recommend_action(tier, number_inpatient):
    if tier == "High":
        if number_inpatient >= 2:
            return "Urgent follow-up call within 7 days"
        return "Schedule follow-up within 7 days"
    elif tier == "Medium":
        return "Check-in within 30 days"
    else:
        return "Routine monitoring"


patients_df = load_sample_patients(50)

# ---- Run predictions for each patient ----
results = []
for _, row in patients_df.iterrows():
    patient_dict = row.to_dict()
    try:
        prob, tier = predict_risk(patient_dict)
    except Exception:
        prob, tier = 0.0, "Low"
    action = recommend_action(tier, row["number_inpatient"])
    results.append({
        "Patient ID": _ + 1,
        "Age": row["age"],
        "Time in Hospital": row["time_in_hospital"],
        "Num Medications": row["num_medications"],
        "Prior Inpatient Visits": row["number_inpatient"],
        "Risk Score": round(prob * 100, 1),
        "Risk Tier": tier,
        "Recommended Action": action,
    })

results_df = pd.DataFrame(results)

# ---- Sidebar filters ----
st.sidebar.header("Filters")
tier_filter = st.sidebar.multiselect(
    "Risk Tier", options=["Low", "Medium", "High"], default=["Low", "Medium", "High"]
)
age_filter = st.sidebar.multiselect(
    "Age Bracket", options=sorted(results_df["Age"].unique()), default=sorted(results_df["Age"].unique())
)

filtered_df = results_df[
    results_df["Risk Tier"].isin(tier_filter) & results_df["Age"].isin(age_filter)
].sort_values("Risk Score", ascending=False)

# ---- Summary metrics ----
c1, c2, c3 = st.columns(3)
c1.metric("Total Patients Shown", len(filtered_df))
c2.metric("High Risk", int((filtered_df["Risk Tier"] == "High").sum()))
c3.metric("Avg Risk Score", f"{filtered_df['Risk Score'].mean():.1f}%" if len(filtered_df) else "N/A")

st.divider()
st.subheader("Flagged Patients")
st.dataframe(filtered_df, use_container_width=True, hide_index=True)

# ---- Download button ----
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download flagged patients as CSV",
    data=csv,
    file_name="flagged_patients.csv",
    mime="text/csv",
)
