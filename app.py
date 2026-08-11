"""
Hospital Readmission Risk Predictor + Care Management Dashboard
Main entry point (Home / Overview page)

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Readmission Risk Dashboard",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Hospital Readmission Risk Predictor")
st.caption("Care Management Dashboard — built on the UCI Diabetes 130-US Hospitals dataset (1999–2008)")

DATA_PATH = os.path.join("data", "diabetic_data.csv")

@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df

df = load_data(DATA_PATH)

if df is None:
    st.warning(
        f"Dataset not found at `{DATA_PATH}`. "
        "Download the Diabetes 130-US Hospitals dataset from the UCI ML Repository "
        "and place it there as `diabetic_data.csv`."
    )
else:
    st.success(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ---- Key metrics ----
    col1, col2, col3, col4 = st.columns(4)

    total_patients = df.shape[0]
    if "readmitted" in df.columns:
        readmit_rate = (df["readmitted"] != "NO").mean() * 100
    else:
        readmit_rate = None

    avg_stay = df["time_in_hospital"].mean() if "time_in_hospital" in df.columns else None
    avg_meds = df["num_medications"].mean() if "num_medications" in df.columns else None

    col1.metric("Total Patients", f"{total_patients:,}")
    col2.metric("Readmission Rate", f"{readmit_rate:.1f}%" if readmit_rate is not None else "N/A")
    col3.metric("Avg. Length of Stay", f"{avg_stay:.1f} days" if avg_stay is not None else "N/A")
    col4.metric("Avg. Medications", f"{avg_meds:.1f}" if avg_meds is not None else "N/A")

    st.divider()
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

st.divider()
st.markdown(
    """
    ### Navigation
    Use the sidebar to explore:
    - **Data Explorer** — distributions, correlations, feature breakdowns
    - **Risk Prediction** — enter patient details, get a readmission risk score
    - **Care Dashboard** — batch view of flagged high-risk patients
    - **Model Performance** — accuracy, ROC curve, confusion matrix
    """
)
