"""
Data Explorer page — distributions, correlations, and readmission breakdowns
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
st.title("📊 Data Explorer")
st.caption("Explore patterns in the dataset before trusting the model's predictions.")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diabetic_data.csv")


@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)
    df["readmitted_flag"] = (df["readmitted"] != "NO").astype(int)
    return df


df = load_data(DATA_PATH)

if df is None:
    st.warning(f"Dataset not found at `{DATA_PATH}`. Make sure `diabetic_data.csv` is in the `data/` folder.")
    st.stop()

# ---- Sidebar filters ----
st.sidebar.header("Filters")
age_options = sorted(df["age"].dropna().unique())
selected_ages = st.sidebar.multiselect("Age bracket", age_options, default=age_options)

gender_options = df["gender"].dropna().unique().tolist()
selected_genders = st.sidebar.multiselect("Gender", gender_options, default=gender_options)

filtered_df = df[df["age"].isin(selected_ages) & df["gender"].isin(selected_genders)]

st.caption(f"Showing {len(filtered_df):,} of {len(df):,} patients based on current filters.")

st.divider()

# ---- Row 1: Age distribution + readmission by age ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Count by Age Bracket")
    age_counts = filtered_df["age"].value_counts().reindex(age_options).fillna(0)
    fig = px.bar(
        x=age_counts.index, y=age_counts.values,
        labels={"x": "Age Bracket", "y": "Number of Patients"},
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Readmission Rate by Age Bracket")
    readmit_by_age = filtered_df.groupby("age")["readmitted_flag"].mean().reindex(age_options) * 100
    fig = px.bar(
        x=readmit_by_age.index, y=readmit_by_age.values,
        labels={"x": "Age Bracket", "y": "Readmission Rate (%)"},
        color=readmit_by_age.values,
        color_continuous_scale="Reds",
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- Row 2: Time in hospital + num medications distributions ----
col3, col4 = st.columns(2)

with col3:
    st.subheader("Distribution: Time in Hospital")
    fig = px.histogram(filtered_df, x="time_in_hospital", nbins=14,
                        labels={"time_in_hospital": "Days in Hospital"})
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Distribution: Number of Medications")
    fig = px.histogram(filtered_df, x="num_medications", nbins=30,
                        labels={"num_medications": "Number of Medications"})
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- Row 3: Readmission by admission type / prior inpatient visits ----
col5, col6 = st.columns(2)

with col5:
    st.subheader("Readmission Rate by Insulin Status")
    if "insulin" in filtered_df.columns:
        readmit_by_insulin = filtered_df.groupby("insulin")["readmitted_flag"].mean() * 100
        fig = px.bar(
            x=readmit_by_insulin.index, y=readmit_by_insulin.values,
            labels={"x": "Insulin", "y": "Readmission Rate (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)

with col6:
    st.subheader("Readmission Rate by Prior Inpatient Visits")
    inpatient_capped = filtered_df["number_inpatient"].clip(upper=5)
    readmit_by_inpatient = filtered_df.assign(inpatient_capped=inpatient_capped) \
        .groupby("inpatient_capped")["readmitted_flag"].mean() * 100
    fig = px.bar(
        x=readmit_by_inpatient.index, y=readmit_by_inpatient.values,
        labels={"x": "Prior Inpatient Visits (5+ capped)", "y": "Readmission Rate (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- Row 4: Correlation heatmap ----
st.subheader("Correlation Between Numeric Features")
numeric_cols = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "readmitted_flag",
]
numeric_cols = [c for c in numeric_cols if c in filtered_df.columns]
corr = filtered_df[numeric_cols].corr()

fig = px.imshow(
    corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    aspect="auto",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Tip: `number_inpatient` typically shows the strongest correlation with readmission — "
    "patients with more prior hospital stays are more likely to be readmitted."
)

st.divider()

# ---- Raw data table ----
st.subheader("Filtered Data Table")
st.dataframe(filtered_df.head(200), use_container_width=True)
