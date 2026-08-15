"""
Model Performance page — accuracy, ROC curve, confusion matrix, feature importance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.metrics import roc_curve

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("📈 Model Performance")
st.caption("How well the readmission risk model performs on unseen test data.")

METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "metrics.pkl")

if not os.path.exists(METRICS_PATH):
    st.warning(
        "⚠️ No metrics found yet. Run `python model/train_model.py` first — "
        "this generates `model/metrics.pkl` alongside the trained model."
    )
    st.stop()

metrics = joblib.load(METRICS_PATH)

y_test = metrics["y_test"]
preds = metrics["preds"]
probs = metrics["probs"]
accuracy = metrics["accuracy"]
roc_auc = metrics["roc_auc"]
cm = metrics["confusion_matrix"]
feature_importance = metrics["feature_importance"]

# ---- Top-level metrics ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{accuracy*100:.1f}%")
c2.metric("ROC AUC", f"{roc_auc:.3f}")

tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
c3.metric("Precision (readmitted)", f"{precision*100:.1f}%")
c4.metric("Recall (readmitted)", f"{recall*100:.1f}%")

st.divider()

# ---- Confusion matrix + ROC curve side by side ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")
    labels = ["Not Readmitted", "Readmitted"]
    fig = px.imshow(
        cm, text_auto=True,
        x=labels, y=labels,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"True Negatives: {tn} | False Positives: {fp} | "
        f"False Negatives: {fn} | True Positives: {tp}"
    )

with col2:
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Model (AUC = {roc_auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess",
                              line=dict(dash="dash", color="gray")))
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.5, y=0.05),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- Feature importance ----
st.subheader("Feature Importance")
imp_df = pd.DataFrame(feature_importance, columns=["Feature", "Importance"])
imp_df = imp_df.sort_values("Importance", ascending=True)
fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h")
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "These numbers come from the held-out test set (20% of the data, not used in training), "
    "so they reflect how the model is likely to perform on new patients."
)
