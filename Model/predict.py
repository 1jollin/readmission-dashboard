"""
Shared prediction helper — imported by the Streamlit pages.
Loads the trained model bundle and exposes a simple predict() function.
"""

import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

_bundle = None


def _load_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run train_model.py first."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_risk(patient_dict):
    """
    patient_dict: dict of {feature_name: value} matching FEATURE_COLS
    Returns: (risk_probability, risk_tier)
    """
    bundle = _load_bundle()
    model = bundle["model"]
    encoders = bundle["encoders"]
    features = bundle["features"]

    row = pd.DataFrame([patient_dict])[features]

    # apply saved encoders to categorical columns
    for col, le in encoders.items():
        if col in row.columns:
            row[col] = le.transform(row[col].astype(str))

    prob = model.predict_proba(row)[0, 1]

    if prob < 0.33:
        tier = "Low"
    elif prob < 0.66:
        tier = "Medium"
    else:
        tier = "High"

    return prob, tier


def get_feature_importance():
    bundle = _load_bundle()
    model = bundle["model"]
    features = bundle["features"]
    importances = model.feature_importances_
    return sorted(zip(features, importances), key=lambda x: -x[1])
