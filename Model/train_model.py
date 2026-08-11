"""
Trains a readmission risk model on the UCI Diabetic Data dataset
and saves it to model.pkl for use by the Streamlit app.

Run with: python model/train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diabetic_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Features to use — adjust based on what you find during EDA
FEATURE_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "age",
    "insulin",
    "change",
    "diabetesMed",
]
TARGET_COL = "readmitted"


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)

    # Binary target: readmitted at all (<30 or >30) vs not readmitted
    df["target"] = (df[TARGET_COL] != "NO").astype(int)

    df = df[FEATURE_COLS + ["target"]].dropna()
    return df


def encode_categoricals(df):
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Download it and place it there first.")
        return

    df = load_and_clean(DATA_PATH)
    df, encoders = encode_categoricals(df)

    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    joblib.dump(
        {
            "model": model,
            "encoders": encoders,
            "features": FEATURE_COLS,
        },
        MODEL_PATH,
    )
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
