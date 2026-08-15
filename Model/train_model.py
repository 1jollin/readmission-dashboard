"""
Trains a readmission risk model on the UCI Diabetic Data dataset
and saves it to model.pkl for use by the Streamlit app.

Improvements over the baseline version:
- Removes a known data-leakage issue: patients discharged to hospice or
  who died in hospital cannot be "readmitted" in a meaningful sense, and
  including them just adds noise.
- Adds more predictive features (admission/discharge/source type, race,
  gender, lab results).
- Compares Random Forest, Gradient Boosting, and Logistic Regression,
  and keeps whichever scores best on ROC AUC.

Run with: python model/train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diabetic_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.pkl")

# Discharge disposition codes that mean the patient died or went to
# hospice — these can't be meaningfully "readmitted", so we drop them
# to remove a known source of noise/leakage in this dataset.
EXCLUDE_DISCHARGE_CODES = {11, 13, 14, 19, 20, 21}

# Features to use
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
    "race",
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "change",
    "diabetesMed",
]
TARGET_COL = "readmitted"


def load_and_clean(path):
    # keep_default_na=False stops pandas from silently converting the literal
    # text "None" (meaning "not tested" in max_glu_serum/A1Cresult) into NaN.
    # Only "?" is treated as missing, matching how this dataset marks missing values.
    df = pd.read_csv(path, keep_default_na=False, na_values=["?"])

    # Remove hospice/expired discharges (data leakage fix)
    df = df[~df["discharge_disposition_id"].isin(EXCLUDE_DISCHARGE_CODES)]

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
    print(f"Rows after cleaning (hospice/expired removed): {len(df):,}")

    df, encoders = encode_categoricals(df)

    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            random_state=42, class_weight="balanced", n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
        ),
    }

    best_name, best_model, best_auc = None, None, -1
    print("\nComparing models on held-out test set:")
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        print(f"  {name}: ROC AUC = {auc:.4f}")
        if auc > best_auc:
            best_name, best_model, best_auc = name, clf, auc

    print(f"\nBest model: {best_name} (ROC AUC = {best_auc:.4f})")
    model = best_model

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    joblib.dump(
        {
            "model": model,
            "model_name": best_name,
            "encoders": encoders,
            "features": FEATURE_COLS,
        },
        MODEL_PATH,
    )
    print(f"\nModel saved to {MODEL_PATH}")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        # Logistic Regression: use absolute coefficient magnitude instead
        importances = np.abs(model.coef_[0])

    joblib.dump(
        {
            "y_test": y_test.to_numpy(),
            "preds": preds,
            "probs": probs,
            "accuracy": accuracy_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
            "confusion_matrix": confusion_matrix(y_test, preds),
            "feature_importance": list(zip(FEATURE_COLS, importances)),
            "model_name": best_name,
        },
        METRICS_PATH,
    )
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
