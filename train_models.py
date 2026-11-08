"""
train_models.py
---------------
Trains 5 classification models on the CDC Diabetes Health Indicators dataset,
evaluates them, and saves everything the Streamlit app needs.

Run from the project folder:
    python model/train_models.py

What it does (step by step):
  1. Loads full_data.csv (all 999 rows we have).
  2. Splits it: 80% for training, 20% kept aside as test data (test_data.csv).
     The models NEVER see test_data.csv while training.
  3. Trains 5 models: Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM.
  4. Calculates Accuracy, AUC, Precision, Recall, F1, MCC on the test data.
  5. Saves each trained model as a .pkl file inside model/
  6. Saves the metrics table to model/metrics_summary.csv
  7. Writes requirements.txt with the exact library versions used.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

# ---------------------------------------------------------------- settings
TARGET = "Diabetes_012"   # 0 = no diabetes, 1 = prediabetes, 2 = diabetes
RANDOM_STATE = 42
TEST_SIZE = 0.20

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project folder
MODEL_DIR = os.path.join(ROOT, "model")


def build_models():
    """Return a dict of model_name -> sklearn Pipeline.
    Scaling is put INSIDE each pipeline, so the saved .pkl file already
    contains the preprocessing step (no separate scaler file needed)."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000,
                                       random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                           
                                           random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance")),
        ]),
        "Naive Bayes": Pipeline([
            ("clf", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True,
                         random_state=RANDOM_STATE)),
        ]),
    }


def evaluate(model, X, y):
    """Compute the 6 required metrics. Weighted averages are used because
    this is a multi-class problem with imbalanced classes."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob, multi_class="ovr", average="weighted",
                             labels=model.classes_),
        "Precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def file_name(model_name):
    return model_name.lower().replace(" ", "_") + ".pkl"


def main():
    # 1. Load data
    df = pd.read_csv(os.path.join(ROOT, "full_data.csv"))
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 2. Train / test split (stratified keeps class proportions the same)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(ROOT, "train_data.csv"), index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)} (saved as test_data.csv)")

    # 3-5. Train, evaluate, save
    os.makedirs(MODEL_DIR, exist_ok=True)
    rows = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        scores = evaluate(model, X_test, y_test)
        rows.append({"ML Model Name": name, **scores})
        joblib.dump(model, os.path.join(MODEL_DIR, file_name(name)))
        print(f"  trained + saved: {name}")

    # Save feature order so the app can check uploaded CSVs
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_columns.pkl"))

    # 6. Metrics table
    results = pd.DataFrame(rows).round(4)
    results.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"), index=False)
    print("\n=== Evaluation on test_data.csv ===")
    print(results.to_string(index=False))

    # 7. requirements.txt with exact versions (avoids deployment errors)
    import matplotlib, seaborn
    reqs = [
        "streamlit",
        f"scikit-learn=={sklearn.__version__}",
        f"numpy=={np.__version__}",
        f"pandas=={pd.__version__}",
        f"joblib=={joblib.__version__}",
        f"matplotlib=={matplotlib.__version__}",
        f"seaborn=={seaborn.__version__}",
    ]
    with open(os.path.join(ROOT, "requirements.txt"), "w") as f:
        f.write("\n".join(reqs) + "\n")
    print(f"\nrequirements.txt written (Python {sys.version.split()[0]})")


if __name__ == "__main__":
    main()
