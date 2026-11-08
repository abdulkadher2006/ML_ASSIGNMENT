"""
app.py - Diabetes Risk Classifier (Streamlit app)

Upload test_data.csv -> choose a model -> see metrics, confusion matrix
and classification report. Models are loaded from the model/ folder;
nothing is retrained here.

Run locally:  streamlit run app.py
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

TARGET = "Diabetes_012"
CLASS_NAMES = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}

st.set_page_config(page_title="Diabetes Risk Classifier", page_icon="🩺", layout="wide")


# ------------------------------------------------------------- helpers
@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


@st.cache_resource
def load_feature_columns():
    return joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    try:
        auc = roc_auc_score(y, y_prob, multi_class="ovr", average="weighted",
                            labels=model.classes_)
    except ValueError:          # e.g. uploaded file has only one class
        auc = float("nan")
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred


# ------------------------------------------------------------- sidebar
st.sidebar.title("🩺 Controls")
uploaded = st.sidebar.file_uploader("1. Upload test data (CSV)", type=["csv"])
model_name = st.sidebar.selectbox("2. Choose a model", list(MODEL_FILES.keys()))
st.sidebar.markdown("---")
st.sidebar.caption("Dataset: CDC Diabetes Health Indicators (BRFSS 2015)\n\n"
                   "Target: 0 = No diabetes, 1 = Prediabetes, 2 = Diabetes")

# ------------------------------------------------------------- header
st.title("Diabetes Risk Classifier")
st.write("Compare five pre-trained machine learning models on health-survey data. "
         "Upload the test CSV, pick a model, and the app evaluates it — "
         "the models are **not retrained** here.")

if uploaded is None:
    st.info("👈 Upload **test_data.csv** from the sidebar to begin.")
    summary_path = os.path.join(MODEL_DIR, "metrics_summary.csv")
    if os.path.exists(summary_path):
        st.subheader("Model comparison (from training run)")
        st.dataframe(pd.read_csv(summary_path), width="stretch", hide_index=True)
    st.stop()

# ------------------------------------------------------------- validate upload
df = pd.read_csv(uploaded)
features = load_feature_columns()
missing = [c for c in features + [TARGET] if c not in df.columns]
if missing:
    st.error(f"The uploaded file is missing these columns: {', '.join(missing)}")
    st.stop()

X = df[features]
y = df[TARGET]

with st.expander(f"Preview uploaded data ({df.shape[0]} rows, {df.shape[1]} columns)"):
    st.dataframe(df.head(20), width="stretch")
    st.write("Class counts:", y.map(CLASS_NAMES).value_counts().to_dict())

# ------------------------------------------------------------- selected model
model = load_model(model_name)
metrics, y_pred = compute_metrics(model, X, y)

st.subheader(f"Results: {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, "n/a" if np.isnan(v) else f"{v:.3f}")

left, right = st.columns(2)
labels = sorted(model.classes_)
names = [CLASS_NAMES.get(int(l), str(l)) for l in labels]

with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.markdown("**Classification report**")
    report = classification_report(y, y_pred, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

# ------------------------------------------------------------- all models
st.markdown("---")
st.subheader("All five models on the uploaded data")
rows = []
for name in MODEL_FILES:
    m, _ = compute_metrics(load_model(name), X, y)
    rows.append({"Model": name, **m})
all_df = pd.DataFrame(rows).set_index("Model").round(4)
st.dataframe(all_df.style.format(precision=4).highlight_max(axis=0, color="#c6efce"),
             width="stretch")
st.bar_chart(all_df[["Accuracy", "AUC", "F1", "MCC"]])
