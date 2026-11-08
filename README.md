# Diabetes Risk Classification – ML Assignment 2

**Live app:** https://mlassignment-slfxwj4b8kewtsu8w8jjuj.streamlit.app
**GitHub repo:** https://github.com/abdulkadher2006/ML_ASSIGNMENT

---

## a. Problem Statement

Diabetes is one of the most common chronic diseases, and many people with prediabetes or diabetes do not know they have it. The goal of this project is to predict a person's diabetes status — **no diabetes, prediabetes, or diabetes** — from simple health-survey answers (blood pressure, cholesterol, BMI, lifestyle habits, general health, age, etc.).

This is a **multi-class classification** problem. Five machine learning models are trained and compared, and an interactive Streamlit web app lets anyone upload test data and evaluate each saved model.

## b. Dataset Description

- **Name:** CDC Diabetes Health Indicators (BRFSS 2015) — `diabetes_012_health_indicators`
- **Source:** Kaggle – https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset (also on UCI ML Repository, ID 891)
- **Rows used:** 999 instances
- **Features:** 19 input features + 1 target column (no missing values)
- **Train/test split:** 80% train (799 rows, `train_data.csv`) / 20% test (200 rows, `test_data.csv`), stratified by class

**Target – `Diabetes_012`**

| Value | Meaning | Count (all 999 rows) |
|---|---|---|
| 0 | No diabetes | 757 |
| 1 | Prediabetes | 21 |
| 2 | Diabetes | 221 |

The classes are **imbalanced** — prediabetes is very rare (about 2%).

**Features**

| Feature | Description |
|---|---|
| HighBP | High blood pressure (0/1) |
| HighChol | High cholesterol (0/1) |
| CholCheck | Cholesterol check in last 5 years (0/1) |
| BMI | Body Mass Index |
| Smoker | Smoked at least 100 cigarettes in life (0/1) |
| Stroke | Ever had a stroke (0/1) |
| HeartDiseaseorAttack | Coronary heart disease or heart attack (0/1) |
| PhysActivity | Physical activity in last 30 days (0/1) |
| Fruits | Eats fruit daily (0/1) |
| Veggies | Eats vegetables daily (0/1) |
| HvyAlcoholConsump | Heavy alcohol drinker (0/1) |
| AnyHealthcare | Has health coverage (0/1) |
| NoDocbcCost | Could not see doctor due to cost (0/1) |
| GenHlth | General health, 1 (excellent) – 5 (poor) |
| MentHlth | Days of poor mental health in last 30 days |
| PhysHlth | Days of poor physical health in last 30 days |
| DiffWalk | Difficulty walking/climbing stairs (0/1) |
| Sex | 0 = female, 1 = male |
| Age | Age group, 1 (18–24) – 13 (80+) |

## c. GitHub Repository Link

https://github.com/abdulkadher2006/ML_ASSIGNMENT

```
ML_ASSIGNMENT/
│-- app.py                  # Streamlit web app
│-- requirements.txt        # Python libraries needed
│-- README.md               # This file
│-- test_data.csv           # 20% held-out test data (upload this in the app)
│-- train_data.csv          # 80% training data
│-- full_data.csv           # All 999 rows (input to training script)
│-- model/
    │-- train_models.py     # Training + evaluation source code
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- support_vector_machine.pkl
    │-- feature_columns.pkl # Feature order used by the models
    │-- metrics_summary.csv # Comparison table
```

Preprocessing (StandardScaler for Logistic Regression, KNN and SVM) is saved **inside** each model's `.pkl` pipeline, so no separate scaler file is needed.

## d. Models Used

1. Logistic Regression
2. Decision Tree Classifier (max_depth = 5)
3. K-Nearest Neighbors (k = 15, distance-weighted)
4. Naive Bayes (Gaussian)
5. Support Vector Machine (RBF kernel)

Metrics are computed on the 200-row test set. Because the problem is multi-class, Precision, Recall, F1 and AUC use **weighted averages** (AUC is one-vs-rest).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7600 | 0.7421 | 0.7080 | 0.7600 | 0.7120 | 0.1941 |
| Decision Tree | 0.7450 | 0.6781 | 0.6595 | 0.7450 | 0.6797 | 0.0848 |
| KNN | 0.7300 | 0.7474 | 0.6319 | 0.7300 | 0.6643 | 0.0239 |
| Naive Bayes | 0.2900 | 0.7297 | 0.7780 | 0.2900 | 0.2653 | 0.0899 |
| Support Vector Machine | 0.7550 | 0.6961 | 0.6570 | 0.7550 | 0.6722 | 0.0760 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall: highest accuracy (0.76), F1 (0.71) and MCC (0.19), with a strong AUC (0.74). It finds the most actual diabetes cases (18%) of all the models that mostly predict "no diabetes", so its predictions are the most balanced. A simple linear model works well here because risk rises steadily with features like BMI, high BP, age and poor general health. |
| Decision Tree | Good accuracy (0.745) but the lowest AUC (0.68) and a low MCC (0.08). It predicts "no diabetes" for almost everyone and catches only 9% of diabetes cases. With only ~800 training rows, a single tree gives coarse, unstable probability estimates. |
| KNN | Good AUC (0.75, the highest) but the lowest MCC (0.02): it ranks patients well by risk, yet its final labels are almost all "no diabetes" because nearby neighbours are mostly from the majority class. Accuracy (0.73) is close to just always guessing class 0 (0.76). |
| Naive Bayes | Lowest accuracy (0.29) and F1 (0.27). It labels most people as diabetic, so it catches 84% of real diabetes cases but misclassifies most healthy people. This happens because Gaussian NB assumes features are independent and normally distributed, which is not true for these 0/1 survey answers. Its AUC (0.73) is still reasonable, so its ranking of risk is useful even though its labels are not. |
| Support Vector Machine | Second-highest accuracy (0.755) but a low MCC (0.08) and recall of only 5% for diabetes. It almost always predicts "no diabetes", which looks good on accuracy only because that class is 76% of the data. |
| **Overall winner for this dataset** | **Logistic Regression** — best accuracy, F1 and MCC with a strong AUC. No model could detect the rare prediabetes class (only 4 test rows), and accuracy alone is misleading on this imbalanced data, which is why MCC and F1 matter more here. Collecting more data or class balancing would be the next step. |

## Streamlit App Features

- CSV upload of test data (sidebar)
- Model selection dropdown (5 models)
- Display of all six evaluation metrics
- Confusion matrix and classification report for the chosen model
- Side-by-side comparison of all five models on the uploaded file

The app only **loads** the saved models — it never retrains on the uploaded data.

## How to Run

```bash
pip install -r requirements.txt
python model/train_models.py   # retrains models and rewrites the .pkl files
streamlit run app.py
```
