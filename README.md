# 🛡️ Risk Alert Classifier

### Predicting High-Risk Customer Behavior for a Digital Banking Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-Resampling-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-Academic%20Project-lightgrey)

---

## 📌 Project Overview

This project implements an **early-warning classification system** for a digital banking platform to identify **high-risk customers** who are likely to default on payments or engage in fraudulent behavior.

The dataset is **highly imbalanced** (~88% Low Risk vs ~12% High Risk), which makes this a realistic reflection of real-world fraud/credit-risk problems, where the class you care about most is the rarest one. The project builds a complete, end-to-end classification pipeline — from raw data to a tuned, business-justified final model.

**Type of problem:** Supervised Learning → Binary Classification
**Business goal:** Minimize false negatives (missed high-risk customers) while keeping the model interpretable and production-ready.

---

## 🎯 Objective

- Design, evaluate, and optimize a classification system that predicts high-risk customer behavior.
- Implement multiple classification algorithms (Logistic Regression, Decision Tree, Random Forest).
- Evaluate models using advanced classification metrics (Precision, Recall, F1, AUC-ROC) — not just accuracy.
- Handle severe class imbalance using Under-Sampling, Over-Sampling, SMOTE, and ADASYN.
- Improve performance using RandomizedSearchCV and GridSearchCV hyperparameter tuning.

---

## 🗂️ Repository Structure

```
Risk-alert-classifier/
│
├── dataset/
│   └── Risk_Alert_Classifier_Dataset_4600.csv     # Raw dataset (4,600 customer records)
│
├── Risk_Alert_Classifier.ipynb                    # Main Jupyter Notebook (full pipeline)
│
└── README.md                                      # Project documentation (this file)
```

---

## 📊 Dataset Description

The dataset contains **4,600 customer records** with **19 columns**, combining demographic, behavioral, and transactional information.

| Category | Features |
|---|---|
| **Demographic** | `age`, `gender`, `region`, `employment_type`, `annual_income_inr` |
| **Credit Behavior** | `credit_score`, `credit_utilization_ratio`, `missed_payments_12m`, `avg_late_payment_days`, `debt_balance_inr` |
| **Transaction Activity** | `monthly_transaction_count`, `monthly_spend_inr`, `cash_advance_count_6m`, `last_transaction_date` |
| **Risk/Security Signals** | `complaints_last_6m`, `failed_login_attempts_3m`, `account_tenure_months` |
| **Identifier** | `customer_id` |
| **Target Variable** | `risk_status` → `0` = Low Risk, `1` = High Risk |

**Class Distribution:** ~88% Low Risk vs ~12% High Risk → **significant class imbalance**, directly motivating Part D of this project.

**Missing Values:** Present in both numeric (`age`, `annual_income_inr`, `credit_score`, `credit_utilization_ratio`, `monthly_spend_inr`) and categorical (`region`, `employment_type`) columns — handled via **KNN Imputation** (numeric) and **mode imputation** (categorical).

---

## 🧠 Methodology / Project Workflow

The notebook is structured into 8 parts, mirroring the full data science lifecycle:

| Part | Stage | What's Covered |
|---|---|---|
| **A** | Conceptual Understanding | Theory: Logistic Regression, classification metrics, Type-I/II errors, Precision/Recall/F1/TPR/FPR, AUC-ROC, imbalance problems |
| **B** | Data Preparation | EDA, feature/target identification, stratified train-test split, KNN Imputation |
| **C** | Baseline Model | Logistic Regression + Confusion Matrix + Accuracy/Precision/Recall/F1 |
| **D** | Imbalance Handling | Under-Sampling, Over-Sampling, SMOTE, ADASYN — compared on Recall/F1/AUC |
| **E** | Tree-Based Models | Decision Tree vs Random Forest, overfitting analysis (train vs test gap) |
| **F** | Hyperparameter Tuning | RandomizedSearchCV → GridSearchCV fine-tuning |
| **G** | Model Evaluation | ROC Curve comparison across all models, AUC-ROC ranking, final model selection |
| **H** | Reporting | Business interpretation, final recommendations |

### Pipeline Flow

```
Raw Data → Missing Value Treatment (KNN Imputer) → Encoding → Train-Test Split (stratified)
       → Feature Scaling → Baseline Logistic Regression
       → Imbalance Handling (SMOTE selected) → Decision Tree / Random Forest
       → Hyperparameter Tuning (RandomizedSearchCV + GridSearchCV)
       → ROC-AUC Evaluation → Final Model Selection → Business Report
```

---

## ⚙️ Tech Stack

| Category | Tools/Libraries |
|---|---|
| **Language** | Python 3.11 |
| **Data Handling** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Machine Learning** | scikit-learn |
| **Imbalanced Data** | imbalanced-learn (SMOTE, ADASYN, RandomUnderSampler, RandomOverSampler) |
| **Environment** | Jupyter Notebook |

---

## 📈 Key Results

| Model | Recall (High Risk) | F1-Score | AUC-ROC |
|---|---|---|---|
| Logistic Regression (Baseline) | High | High | High |
| Logistic Regression + SMOTE | Improved | Improved | Improved |
| Decision Tree (Tuned) | Strong | Strong | Strong |
| **Random Forest (Tuned) — Final Model** | **Best** | **Best** | **Best** |

> 📌 *Exact metric values, confusion matrices, and ROC plots are available inside the notebook (`Risk_Alert_Classifier.ipynb`) with cell-by-cell interpretations.*

**Top predictive features (via Random Forest feature importance):**
1. `avg_late_payment_days`
2. `missed_payments_12m`
3. `credit_score`
4. `credit_utilization_ratio`
5. `cash_advance_count_6m`

These align well with real-world credit risk intuition — payment delinquency and credit utilization are classic early indicators of financial distress.

### ✅ Final Model Selected: **Tuned Random Forest Classifier**

Chosen because it delivers the best balance of **Recall, F1-Score, and AUC-ROC**, and generalizes better than a single Decision Tree (smaller train-test accuracy gap → less overfitting). Since the business priority is **minimizing false negatives** (missed risky customers), high recall on the minority class was weighted heavily in model selection.

---

## 💼 Business Interpretation

| Error Type | Meaning | Business Impact |
|---|---|---|
| **False Positive** (Type-I) | Low-risk customer flagged as High Risk | Customer friction (extra verification), but no direct financial loss |
| **False Negative** (Type-II) | High-risk customer missed and labeled Low Risk | **Costlier** — a genuinely risky/fraudulent customer goes undetected |

Given this, the model is tuned and selected to **favor recall over raw accuracy**, since catching a risky customer (even at the cost of a few extra false alarms) is far more valuable to the bank than optimizing overall accuracy alone.

---

## 🚀 How to Run This Project

### 1. Clone the repository
```bash
git clone https://github.com/hetttk/Risk-alert-classifier.git
cd Risk-alert-classifier
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### 3. Launch the notebook
```bash
jupyter notebook Risk_Alert_Classifier.ipynb
```

### 4. Run all cells
`Kernel → Restart & Run All` — the notebook is designed to run top-to-bottom with no errors, provided the dataset CSV is in the same directory (or update the file path in the data-loading cell to point to `dataset/Risk_Alert_Classifier_Dataset_4600.csv`).

---

## 🔮 Future Improvements

- Feature engineering from `last_transaction_date` (e.g., recency-based features)
- Experiment with Gradient Boosting models (XGBoost, LightGBM, CatBoost)
- Cost-sensitive learning with explicit misclassification cost weighting
- Threshold tuning (beyond default 0.5) based on business risk appetite
- Periodic model retraining pipeline to adapt to evolving risk patterns

---

## 👤 Author

**[het k]**

⭐ *If you found this project useful, consider giving it a star!*
