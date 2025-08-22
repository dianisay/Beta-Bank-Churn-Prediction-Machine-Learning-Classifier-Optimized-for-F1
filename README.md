# 🏦 Beta Bank Churn Prediction — Machine Learning Classifier Optimized for F1

## 📌 Introduction
Build a supervised ML classifier to **predict customer churn (Exited)** and support retention strategies.  
Primary KPI: **F1 ≥ 0.59** on the test set. Also report **AUC‑ROC** and compare it to F1 to assess ranking quality vs. classification balance.

---

## 🎯 Business goals & metrics
- **Objective:** Predict which customers will churn to enable timely retention actions.
- **Primary metric:** F1 on the positive class (Exited=1) with a target of **≥ 0.59**.
- **Secondary metric:** AUC‑ROC for ranking performance; report both and discuss trade‑offs.
- **Operational use:** Threshold tuning to balance recall (catch churners) vs. precision (avoid false alarms).

---

## 📂 Dataset
- **Source file:** `datasets/Churn.csv`
- **Features:**
  - `RowNumber`, `CustomerId`, `Surname`
  - `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`
  - `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
- **Target:**
  - `Exited` — churn label (1=yes; 0=no)

---

## 🧭 Methodology

### 1) Data preparation
- **Loading & audit:** Check dtypes, missing values, duplicates, outliers.
- **Feature handling:** Encode categoricals (`Geography`, `Gender`), optional scaling for linear/distance models.
- **Leakage guard:** Exclude identifiers (`RowNumber`, `CustomerId`, `Surname`) from features.

### 2) Splits & baselines
- **Stratified split:** Train/Validation/Test (e.g., 60/20/20) with fixed seeds.
- **Baseline:** Dummy and untuned models to establish F1/AUC‑ROC floors.

### 3) Class imbalance strategies (use at least two)
- **Class weights:** `class_weight='balanced'` (LogReg, Tree/Forest, etc.).
- **Resampling:** SMOTE (oversampling) and/or RandomUnderSampler; apply **within CV** to avoid leakage.
- **Threshold optimization:** Choose decision threshold on validation to maximize F1.

### 4) Modeling & tuning
- **Candidates:** Logistic Regression, Random Forest, Gradient Boosting (e.g., XGBoost/LightGBM if allowed), Balanced variants.
- **Search:** Grid/Random search with stratified CV on Train; select by **validation F1**.
- **Calibration (optional):** Platt/Isotonic if probability quality matters for cost‑based decisions.

### 5) Evaluation & reporting
- **Final test:** Report F1 (primary) and AUC‑ROC (secondary) on the hold‑out test set.
- **Diagnostics:** Confusion matrix, precision/recall, PR and ROC curves.
- **Interpretability:** Feature importance (tree/boosting) or coefficients (logistic).

---

## 📊 Key analyses & visuals
- **EDA:** Distributions and correlations (e.g., `Age`, `Balance`, `IsActiveMember` vs `Exited`).
- **Imbalance:** Class distribution bar + PR curve (more informative than ROC under imbalance).
- **Model comparison:** Validation F1 by model/strategy (weights, SMOTE, threshold).
- **Final quality:** Test confusion matrix; PR/ROC curves; top features bar chart.

---

## ✅ Results summary
- **Best model:** <model_name> with <strategy> achieved **F1 = <value>** on test (target ≥ 0.59).
- **AUC‑ROC:** <value>; compare behavior vs F1 (e.g., high AUC but modest F1 under threshold).
- **Insights:** Drivers of churn (e.g., low activity, multiple products interactions, geography effects).
- **Ops note:** Recommended threshold = <t*> for retention campaigns prioritizing recall.

---

## 🧰 Tech stack
- **Python:** pandas, numpy, scikit‑learn, matplotlib, seaborn
- **Imbalance:** imbalanced‑learn (SMOTE/undersampling)
- **Utilities:** joblib (model persistence)

---

## 🚀 Repro steps
1. **Install:** `pip install -r requirements.txt`
2. **Data:** Place `Churn.csv` in `datasets/`
3. **Run:** Notebook `megaline_churn_classifier.ipynb` or `python src/train.py`
4. **Outputs:** Metrics → `reports/`, figures → `figures/`, model → `models/`

---

## 📈 Sample figures to include
- **Class balance plot**
- **Validation F1 by model/strategy**
- **Test confusion matrix**
- **ROC & PR curves**
- **Feature importance / coefficients**

---

## 🤝 Contact
Created by **Diana <Last Name>**  
🔗 [LinkedIn](https://linkedin.com/in/tuusuario) · 🌐 [Portfolio](https://tuportfolio.com)
