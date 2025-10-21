# 🧠 Clinical Trial Outcome Prediction  
### End-to-End Machine Learning Pipeline using AACT (ClinicalTrials.gov) Data  

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

### 📄 Short Overview  
This repository presents an **end-to-end machine learning workflow** to predict whether a **clinical trial** is likely to **succeed or fail**, based purely on its *design characteristics*.  

Built from the **AACT (ClinicalTrials.gov)** open registry, the project covers:  
- Complete **data pipeline** — extraction, cleaning, and feature engineering  
- **Exploratory Data Analysis (EDA)** — univariate, bivariate, and multivariate insights  
- **Modeling** — Logistic Regression (SAGA) & XGBoost with optimized thresholds  
- **Interpretability** — SHAP-based global & local explanations  
- **Deployment** — Streamlit app for real-time design feasibility prediction  

This work bridges **clinical research design** and **data science**, demonstrating how data-driven insights can guide early-phase feasibility assessments.

---

## 📘 Project Overview  

Clinical trials often fail due to design inefficiencies, under-enrollment, or unrealistic expectations of intervention success.  
This project aims to **predict the likelihood of trial completion (Success vs Failure)** by analyzing historical trial designs from the **AACT (ClinicalTrials.gov)** database.

Unlike many studies that focus on biological efficacy, this project evaluates the **design feasibility** of trials —  
how factors such as **phase, intervention type, conditions, allocation model, masking, sponsor type, enrollment size and duration** influence successful study completion.

### 🎯 Objectives  
- Identify which **design parameters** most influence trial success or failure.  
- Build an interpretable **machine-learning pipeline** to forecast trial outcomes.  
- Create a **Streamlit web app** for real-time prediction and feasibility validation.  
- Enable early detection of potentially **high-risk trial designs** before execution.  

### 🧱 Key Features  
- End-to-end reproducible pipeline: from raw AACT tables → cleaned dataset → EDA → modeling → deployment  
- Dual modeling approach: **Logistic Regression** and **XGBoost** with Youden’s J thresholds  
- Integrated **clinical validity checker** to flag impossible design combinations  
- Full **SHAP interpretability** for transparent model explanations  
- Lightweight **Streamlit interface** for research and educational use

---

## ⚙️ Workflow / Methodology  

The project follows a structured, modular pipeline designed to transform raw AACT data into a deployable prediction system.  
Each stage is implemented as an independent Jupyter notebook to maintain reproducibility and clarity.

| Stage | Notebook | Key Tasks |
|:------|:----------|:----------|
| **1. Data Cleaning & Integration** | `1_df_studies` → `8_df_merged` | Extract, clean, and merge core AACT tables (`studies`, `interventions`, `conditions`, `designs`, `eligibilities`, `sponsors`, etc.) |
| **2. Exploratory Data Analysis (EDA)** | `EDA_1`, `EDA_2`, `EDA_3` | Univariate → Bivariate → Multivariate analysis to identify influential design factors |
| **3. Feature Engineering & Preparation** | `9_df_final` | Handle missing values, apply log transformations, encode categorical variables, and prepare model-ready datasets |
| **4. Modeling & Evaluation** | `model_logreg`, `model_xgboost` | Train and optimize Logistic Regression and XGBoost models using Youden’s J threshold selection |
| **5. Interpretability (Explainability)** | `SHAP` | Global & local SHAP plots highlighting each feature’s contribution to success probability |
| **6. Deployment** | `/app/app.py` | Streamlit web app integrating both models with an interactive user interface and validity checks |

---

### 🔄 End-to-End Flow  
**AACT Database → Data Cleaning → EDA (1 → 2 → 3) → Feature Engineering → Modeling → SHAP Interpretation → Streamlit App Deployment**
Each step feeds the next, ensuring a transparent and reproducible analysis pipeline from **raw data to real-time prediction**.

---

## 🧩 Data Pipeline (Summary)

The dataset used in this project was sourced from the **AACT (Aggregate Analysis of ClinicalTrials.gov)** database,  
which provides standardized information for over 430,000 registered clinical trials.

Six key tables were extracted using PostgreSQL and SQLAlchemy:
`studies`, `interventions`, `conditions`, `eligibilities`, `sponsors`, and `designs`.

Each table underwent domain-specific cleaning and transformation:
- Standardized missing values (e.g., `phase`, `enrollment`, `masking`, `allocation`)
- Applied log transformation to enrollment and duration
- Reassigned hybrid phases (e.g., *Phase 1/2*, *Phase 2/3*) based on enrollment size
- Grouped medical conditions and interventions into broader categories
- Simplified sponsor roles (Government, Industry, Other)
- Encoded trial design features (allocation, masking, model, purpose)  
- Added binary trial-level flags (`has_dmc`, `is_fda_regulated_drug`, `healthy_volunteers`, etc.)

The cleaned tables were merged sequentially using `nct_id` as a common key to create: **df_merged.csv**


Exploratory Data Analysis (EDA 1–3) was then performed to:
- Examine univariate, bivariate, and multivariate feature behavior  
- Identify redundant, weak, or highly correlated variables  
- Decide which categorical and numerical features to retain for modeling  

Following EDA, the final **model-ready datasets** were prepared:
- **One-hot encoded version** for Logistic Regression  
- **Grouped categorical version** for XGBoost  

All cleaned and processed outputs were organized as follows:
/data/processed/ – Merged and cleaned datasets
/results/ – EDA plots, statistical outputs, model metrics, SHAP visuals
/models/ – Trained pipelines and serialized feature lists
/app/ – Streamlit deployment application


*For detailed preprocessing and EDA steps, refer to notebooks `1_df_studies` through `EDA_3`.*

---

## 🤖 Modeling & Evaluation (Summary)

Two supervised machine-learning models were developed to predict **clinical-trial completion success (1) or failure (0)**  
using the cleaned and feature-engineered datasets created during the EDA phase.

### 🧱 Models Used
- **Logistic Regression** — interpretable baseline model using one-hot-encoded categorical features  
- **XGBoost** — gradient-boosted ensemble model using grouped categorical representations  

Both models were trained on the same balanced feature set containing:
- Phase, Intervention Type, Condition, Allocation, Masking, Model, Purpose, Sponsor  
- Log-scaled numerical features (`log_enrollment`, `log_duration`, `number_of_arms`, `intervention_count`)  
- Binary design flags (`has_dmc`, `is_fda_regulated_drug`, `healthy_volunteers`, etc.)

### ⚙️ Training Workflow
1. **Data Split:** 80 % training / 20 % testing, stratified by outcome  
2. **Hyperparameter Tuning:** Grid-search with 3-fold stratified cross-validation  
3. **Optimization Metric:** Youden’s J Index ( *Sensitivity + Specificity – 1* )  
4. **Threshold Selection:**  
   - Logistic Regression ≈ 0.42  
   - XGBoost ≈ 0.85  
   ensuring balanced detection of both successes and failures  

### 📊 Evaluation Metrics
Performance was assessed on the held-out test set using:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- Confusion Matrix for class-wise evaluation  
- Precision–Recall curves for threshold validation  

### 💡 Model Interpretability
Model explainability was performed using **SHAP (SHapley Additive exPlanations)**:
- **Global Interpretation:** Feature-importance summary and interaction plots identified the strongest predictors of trial success  
- **Local Interpretation:** Force plots and waterfall charts illustrated how each variable influenced individual predictions  

Key influential features included:  
`phase`, `intervention_type`, `allocation`, `masking`, `model`, `purpose`, and `sponsor_type`.  

### 📂 Output Artifacts
All model outputs and metrics are stored in:
## 🤖 Modeling & Evaluation (Summary)

Two supervised machine learning models were developed to predict **clinical trial success (1)** or **failure (0)**  
using the cleaned and feature-engineered datasets prepared after the EDA phase.

### 🧱 Models Used
- **Logistic Regression (SAGA + L2 regularization)** — interpretable baseline model trained on one-hot encoded categorical features  
- **XGBoost** — ensemble tree-based model trained on grouped categorical features for improved generalization  

Both models used the same harmonized feature set containing:
- Categorical design elements: Phase, Intervention Type, Condition, Allocation, Masking, Model, Purpose, Sponsor  
- Log-scaled numerical features: `log_enrollment`, `log_duration`, `number_of_arms`, `intervention_count`  
- Binary trial design flags: `has_dmc`, `is_fda_regulated_drug`, `healthy_volunteers`, etc.  

---

### ⚙️ Training Workflow
1. **Data Split:** 80 % training / 20 % testing (stratified by success/failure outcome)  
2. **Model Selection:**  
   - Logistic Regression: manually compared multiple solvers (LBFGS, SAGA) and regularization strengths (C = 0.01 → 3)  
   - XGBoost: evaluated variations in tree depth (6–8) and estimators (200–500) to balance accuracy and overfitting  
3. **Threshold Optimization:** Youden’s J Index (*Sensitivity + Specificity – 1*) was used to find the best cutoff for each model:  
   - Logistic Regression ≈ **0.42**  
   - XGBoost ≈ **0.85**  
   This ensured a fair balance between identifying both successful and failed trials.

---

### 📊 Evaluation Metrics
Model performance was assessed using:
- Accuracy, Precision, Recall, F1-score, and ROC-AUC  
- Confusion matrix to examine class-level predictions  
- Precision–Recall curve to visualize trade-offs across thresholds  

---

### 💡 Model Interpretability (SHAP Analysis)
To enhance explainability, **SHAP (SHapley Additive exPlanations)** was used for both global and local interpretation:  
- **Global:** Identified the strongest contributors to trial success — `phase`, `intervention_type`, `allocation`, `masking`, and `purpose`.  
- **Local:** Individual-level SHAP force and waterfall plots showed how specific features influenced each prediction.  

This allowed visualization of why similar trials could yield opposite outcomes despite shared parameters.

---

### 📂 Output Artifacts
All model-related files and evaluation results were saved under:

---

## 🤖 Modeling & Evaluation (Summary)

Two supervised machine learning models were developed to predict **clinical trial success (1)** or **failure (0)**  
using the cleaned and feature-engineered datasets finalized after EDA.

### 🧱 Models Used
- **Logistic Regression** — interpretable baseline model trained on one-hot encoded categorical features.  
- **XGBoost** — ensemble tree-based model trained on grouped categorical features for improved generalization and non-linear relationships.

Both models were trained on harmonized features, including:
- **Categorical:** Phase, Intervention Type, Condition, Allocation, Masking, Model, Purpose, Sponsor  
- **Numerical:** `log_enrollment`, `log_duration`, `number_of_arms`, `intervention_count`  
- **Binary Flags:** `has_dmc`, `is_fda_regulated_drug`, `healthy_volunteers`, `has_expanded_access`, `is_fda_regulated_device`.

---

### ⚙️ Training Workflow
1. **Data Split:** 80% training / 20% testing (stratified by success/failure outcome).  
2. **Model Configuration:**  
   - Multiple parameter sets were **experimentally compared** in earlier notebooks to identify stable, well-performing configurations.  
   - Final models used:
     - Logistic Regression → `solver = "lbfgs"`, `C = 1`, `penalty = "l2"`  
     - XGBoost → `n_estimators = 200`, `max_depth = 8`, `scale_pos_weight = 1`
3. **Threshold Optimization:**  
   - The **Youden’s J Index** (*Sensitivity + Specificity − 1*) was used to determine optimal cutoffs:  
     - Logistic Regression ≈ **0.42**  
     - XGBoost ≈ **0.85**  
   - These thresholds were later used in both the evaluation and Streamlit app deployment.

---

### 📊 Evaluation Metrics
Performance was measured on the held-out test data using:
- Accuracy, Precision, Recall, F1-score, ROC-AUC  
- Confusion matrix for class-level comparison  
- Precision–Recall curve to confirm threshold consistency  

---

### 💡 Model Interpretability (SHAP Analysis)
SHAP (SHapley Additive exPlanations) was applied to both models for interpretability:
- **Global Interpretation:** Ranked features by average SHAP value to identify strong predictors (e.g., `log_enrollment`, `log_duration`, `phase`, `sponsor`, `model`, `intervention_type`).  
- **Local Interpretation:** Used force and waterfall plots to visualize how each feature influenced individual trial predictions.

This dual-level explainability made the models both **transparent and scientifically interpretable**.

---

### 📂 Output Artifacts
All model files and results were saved under:
/models/ – Serialized pipelines (logreg_pipeline.pkl, xgb_pipeline.pkl)
and feature lists (X_cols_logreg.pkl, X_cols_xgb.pkl)
/results/ – Metrics tables, ROC & PR curves, SHAP visualizations
/app/ – Streamlit deployment integrating both models and Youden thresholds

---

### ✅ Outcome
- **Logistic Regression** offered interpretability and transparent coefficients, serving as a robust statistical baseline.  
- **XGBoost** achieved better recall and overall balance, identifying more realistic design patterns linked to successful outcomes.  
- The combination of both provided a complete perspective — **explainability** from Logistic Regression and **predictive strength** from XGBoost.

