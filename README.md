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

---

## 🌐 Streamlit App (Deployment Summary)

A Streamlit web application was built to make the model results interactive and interpretable.  
It integrates both **Logistic Regression** and **XGBoost** models with their **Youden’s J thresholds**
(LogReg ≈ 0.42, XGBoost ≈ 0.85).

### 🔹 Key Features
- **Interactive design input:** Users can select trial parameters such as phase, intervention type, conditions, allocation, masking, sponsor etc.  
- **Automatic validity checks:** Flags logically invalid configurations (e.g., randomized single-group trials).  
- **Model interpretation section:** Explains threshold rationale, limitations, and ethical disclaimer in a collapsible block.  
- **Local and cloud ready:** Can be run via  
  ```bash
  streamlit run app/app.py

or deployed directly to Streamlit Cloud.

All supporting files (app.py, requirements.txt, README_app.txt, and model artifacts) are located in the /app directory.

Purpose: Provide a transparent, user-friendly interface to explore how design features affect predicted trial outcomes.

---

## 📊 Results & Key Insights (Summary)

### 🧩 Exploratory Data Analysis
- Univariate, bivariate, and multivariate analyses revealed clear design-level patterns in clinical trial outcomes.  
- Higher success rates were observed in **Phase 3**, **randomized**, **parallel**, and **double-masked** trials.  
- Behavioral and open-label single-group studies showed comparatively lower completion rates.  
- EDA results guided final feature selection and informed model input choices.

---

### 🤖 Modeling
- Two models were trained and evaluated:
  - **Logistic Regression** (Youden J ≈ 0.42)  
  - **XGBoost** (Youden J ≈ 0.85)  
- XGBoost achieved stronger recall and overall balance, while Logistic Regression provided interpretability.  
- Youden’s J thresholds improved recognition of both successful and failed designs compared with the standard 0.5 cut-off.  

---

### 💡 Model Interpretability (SHAP)
- Key global contributors: **Phase**, **Masking**, **Model**, **Allocation**, and **Sponsor**.  
- Local SHAP plots confirmed that robust trial designs (e.g., randomized + double-masked + industry-sponsored) increased predicted success probabilities.

---

### ✅ Overall Outcome
- Developed a consistent, interpretable framework for predicting **trial design feasibility** using AACT data.  
- Cleaned, feature-optimized datasets were prepared for modeling and deployment.  
- Streamlit app enables real-time exploration of model predictions with automatic design validity checks.

---

## 🧭 Conclusion & Future Scope

This project demonstrates how **data-driven modeling can assist in clinical trial design assessment** using publicly available AACT data.  
By combining rigorous EDA, interpretable machine learning, and real-time deployment, the workflow provides a foundation for analyzing and predicting trial feasibility.

### 🔹 Key Outcomes
- Built a reproducible pipeline from raw AACT tables to model-ready datasets.  
- Identified statistically significant design factors influencing trial completion.  
- Implemented and deployed predictive models with explainability using SHAP.  
- Created an interactive Streamlit app for transparent result exploration.

### 🔹 Future Scope
- Extend modeling to **time-to-completion** or **success probability over time** using survival analysis.  
- Integrate **textual and protocol-level data** (e.g., study abstracts, interventions description).  
- Develop **Power BI dashboards** or cloud APIs for institutional analytics use.  
- Expand dataset to include **safety-related and adverse event metrics** for deeper clinical insights.

---

## 📚 Citation & Credits

If you reference or build upon this work, please cite:

> **Bhogadi, Sai Santoshi (2025).**  
> *Clinical Trial Outcome Prediction: A Machine Learning Approach Using AACT Data.*  
> GitHub Repository: [https://github.com/USERNAME/clinical-trial-outcome-prediction](https://github.com/USERNAME/clinical-trial-outcome-prediction)

**Data Source:** AACT (ClinicalTrials.gov)  
**Models Used:** Logistic Regression, XGBoost  
**Developed with:** Python, Pandas, scikit-learn, XGBoost, Streamlit  

---

📄 *This concludes the main README for the Clinical Trial Outcome Prediction project.*
