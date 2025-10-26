# 🧠 Clinical Trial Outcome Prediction  
### End-to-End Machine Learning Pipeline using AACT (ClinicalTrials.gov) Data  

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📄 Short Overview
This repository presents an **end-to-end machine learning workflow** to predict whether a **clinical trial** is likely to **succeed or fail**, based on its *design characteristics* (phase, intervention type, allocation, masking, model, purpose, sponsor type, eligibility, enrollment, duration, etc.).

Built from the **AACT (ClinicalTrials.gov)** open registry, the project includes:
- Complete **data pipeline** — extraction, cleaning, merging, feature engineering
- **Exploratory Data Analysis (EDA)** — univariate, bivariate, and multivariate insights     
- **Modeling** — Logistic Regression & XGBoost with **Youden’s J** thresholds  
- **Interpretability** — SHAP (global + local)  
- **Deployment** — Streamlit app for real-time design feasibility prediction

> This work bridges **clinical research design** and **data science**, demonstrating how data-driven insights can guide early-phase feasibility assessments.

---

## 📘 Project Overview  

Clinical trials often fail due to design inefficiencies, under-enrollment, or unrealistic expectations of intervention success.  
This project aims to **predict the likelihood of trial completion (Success vs Failure)** by analyzing historical trial designs from the **AACT (ClinicalTrials.gov)** database.

### 🎯 Objectives  
- Identify which **design parameters** most influence trial success or failure.  
- Build an interpretable **machine-learning pipeline** to forecast trial outcomes.  
- Create a **Streamlit web app** for real-time prediction and feasibility validation.  
- Enable early detection of potentially **high-risk trial designs** before execution.  

### 🧱 Key Features  
- End-to-end reproducible pipeline: from raw AACT tables to deployment  
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
| **1. Data Cleaning & Integration** | `1_df_studies` → `8_df_merged` | Extract, clean, and merge core AACT tables (`studies`, `interventions`, `conditions`, `designs`, `eligibilities`, `sponsors`) |
| **2. Exploratory Data Analysis (EDA)** | `EDA_1`, `EDA_2`, `EDA_3` | Univariate → Bivariate → Multivariate analysis to identify influential design factors |
| **3. Feature Engineering & Preparation** | `9_df_final` | Handle missing values, apply log transformations, encode categorical variables, and prepare model-ready datasets |
| **4. Modeling & Evaluation** | `model_logreg`, `model_xgboost` | Train and optimize Logistic Regression and XGBoost models using Youden’s J threshold selection |
| **5. Interpretability (Explainability)** | `SHAP` | Global & local SHAP plots highlighting each feature’s contribution to success probability |
| **6. Deployment** | `/app/app.py` | Streamlit web app integrating both models with an interactive user interface and validity checks |

**End-to-End Flow**  
`AACT Database → Data Cleaning → EDA (1 → 2 → 3) → Feature Engineering → Modeling → SHAP Interpretation → Streamlit App Deployment`

---

## 🧩 Data Pipeline (Summary)
**Sources:** AACT (Aggregate Analysis of ClinicalTrials.gov) — standardized data for >430k trials.  
**Core tables:** `studies`, `interventions`, `conditions`, `eligibilities`, `sponsors`, `designs` were extracted using PostgreSQL and SQLAlchemy

### Cleaning & Transformations (high level)
- Standardized missing values; log-scaled `enrollment` & `duration`.
- Reassigned hybrid phases (e.g., *Phase 1/2*, *Phase 2/3*) using enrollment distribution logic.
- Grouped granular conditions/interventions into broader categories.
- Encoded trial design features (allocation, masking, model, purpose)  
- Added binary trial-level flags (`has_dmc`, `is_fda_regulated_drug`, `healthy_volunteers`, etc.)
- Simplified sponsor types; removed `sponsor_role` (dominantly “lead”).

### Merging & Outputs
- Tables merged on `nct_id` → `df_merged.csv` → (EDA) → final model-ready datasets:
  - **One-hot version** for Logistic Regression
  - **Grouped-categorical version** for XGBoost

### Storage Notes
- **Large/full datasets are stored locally** (to avoid GitHub size limits).  
- **Sample / downsampled** processed files are included in the repo for illustration and to ensure code runs.

### Folder mapping
- `/data/processed/` – cleaned and merged (sample) datasets
- `/data/final/` – model ready datasets (one-hot and grouped versions)
- `/results/` – EDA plots, statistical outputs, model metrics, SHAP visuals  
- `/models/` – serialized pipelines & feature lists  
- `/app/` – Streamlit deployment

*Detailed steps live inside the notebook markdowns and the `notebooks/README`.*

---

## 🤖 Modeling & Evaluation (Summary)
Two supervised machine learning models were developed to predict **clinical trial success (1)** or **failure (0)**:

- **Logistic Regression** (one-hot features) — interpretable baseline  
- **XGBoost** (grouped features) — stronger non-linear performance

### Features used
- **Categorical:** `Phase`, `Intervention Type`, `Condition`, `Allocation`, `Masking`, `Model`, `Purpose`, `Sponsor`  
- **Numerical:** `log_enrollment`, `log_duration`, `number_of_arms`, `intervention_count`  
- **Binary Flags:** `has_dmc`, `is_fda_regulated_drug`, `healthy_volunteers`, `has_expanded_access`, `is_fda_regulated_device`.

### Training
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

### Evaluation
- Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- PR curves to validate threshold choice

### Model Interpretability (SHAP Analysis)
- **Global:** Ranked features by average SHAP value to identify strong predictors (e.g., `log_enrollment`, `log_duration`, `phase`, `sponsor`, `model`, `intervention_type`). 
- **Local:** Force/waterfall plots explain individual predictions.

This dual-level explainability made the models both **transparent and scientifically interpretable**.

### Artifacts
- `/models/` – `logreg_pipeline.pkl`, `xgb_pipeline.pkl`, `X_cols_logreg.pkl`, `X_cols_xgb.pkl`  
- `/results/` – metrics, curves, SHAP plots

---

## 🌐 Streamlit App (Deployment Summary)
A Streamlit web application was built to make the model results interactive and interpretable.  
It integrates both **Logistic Regression** and **XGBoost** models with their **Youden’s J thresholds**
(LogReg ≈ 0.42, XGBoost ≈ 0.85).

### Highlights
- **Interactive design input:** Users can select trial parameters such as phase, intervention type, conditions, allocation, masking, sponsor etc.  
- **Automatic validity checks:** Flags logically invalid configurations (e.g., randomized single-group trials).  
- **Model interpretation section:** Explains threshold rationale, limitations, and ethical disclaimer in a collapsible block.

**Run locally**
```bash
cd app
streamlit run app.py
```

*All supporting files (app.py, requirements.txt, and model artifacts) are located in the /app directory.*
*Full app details live in /app/README*

---

## 📊 Results & Key Insights (Summary)
- Univariate, bivariate, and multivariate analyses revealed clear design-level patterns in clinical trial outcomes.  
- Higher completion rates: Phase 3, randomized, parallel, double-masked, often industry-sponsored.
- Lower completion rates: open-label single-group.  
- Behavioral interventions showed higher than expected completion rates. 
- **Logistic Regression** offered interpretability and transparent coefficients, serving as a robust statistical baseline.  
- **XGBoost** achieved better recall and overall balance, identifying more realistic design patterns linked to successful outcomes.
- Youden thresholds improved failure detection
- The combination of both provided a complete perspective — **explainability** from Logistic Regression and **predictive strength** from XGBoost.

---

## 🧭 Conclusion & Future Scope

This project demonstrates how **data-driven modeling can assist in clinical trial design assessment** using publicly available AACT data.  
By combining rigorous EDA, interpretable machine learning, and real-time deployment, the workflow provides a foundation for analyzing and predicting trial feasibility.

### Key Outcomes
- Built a reproducible pipeline from raw AACT tables to model-ready datasets.  
- Identified statistically significant design factors influencing trial completion.  
- Implemented and deployed predictive models with explainability using SHAP.  
- Created an interactive Streamlit app for transparent result exploration.

### Future Scope
- Extend modeling to **time-to-completion** or **success probability over time** using survival analysis.  
- Integrate **textual and protocol-level data** (e.g., study abstracts, interventions description).   
- Expand dataset to include **safety-related and adverse event metrics** for deeper clinical insights.

---

## ⚙️ Setup and Installation Instructions
This project can be run using either pip or conda.
Both **requirements.txt** and **environment.yml** files are provided to ensure reproducibility across environments.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/clinical-trial-outcome-prediction.git
cd clinical-trial-outcome-prediction
```

### 2️⃣ Setup Using pip (Lightweight Option)
Create and activate a virtual environment
```bash
python -m venv venv
# Activate the environment
venv\Scripts\activate       # For Windows
source venv/bin/activate    # For Mac/Linux
```
Install all required dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Setup Using Conda (Full Reproducibility)
Create the environment from the provided YAML file
```bash
conda env create -f environment.yml
# Activate the environment
conda activate ct_outcome
```

### 4️⃣ Verify Installation
Run the following command to ensure all dependencies are properly installed:
```bash
python -c "import pandas, sklearn, xgboost, shap, streamlit; print('✅ Environment setup successful!')"
```
If you see `✅ Environment setup successful!`, your environment is ready to use.

### 5️⃣ Run Jupyter Notebooks
If you plan to explore or modify the notebooks:
```bash
pip install ipykernel
python -m ipykernel install --user --name=ct_outcome --display-name "Python (ct_outcome)"
jupyter notebook
```
Then, in Jupyter, select the Python (ct_outcome) kernel before running any notebook.

### 6️⃣ Run the Streamlit App

```bash
cd app
streamlit run app.py
```
The app will open in your browser at **http://localhost:8501/**

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
