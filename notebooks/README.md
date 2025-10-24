# Clinical Trial Outcome Prediction — Notebooks Overview

This folder contains the complete **data preparation, exploratory analysis, and modeling workflow** for the *Clinical Trial Outcome Prediction* project.  
Each notebook builds sequentially on the previous step, ensuring a reproducible, end-to-end data science pipeline — from raw AACT tables to model interpretability and Streamlit deployment.

---

## 🧩 Notebook Sequence & Descriptions

### 🧹 **Data Preprocessing & Cleaning**

Each table from the AACT (ClinicalTrials.gov) database was cleaned individually and stored in the `../data/processed/` folder for traceability.

| Notebook | Description | Output |
|-----------|-------------|---------|
| `1_df_studies.ipynb` | Cleans and standardizes the **`studies`** table. Handles missing values, converts date columns, calculates duration of the study, fills `enrollment` by phase-wise median, and reassigns combined phases (e.g., *Phase 1/2*) based on enrollment distribution. | `1_df_studies.csv` |
| `2_df_baseline_features.ipynb` | Processes **baseline counts & measurements** to generate participant-level demographics. Handles missingness and merges into summarized baseline features. **baseline features** were later dropped since most of the data was missing and inconsistent | `2_df_baseline_features.csv` |
| `3_df_interventions.ipynb` | Simplifies and one-hot encodes **intervention types** (Drug, Device, Behavioral, etc.), aggregating them by trial ID. | `3_df_interventions.csv` |
| `4_df_conditions.ipynb` | Groups 250+ specific disease terms into 15+ broader **condition categories** (e.g., Cancer, Cardiovascular, Neurological) and one-hot encodes condition groups. | `4_df_conditions.csv` |
| `5_df_designs.ipynb` | Extracts, standardizes and one-hot encodes **design attributes** such as Allocation, Masking, Model, and Purpose for modeling. | `5_df_designs.csv` |
| `6_df_eligibilities.ipynb` | Cleans eligibility-level data, generating flags for **age group**, **gender**, **healthy volunteers** categories. | `6_df_eligibilities.csv` |
| `7_df_sponsors.ipynb` | Aggregates and classifies sponsors (Industry, Government, Other). **Sponsor roles** were later dropped since most were “Lead.” | `7_df_sponsors.csv` |
| `8_df_merged.ipynb` | Merges all cleaned datasets into a **single master file (`df_merged.csv`)** for EDA. | `df_merged.csv` |
| `9_df_final.ipynb` | Incorporates EDA insights to remove non-informative columns (e.g., `sponsor_role`, `high_enrollment_flag`) and apply log-transformations to `enrollment` and `duration`, finalizing the **model-ready dataset**. | `df_final_grouped.csv`, `df_final_onehot.csv`|

📂 **Processed datasets:** stored in `../data/processed/`
📂 **Final modeling-ready datasets:** stored in `../data/final/` 

---

### 📊 **Exploratory Data Analysis (EDA)**

| Notebook | Focus | Output |
|-----------|--------|---------|
| `df_EDA_1.ipynb` | **Univariate Analysis** – explores distributions of numerical and categorical variables; identifies outliers, visualizes class imbalance and summary statistics. | `df_EDA_1.csv` |
| `df_EDA_2.ipynb` | **Bivariate Analysis** –investigates associations between features and `overall_status` using Chi², Cramér’s V, and Mann–Whitney U tests. Inline plots are shown for features with strong associations (Cramér’s V > 0.1). | `df_EDA_2.csv` |
| `df_EDA_3.ipynb` | **Multivariate Analysis** – explores interactions between multiple variables (e.g., Phase × Intervention, Condition × Model) that influence trial success, residual heatmaps, and dependency patterns. | `df_EDA_3.csv` |

📂 **EDA Outputs:**  
All visualizations, residual heatmaps, and test summaries are saved under:
`../results/EDA1_outputs/`
`../results/EDA2_outputs/`
`../results/EDA3_outputs/`

📂 **Each stage also exports cleaned intermediate datasets to**: `../data/processed/`

---

### 🤖 **Modeling & Interpretability**

| Notebook | Description | Output |
|-----------|-------------|---------|
| `model_logreg.ipynb` | Trains a **Logistic Regression** model using the Youden’s J threshold (~0.42). Tests multiple solvers (LBFGS, SAGA) and regularizations (L1/L2). | `logreg_pipeline.pkl` in `../models/` |
| `model_xgboost.ipynb` | Builds an **XGBoost Classifier** with tuned hyperparameters (`n_estimators`, `max_depth`, `scale_pos_weight`) and applies the Youden threshold (~0.85). | `xgb_pipeline.pkl` in `../models/` |
| `SHAP.ipynb` | Performs **SHAP-based interpretability** for both models, including global (summary, bar) and local (force, waterfall) visualizations. | SHAP plots in `../results/shap_outputs/` |

📂 **Model Artifacts & Results**
- Pipelines: `../models/logreg_pipeline.pkl`, `../models/xgb_pipeline.pkl`  
- Feature Encodings: `../models/X_cols_logreg.pkl`, `../models/X_cols_xgb.pkl`  
- Metrics & thresholds:  
  - `../results/model_logreg/model_logreg_metrics.csv`  
  - `../results/model_xgb/model_xgb_metrics.csv`

---
## 🔑 Key Insights

- **Phase matters most:** Later phases (3–4) show the highest completion rates, while **Phase 2** carries a higher share of failures compared to 3–4.  
- **Condition group signal:** **Oncology (Cancer)** trials are over-represented among failures, whereas most other disease groups stay near baseline.  
- **Intervention pattern:**  **Drug** and **Device** trials exhibit the **highest success rates (82–92%)**, reflecting strong regulatory oversight and standardized protocols.  
  **Behavioral** interventions also show **better-than-expected success rates**, possibly due to shorter study durations and lower operational complexity.  
- **Design strength:** **Randomized**, **Parallel**, and **blinded (Double/Triple/Quadruple)** designs associate with higher completion than **Non-randomized**, **Single-group**, or open-label setups.  
- **Scale effects:** Higher **log_enrollment** is strongly linked to success; **log_duration** shows a smaller positive effect. **>2 interventions** and overly complex setups reduce completion.  
- **SHAP interpretability (both models):** Top contributors are **Phase**, **Intervention Type**, and **Study Model**; **Masking** and **Allocation** add consistent lift; **log_enrollment** is a stable positive driver.  
- **Thresholding:** Youden’s J was used to balance sensitivity/specificity given class imbalance (approx. **0.42** for Logistic Regression; **0.85** for XGBoost).  
- **Scope note:** Models reflect **design-level feasibility** (registry patterns). Biological, safety, and operational factors such as adverse events, funding, or recruitment issues are not modeled directly.

---

## 🚀 Next Steps

1. Move to the `/app/` folder for **Streamlit deployment** of both models.  
2. Refer to the **root-level `README.md`** for project overview, data sources, and environment setup.

---

**Developed by:** *Sai Santoshi Bhogadi*  
**Project:** *Clinical Trial Outcome Prediction (2025)*  
**Data Source:** AACT (ClinicalTrials.gov)  
**Models:** Logistic Regression | XGBoost | SHAP Interpretability  

---
