#!/usr/bin/env python
# coding: utf-8
"""
Clinical Trial Outcome Predictor — Streamlit App
------------------------------------------------
A Streamlit-based web application that predicts the likelihood of clinical trial success or failure 
based on trial design parameters. It integrates two machine learning models:

- Logistic Regression (optimized using Youden’s J threshold)  
- XGBoost (optimized using Youden’s J threshold)  

Both models were trained on the AACT (ClinicalTrials.gov) registry data.  
The app also performs basic clinical design validity checks and provides interpretive guidance to support design feasibility analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# === Page Configuration ===
st.set_page_config(page_title = "Clinical Trial Outcome Predictor", layout = "centered")
st.title("Clinical Trial Outcome Predictor")

st.markdown("""
This tool estimates the **likelihood of trial completion (success or failure)**  
based on design characteristics using machine learning models trained on the **AACT (ClinicalTrials.gov)** database.  
Predictions are derived from statistical patterns in historical clinical trial data.
""")

# === Load Models ===
@st.cache_resource
def load_models():
    logreg_model = joblib.load("../models/logreg_pipeline.pkl")
    with open("../models/X_cols_logreg.pkl", "rb") as f:
        X_cols_logreg = pickle.load(f)

    xgb_model = joblib.load("../models/xgb_pipeline.pkl")
    with open("../models/X_cols_xgb.pkl", "rb") as f:
        X_cols_xgb = pickle.load(f)

    # Load thresholds from metrics files
    best_thr_lr = pd.read_csv("../results/model_logreg/model_logreg_metrics.csv")['Best_threshold'].iloc[0]
    best_thr_xgb = pd.read_csv("../results/model_xgb/model_xgb_metrics.csv")['Best_threshold'].iloc[0]
    
    return logreg_model, X_cols_logreg, xgb_model, X_cols_xgb, best_thr_lr, best_thr_xgb

logreg_model, X_cols_logreg, xgb_model, X_cols_xgb, best_thr_lr, best_thr_xgb = load_models()

# === Mapping Dictionaries ===
phase_map = {'Not Applicable': 'phase_not applicable', 'Phase 1': 'phase_1', 'Phase 2': 'phase_2', 'Phase 3': 'phase_3', 'Phase 4': 'phase_4'}

intervention_map = {
    'Behavioral' : 'behavioral', 'Biological': 'biological', 'Combination Product': 'combination_product', 
    'Device': 'device', 'Diagnostic Test': 'diagnostic_test', 'Dietary Supplement': 'dietary_supplement',
    'Drug': 'drug', 'Genetic': 'genetic', 'Others': 'other', 'Procedure': 'procedure', 'Radiation': 'radiation'
}

condt_map = {
    'None': None, 'Cancer': 'cancers', 'Cardiovascular Disorder': 'cardiovascular_diseases',
    'Dental Disorder': 'dental_disorders', 'Dermatological Disorder': 'dermatological_disorders',
    'Endocrine/Metabolic Disorder': 'endocrine/metabolic_disorders',
    'Gastrointestinal Disorder': 'gastrointestinal_disorders', 'Genetic Disorder': 'genetic_disorders',
    'Infectious Disease': 'infectious_diseases', 'Mental Disorder': 'mental_disorders', 
    'Musculoskeletal Disorder': 'musculoskeletal_disorders', 'Neurological Disorder': 'neurological_disorders',
    'Ophthalmological Disorder': 'ophthalmological_disorders', 
    'Others': 'others', 'Pain Disorder': 'pain_disorders', 'Renal/Urological Disorder': 'renal/urological_disorders',
    'Reproductive Health': 'reproductive_health', 'Respiratory Disorder': 'respiratory_disorders'
}

allo_map = {'None': None, 'Non-Randomized': 'non_randomized', 'Randomized': 'randomized', 'Unknown': 'unknown'}

masking_map = {'None': None, 'Double' : 'double', 'Quadruple' : 'quadruple', 'Single': 'single', 'Triple' : 'triple', 'Unknown': 'unknown'}

model_map = {
    'None': None, 'Crossover' : 'crossover', 'Factorial' : 'factorial', 'Parallel' : 'parallel', 
    'Sequential': 'sequential', 'Single Group': 'single_group', 'Unknown': 'unknown'
} 
    
purpose_map = {
    'None': None, 'Diagnostic': 'diagnostic', 'Others': 'other', 'Prevention': 'prevention', 
    'Research': 'research', 'Supportive Care':  'supportive_care', 'Treatment' : 'treatment'
}

sponsor_map = {'Other': 'Other', 'Industry': 'Industry', 'Multiple': 'Multiple', 'Government': 'Government', 'Unknown' : 'Unknown'}

age_map = {'Adult': 'adult', 'Child': 'child', 'Senior': 'senior', 'Mixed': 'mixed', 'Unknown' : 'Unknown'}

gender_map = {'All': 'all', 'Male': 'male', 'Female': 'female'}

binary_flags = [
    "has_dmc", "has_expanded_access", "is_fda_regulated_drug",
    "is_fda_regulated_device", "healthy_volunteers"
]

flag_map = {"No": 0, "Yes": 1, "Unknown": -1}
invt_map = {"No": 0, "Yes": 1}

# === Model Selection ===
model_choice = st.radio("Choose Model:", ["Logistic Regression", "XGBoost"])
user_input = {}

# === User Input Widgets ===
st.markdown("### Trial Design Details")

# Phase and Intervention
col1, col2 = st.columns(2)
with col1: sel_phase = st.selectbox("Phase", sorted(list(phase_map.keys())))
with col2: sel_int = st.selectbox("Intervention Type", sorted(list(intervention_map.keys())))

# Condition and Allocation
col3, col4 = st.columns(2)
with col3: sel_cond = st.selectbox("Condition", sorted(list(condt_map.keys())))
with col4: sel_allo = st.selectbox("Allocation", sorted(list(allo_map.keys())))

# Masking and Model
col5, col6 = st.columns(2)
with col5: sel_masking = st.selectbox("Masking", sorted(list(masking_map.keys())))
with col6: sel_model = st.selectbox("Study Model", sorted(list(model_map.keys())))

# Purpose and Sponsor
col7, col8 = st.columns(2)
with col7: sel_pur = st.selectbox("Study Purpose", sorted(list(purpose_map.keys())))
with col8: sel_sponsor = st.selectbox("Sponsor Type", sorted(list(sponsor_map.keys())))

# Age and Gender
col9, col10 = st.columns(2)
with col9: sel_age = st.selectbox("Age (eligibility)", sorted(list(age_map.keys())))
with col10: sel_gender = st.selectbox("Gender (eligibility)", sorted(list(gender_map.keys())))

# Numeric Inputs
st.markdown("### Numerical Inputs")
col11, col12 = st.columns(2)
with col11:
    enrollment_raw = st.number_input("Enrollment (participants)", min_value=0, max_value=200000, step=10)
with col12:
    duration_days  = st.number_input("Duration (days)", min_value=0, max_value=10000, step=10)

user_input["log_enrollment"] = float(np.log1p(enrollment_raw))  
user_input["log_duration"]   = float(np.log1p(duration_days))

col13, col14 = st.columns(2)
with col13:
    user_input["number_of_arms"] = st.number_input("Number of Arms", min_value=1, max_value=20, step=1)
with col14:
    user_input["intervention_count"] = st.number_input("Intervention Count", min_value=1, max_value=10, step=1)

# === Flag Inputs ===
st.markdown("### Flags")
for flag in binary_flags:
    sel = st.selectbox(flag.replace("_", " ").capitalize(), list(flag_map.keys()))
    user_input[flag] = flag_map[sel]
    
sel_invt_option = st.selectbox('Has Multiple Intervention Types', list(invt_map.keys()))
user_input["has_multiple_intervention_types"] = invt_map[sel_invt_option]

# === Clinical Trial Design Validity Checker ===
invalid = False
invalid_reasons = []

# Critical Invalid Rules (Block Prediction)
if sel_allo == "Randomized" and sel_model == "Single Group":
    invalid = True
    invalid_reasons.append("Randomized allocation with Single Group model is invalid.")

if sel_masking == "Quadruple" and sel_model == "Single Group":
    invalid = True
    invalid_reasons.append("Quadruple masking cannot be used with Single Group design.")

if sel_model == "Parallel" and sel_allo == "Randomized" and number_of_arms == 1:
    invalid = True
    invalid_reasons.append("Parallel Randomized studies must have more than one arm.")

# Display Validation Messages
if invalid:
    st.error("⚠️ Clinically invalid trial design detected. Please review the issues below:")
    for r in invalid_reasons:
        st.markdown(f"- {r}")
    st.stop() 

# === Prediction ===
if st.button("Predict Outcome"):
    if model_choice == "Logistic Regression":
        # Reset all one-hot columns
        for col in X_cols_logreg:
            if col.startswith(("phase_", "intervention_", "condt_", "allocation_", "masking_", "model_", "purpose_", "sponsor_", "elig_age_", "elig_gender_")):
                user_input[col] = 0

        # Set selected categorical values (one-hot encoding)
        for sel, mapping, prefix in [
            (sel_phase, phase_map, ""),
            (sel_int, intervention_map, "intervention_"),
            (sel_cond, condt_map, "condt_"),
            (sel_allo, allo_map, "allocation_"),
            (sel_masking, masking_map, "masking_"),
            (sel_model, model_map, "model_"),
            (sel_pur, purpose_map, "purpose_"),
            (sel_sponsor, sponsor_map, "sponsor_"),
            (sel_age, age_map, "elig_age_"),
            (sel_gender, gender_map, "elig_gender_"),
        ]:
            mapped_val = mapping[sel]
            if mapped_val is not None:
                col_name = f"{prefix}{mapped_val}"
                if col_name in X_cols_logreg:
                    user_input[col_name] = 1

        input_df = pd.DataFrame([user_input]).reindex(columns=X_cols_logreg, fill_value=0)
        proba = logreg_model.predict_proba(input_df)[0][1]
        prediction = 1 if proba >= best_thr_lr else 0

    else:  # XGBoost
        user_input["phase_grouped"] = sorted(list(phase_map.keys())).index(sel_phase)
        user_input["intervention_grouped"] = sorted(list(intervention_map.keys())).index(sel_int) 
        user_input["condt_grouped"] = sorted(list(condt_map.keys())).index(sel_cond) 
        user_input["allocation_grouped"] = sorted(list(allo_map.keys())).index(sel_allo) 
        user_input["masking_grouped"] = sorted(list(masking_map.keys())).index(sel_masking) 
        user_input["model_grouped"] = sorted(list(model_map.keys())).index(sel_model) 
        user_input["purpose_grouped"] = sorted(list(purpose_map.keys())).index(sel_pur) 
        user_input["sponsor_grouped"] = sorted(list(sponsor_map.keys())).index(sel_sponsor) 
        user_input["elig_age_grouped"] = sorted(list(age_map.keys())).index(sel_age) 
        user_input["elig_gender_grouped"] = sorted(list(gender_map.keys())).index(sel_gender) 

        input_df = pd.DataFrame([user_input]).reindex(columns=X_cols_xgb, fill_value=0)
        proba = xgb_model.predict_proba(input_df)[0][1]
        prediction = 1 if proba >= best_thr_xgb else 0


    # === Display Results ===
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ Predicted Outcome: Success (Confidence: {proba*100:.2f}%)")
    else:
        st.error(f"❌ Predicted Outcome: Failure (Success probability: {proba*100:.2f}%)")

    st.caption(f"Model Used: {model_choice} (Youden threshold applied)")

# =========================================================
# 🧠 INTERPRETATION, THRESHOLD RATIONALE & DISCLAIMER
# =========================================================
st.markdown("---")
with st.expander("### Model Interpretation, Rationale & Disclaimer", expanded = False):
    st.markdown("""
**Interpretation:**  
The predicted outcome represents how closely the entered trial design resembles *historically successful* or *failed* trials in the AACT dataset.  
A “Success” prediction means the design statistically aligns with trials that completed successfully,  
while a “Failure” prediction indicates similarity to trials that were terminated or incomplete.

---

**Why Youden’s J Threshold:**  
Instead of the default 0.5 cutoff, the Youden index (J = Sensitivity + Specificity − 1) was used to find an **optimal threshold** that balances the ability to identify both successful and failed designs.  
This ensures fair performance in an **imbalanced dataset**, where most trials are labeled as “Completed” and only a minority are “Failed” or “Terminated.”  
- Logistic Regression threshold ≈ **0.42**  
- XGBoost threshold ≈ **0.85**

---

**Why Some Failures May Not Be Captured:**  
The model focuses on *design-level* factors such as phase, intervention type, condition, allocation, masking, model, sponsor, and enrollment.  
However, many real-world failures occur due to **biological inefficacy**, **adverse events**, **patient recruitment**, or 
**financial termination** — factors not represented in the dataset.  
Thus, the model should be viewed as a **design-feasibility predictor**, not a measure of drug efficacy or safety.

---

**Why Some Failures Are Predicted as Success:**  
Because most clinical trials in the dataset are labeled “Completed,”  
the model naturally learns a **bias toward success-like designs** (e.g., randomized, double-masked, parallel studies with adequate enrollment).  
Therefore, some failed trials may still be classified as “Success” if their structure resembles historically robust designs.  
This reflects **statistical similarity**, not real-world outcome certainty.

---

**Clinical Validity Checks:**  
Before prediction, the app automatically flags **invalid design combinations**, such as:  
- Randomized + Single Group design  
- Quadruple masking with Single Group model  
- Parallel randomized study with only one arm  
These are blocked to maintain realistic and clinically valid configurations.

---

**Disclaimer:**  
This tool is designed for **research and educational purposes only**.  
It does **not** substitute for scientific, ethical, or regulatory review.  
Predictions are **probabilistic insights**, not deterministic outcomes.  
Always interpret results in consultation with domain experts and official trial guidelines.

---

**Use Responsibly:**  
- Treat model output as an early-stage **design quality indicator**, not a final decision metric.  
- Combine with expert judgment, prior evidence, and clinical rationale.  
- Invalid combinations indicate design logic issues, not biological impossibility.
    """)

st.markdown(
    "<p style='text-align:center; color:gray; font-size:13px;'>"
    "Developed by Sai Santoshi Bhogadi | Clinical Trial Outcome Prediction Project | 2025<br>"
    "Data Source: AACT (ClinicalTrials.gov) | Models: Logistic Regression & XGBoost"
    "</p>",
    unsafe_allow_html=True
)
