
# 🧠 Clinical Trial Outcome Predictor — Streamlit App

This Streamlit web application predicts the likelihood of **clinical trial success or failure**
based on trial design inputs such as phase, intervention type, allocation, masking, and sponsor type.
It serves as a design-feasibility predictor, built from patterns observed in the **AACT (ClinicalTrials.gov)** dataset.

---

## 🔍 Overview

The app integrates two optimized machine learning models:

- **Logistic Regression (Youden’s J threshold ≈ 0.42)**
- **XGBoost (Youden’s J threshold ≈ 0.85)**

It also includes a **clinical trial design validity checker** that prevents logically impossible configurations
(e.g., randomized single-group studies, quadruple masking with single group).

---

## 🧭 App Workflow

1. **Enter trial design details** — Choose phase, intervention, condition, masking, model, etc.  
2. **Provide numerical inputs** — Enrollment size, study duration, number of arms, and intervention count.  
3. **Set binary flags** — e.g., DMC presence, FDA regulation, healthy volunteers.  
4. **Run Prediction** — Choose between Logistic Regression or XGBoost to predict the likelihood of success.  
5. **View Interpretation** — Expand the *Model Interpretation, Rationale & Disclaimer* section for guidance.  
6. **Automatic Validation** — Invalid trial designs are blocked before prediction.

---

## 🧠 Interpretation & Rationale

- Predictions indicate **statistical similarity** to historically successful or failed trials.  
- Youden’s J threshold ensures a balanced cutoff for identifying both success and failure patterns.  
- The model is optimized for **design-level insights**, not biological outcomes.  
- Failures may still be predicted as “Success” if they resemble historically robust designs.

---

## ⚙️ Key Files

| File | Description |
|------|--------------|
| `app.py` | Main Streamlit application script |
| `__init__.py` | Marks the app folder as a Python package |
| `requirements.txt` | Python dependencies for running the app |

---

## 🧩 Dependencies

Install all required packages before running the app:

pip install -r requirements.txt

Core libraries:
- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- pickle

---

## 🚀 Run the App Locally

From within the `/app` directory:

streamlit run app.py

Then open the local server URL (e.g., http://localhost:8501) in your browser.

---

## ⚖️ Disclaimer

This application is for **research and educational purposes only**.  
It does **not** provide medical, regulatory, or ethical guidance.  
Predictions are **probabilistic insights**, not deterministic outcomes.  
Always interpret results alongside expert judgment and official clinical guidelines.

---

### 🧾 Author & Credits

**Developed by:** Sai Santoshi Bhogadi  
**Project:** Clinical Trial Outcome Prediction (2025)  
**Data Source:** AACT (ClinicalTrials.gov)  
**Models Used:** Logistic Regression, XGBoost  

---

📁 *This README describes the Streamlit deployment module located in the `/app` folder of the Clinical Trial Outcome Prediction project.*

