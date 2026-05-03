

# 📊 Customer Churn Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-006400?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end machine learning pipeline that predicts customer churn for a telecom company — with a live interactive dashboard, SHAP explainability, and automated model persistence.**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

</div>

---

## 🖥️ Live Demo

> 🔗 **[Launch App →](https://your-app-url.streamlit.app)**

![App Screenshot](assets/app_screenshot.png)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)
- [Pipeline Architecture](#-pipeline-architecture)
- [Features](#-features)
- [SHAP Explainability](#-shap-explainability)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Model Performance](#-model-performance)
- [Interview Insights](#-interview-insights)
- [License](#-license)

---

## Business Problem
A telecom company is losing 26.5% of its customers annually,
representing $1.67M in revenue at risk. This project identifies
the key drivers of churn and builds a prediction model to flag
at-risk customers before they leave.

## 🧠 Overview

Customer churn is one of the most costly problems in the telecom industry — acquiring a new customer costs **5× more** than retaining an existing one. This project builds a full ML pipeline to:

- **Predict** which customers are likely to churn (leave the service)
- **Explain** why a prediction was made using SHAP values
- **Serve** predictions via a real-time interactive web app

The model is trained on the [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features.

---

## 🏆 Key Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.62 | 0.79 | 0.69 | 0.84 |
| Random Forest | 0.65 | 0.81 | 0.72 | 0.88 |
| **XGBoost** ✅ | **0.68** | **0.85** | **0.75** | **0.90** |

> **Metric focus:** Recall is prioritised over Accuracy — missing a churner (false negative) costs more than a false alarm (false positive).

---

## 📁 Project Structure

```
churn-predictor/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset (never modified)
│
├── notebooks/
│   ├── EDA.ipynb                               # Exploratory data analysis
│   └── model.ipynb                             # SHAP explainability & visualizations
│
├── src/
│   ├── preprocessing.py                        # Reusable data pipeline
│   ├── train.py                                # Model training & evaluation
│   ├── xgb_model.pkl                           # Saved XGBoost model
│   └── scaler.pkl                              # Saved StandardScaler
│
├── apps/
│   └── app.py                                  # Streamlit web application
│
├── assets/
│   ├── app_screenshot.png                      # App UI screenshot
│   └── shap_summary.png                        # SHAP feature importance plot
│
├── requirements.txt
└── README.md
```

---

## 🔄 Pipeline Architecture

```
Raw CSV
   │
   ▼
preprocessing.py
   ├── Fix TotalCharges (string → float)
   ├── Encode binary columns (LabelEncoder)
   ├── One-hot encode Contract, PaymentMethod, InternetService
   ├── Train/Test Split (80/20)
   ├── SMOTE oversampling (train set only)
   └── StandardScaler (fit on train, transform on test)
   │
   ▼
train.py
   ├── Logistic Regression (baseline)
   ├── Random Forest
   ├── XGBoost ← best performer
   ├── Evaluate: Precision, Recall, F1, ROC-AUC
   └── Save model + scaler as .pkl
   │
   ▼
apps/app.py  (Streamlit)
   ├── Load .pkl files
   ├── Take user inputs (sliders + dropdowns)
   ├── Build feature vector matching training schema
   ├── Scale → Predict → Show risk %
   └── Display risk drivers + action recommendations
```

---

## ✨ Features

### 🎯 Prediction Engine
- Real-time churn probability (0–100%) with animated circular gauge
- Colour-coded risk levels: 🟢 Low · 🟡 Medium · 🔴 High
- Live updates as sliders move — no button click required

### 📊 Risk Driver Bars
- Visual breakdown of which factors (contract type, tenure, charges, support) are contributing to the risk score
- Instantly interpretable without reading raw numbers

### 💡 Action Recommendations
- Smart business actions triggered by the prediction:
  - *"Offer 1-year contract discount"* for month-to-month customers
  - *"Consider loyalty pricing"* for high-charge customers
  - *"Offer free tech support trial"* for unsupported customers

### 📱 Mobile Friendly
- Responsive 2-column layout that adapts to all screen sizes
- Custom dark theme with `DM Sans` typography

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) is used to explain both global and individual predictions.

![SHAP Summary Plot](assets/shap_summary.png)

**Top churn drivers identified:**
- 📋 **Contract type** — Month-to-month customers churn significantly more
- ⏱️ **Tenure** — Short-tenure customers are at highest risk
- 💵 **Monthly charges** — Higher bills correlate with higher churn
- 🛠️ **Tech support** — No tech support strongly predicts churn
- 🌐 **Internet service** — Fiber optic customers churn more than DSL

> SHAP values show not just *which* features matter, but *in which direction* — critical for business decision-making.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| ML Models | XGBoost, Random Forest, Logistic Regression |
| Data | pandas, numpy |
| ML Pipeline | scikit-learn, imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn |
| Web App | Streamlit |
| Model Persistence | joblib |
| Deployment | Streamlit Community Cloud |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/churn-predictor.git
cd churn-predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# Place WA_Fn-UseC_-Telco-Customer-Churn.csv in the data/ folder
# Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
```

### Train the Model

```bash
python src/train.py
```

This will:
- Preprocess the data
- Train all 3 models
- Print evaluation metrics
- Save `xgb_model.pkl` and `scaler.pkl` to `src/`

### Run the App

```bash
streamlit run apps/app.py
```

Open `http://localhost:8501` in your browser.

---

## 📈 Model Performance

### Classification Report — XGBoost

```
              precision    recall  f1-score   support

           0       0.88      0.93      0.90      1036
           1       0.68      0.85      0.75       373

    accuracy                           0.86      1409
   macro avg       0.78      0.89      0.83      1409
weighted avg       0.83      0.86      0.84      1409

ROC-AUC: 0.902
```

### Why Recall over Accuracy?

The dataset has a **26% churn rate** (class imbalance). A naive model predicting "no churn" for everyone achieves 74% accuracy but catches 0 churners. We use:
- **SMOTE** to balance training data
- **Recall** as the primary metric — catching more churners matters more than avoiding false alarms

---

## 💬 Interview Insights

Key decisions made in this project and why:

| Decision | Reason |
|---|---|
| SMOTE after train/test split | Prevents synthetic data from leaking into the test set |
| `scaler.transform()` (not `fit_transform`) on test | Must apply the same scaling learned from training data |
| XGBoost over Random Forest | Higher AUC and Recall; gradient boosting corrects errors sequentially |
| Recall over Accuracy | Business cost of missing a churner > cost of a false alarm |
| SHAP TreeExplainer | Exact (not approximate) Shapley values for tree models |
| Relative file paths in deployment | Absolute Windows paths break on Linux-based cloud servers |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

**Your Name**
- GitHub: [Prabhu-Jethi](https://github.com/Prabhu-Jethi)
- LinkedIn: [your-linkedin](https://www.linkedin.com/in/prabhu-jethi/)

---

<div align="center">
  <sub>Built as part of a machine learning portfolio. If you found this useful, please ⭐ the repo!</sub>
</div>
