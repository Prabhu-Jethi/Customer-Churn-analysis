import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE, 'src'))

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
 
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
 
.stApp {
    background: #0a0e1a;
}
 
section[data-testid="stSidebar"] { display: none; }
 
.main .block-container {
    padding: 2rem 2rem 4rem 2rem;
    max-width: 1100px;
}
 
/* Header */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.header-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
}
.header-title {
    font-size: 24px; font-weight: 600;
    color: #f1f5f9; margin: 0;
}
.header-sub {
    font-size: 13px; color: #64748b; margin: 2px 0 0;
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px #22c55e;
    margin-left: auto;
    display: inline-block;
}
.status-label {
    font-size: 12px; color: #22c55e; margin-left: 6px;
}
 
/* Cards */
.metric-card {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
}
.card-label {
    font-size: 11px; font-weight: 500;
    color: #475569; letter-spacing: 0.08em;
    text-transform: uppercase; margin: 0 0 6px;
}
.card-value {
    font-size: 28px; font-weight: 600;
    color: #f1f5f9; margin: 0;
    font-family: 'DM Mono', monospace;
}
.card-sub {
    font-size: 12px; color: #475569; margin: 4px 0 0;
}
 
/* Risk gauge */
.gauge-container {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.gauge-title {
    font-size: 11px; font-weight: 500;
    color: #475569; letter-spacing: 0.08em;
    text-transform: uppercase; margin: 0 0 1.5rem;
}
.gauge-ring-wrap {
    position: relative;
    width: 180px; height: 180px;
    margin: 0 auto 1rem;
}
.gauge-ring-wrap svg { overflow: visible; }
.gauge-percent {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 28px;  
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    color: #f1f5f9;
    white-space: nowrap;  
}
.gauge-unit {
    font-size: 14px; color: #64748b;
}
.risk-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px; font-weight: 500;
}
 
/* Section label */
.section-label {
    font-size: 11px; font-weight: 500;
    color: #475569; letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0 0 1rem;
}
 
/* Input styling overrides */
div[data-testid="stSlider"] > label {
    font-size: 13px !important;
    color: #94a3b8 !important;
}
div[data-testid="stSelectbox"] > label {
    font-size: 13px !important;
    color: #94a3b8 !important;
}
div[data-testid="stRadio"] > label {
    font-size: 13px !important;
    color: #94a3b8 !important;
}
 
/* Button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.85 !important;
}
 
/* Insights row */
.insight-row {
    display: flex; gap: 10px; flex-wrap: wrap;
    margin-top: 1rem;
}
.insight-chip {
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px; color: #93c5fd;
    display: flex; align-items: center; gap: 6px;
}
.insight-chip.warn {
    background: rgba(234,179,8,0.1);
    border-color: rgba(234,179,8,0.2);
    color: #fde047;
}
.insight-chip.danger {
    background: rgba(239,68,68,0.1);
    border-color: rgba(239,68,68,0.2);
    color: #fca5a5;
}
.insight-chip.safe {
    background: rgba(34,197,94,0.1);
    border-color: rgba(34,197,94,0.2);
    color: #86efac;
}
 
/* Factor bar */
.factor-row {
    display: flex; align-items: center;
    gap: 10px; margin: 8px 0;
}
.factor-name {
    font-size: 12px; color: #94a3b8;
    width: 140px; flex-shrink: 0;
}
.factor-bar-bg {
    flex: 1; height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px; overflow: hidden;
}
.factor-bar-fill {
    height: 100%; border-radius: 3px;
    transition: width 0.6s ease;
}
.factor-val {
    font-size: 11px; color: #475569;
    font-family: 'DM Mono', monospace;
    width: 36px; text-align: right;
}
 
/* Divider */
.divider {
    border: none; border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1.5rem 0;
}
 
/* Stcard */
.stcard {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
}
 
/* Mobile */
@media (max-width: 768px) {
    .main .block-container { padding: 1rem 1rem 3rem; }
    .app-header { flex-wrap: wrap; }
    .card-value { font-size: 22px; }
    .gauge-ring-wrap { width: 150px; height: 150px; }
    .gauge-percent { font-size: 28px; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE, 'src', 'xgb_model.pkl'))
    scaler = joblib.load(os.path.join(BASE, 'src', 'scaler.pkl'))
    return model, scaler


@st.cache_data
def get_feature_cols():
    df = pd.read_csv(os.path.join(BASE, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    # Encode binary cols exactly as in preprocessing.py
    bin_cols = ['gender','Partner','Dependents','PhoneService',
                'MultipleLines','OnlineSecurity','OnlineBackup',
                'DeviceProtection','TechSupport','StreamingTV',
                'StreamingMovies','PaperlessBilling']
    le = LabelEncoder()
    for col in bin_cols:
        df[col] = le.fit_transform(df[col])

    # One-Hot encoding
    df = pd.get_dummies(df, columns=['Contract','PaymentMethod','InternetService'])
    feature_cols = df.drop(['customerID','Churn'], axis=1).columns.tolist()
    df_features = df[feature_cols]
    return feature_cols, df_features
 
model, scaler = load_model()
feature_cols, df_features = get_feature_cols()




# ---- Streamlit UI ----

# ── Header ──────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="header-icon">📊</div>
  <div>
    <p class="header-title">Customer Churn Predictor</p>
    <p class="header-sub">XGBoost · Telco Dataset · ROC-AUC 0.90</p>
  </div>
  <span class="status-dot"></span>
  <span class="status-label">Model live</span>
</div>
""", unsafe_allow_html=True)
 
# ── Layout ───────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")
 
with left:
    st.markdown('<p class="section-label">Customer Profile</p>', unsafe_allow_html=True)
 
    with st.container():
        st.markdown('<div class="stcard">', unsafe_allow_html=True)
 
        tenure = st.slider("📅 Tenure (months)", 0, 72, 12,
                           help="How long the customer has been with the company")
        charges = st.slider("💵 Monthly Charges ($)", 18, 120, 65,
                            help="Current monthly bill amount")
 
        c1, c2 = st.columns(2)
        with c1:
            contract = st.selectbox("📋 Contract Type",
                ["Month-to-month", "One year", "Two year"])
        with c2:
            internet = st.selectbox("🌐 Internet Service",
                ["Fiber optic", "DSL", "No"])
 
        c3, c4 = st.columns(2)
        with c3:
            tech_support = st.selectbox("🛠 Tech Support", ["No", "Yes"])
        with c4:
            online_security = st.selectbox("🔒 Online Security", ["No", "Yes"])
 
        c5, c6 = st.columns(2)
        with c5:
            paperless = st.selectbox("📧 Paperless Billing", ["Yes", "No"])
        with c6:
            senior = st.selectbox("👤 Senior Citizen", ["No", "Yes"])
 
        st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    
 
with right:
    st.markdown('<p class="section-label">Risk Analysis</p>', unsafe_allow_html=True)
 
    # Build feature vector
    input_df = pd.DataFrame([df_features.median()], columns=feature_cols)
    input_df['tenure'] = tenure
    input_df['MonthlyCharges'] = charges
    input_df['SeniorCitizen'] = 1 if senior == "Yes" else 0
    input_df['TechSupport'] = 1 if tech_support == "Yes" else 0
    input_df['OnlineSecurity'] = 1 if online_security == "Yes" else 0
    input_df['PaperlessBilling'] = 1 if paperless == "Yes" else 0
    input_df['Contract_Month-to-month'] = 1 if contract == "Month-to-month" else 0
    input_df['Contract_One year']        = 1 if contract == "One year" else 0
    input_df['Contract_Two year']        = 1 if contract == "Two year" else 0
    input_df['InternetService_DSL']         = 1 if internet == "DSL" else 0
    input_df['InternetService_Fiber optic'] = 1 if internet == "Fiber optic" else 0
    input_df['InternetService_No']          = 1 if internet == "No" else 0
 
    input_scaled = scaler.transform(input_df)
    prob = float(model.predict_proba(input_scaled)[0][1])
    pct = round(prob * 100, 1)
 
    # Risk level
    if prob > 0.7:
        risk_label = "High Risk"
        ring_color = "#ef4444"
        badge_style = "background:rgba(239,68,68,0.15);color:#fca5a5;border:1px solid rgba(239,68,68,0.3)"
    elif prob > 0.4:
        risk_label = "Medium Risk"
        ring_color = "#eab308"
        badge_style = "background:rgba(234,179,8,0.15);color:#fde047;border:1px solid rgba(234,179,8,0.3)"
    else:
        risk_label = "Low Risk"
        ring_color = "#22c55e"
        badge_style = "background:rgba(34,197,94,0.15);color:#86efac;border:1px solid rgba(34,197,94,0.3)"
 
    # Circumference math for ring
    r = 70
    circ = round(2 * 3.14159 * r, 1)
    dash = round(circ * prob, 1)
 
    st.markdown(f"""
    <div class="gauge-container">
      <p class="gauge-title">Churn Probability</p>
      <div class="gauge-ring-wrap">
        <svg width="180" height="180" viewBox="0 0 180 180">
          <circle cx="90" cy="90" r="{r}"
            fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="12"/>
          <circle cx="90" cy="90" r="{r}"
            fill="none" stroke="{ring_color}" stroke-width="12"
            stroke-linecap="round"
            stroke-dasharray="{dash} {circ}"
            transform="rotate(-90 90 90)"
            style="transition: stroke-dasharray 0.8s ease;"/>
        </svg>
        <div class="gauge-percent">{pct:.1f}<span class="gauge-unit">%</span></div>
      </div>
      <span class="risk-badge" style="{badge_style}">{risk_label}</span>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # Metrics row
    m1, m2, m3 = st.columns(3)
    retention = round((1 - prob) * 100, 1)
    retention = float(f"{retention:.1f}")
    ltv_risk = "High" if prob > 0.7 else "Med" if prob > 0.4 else "Low"
 
    with m1:
        st.markdown(f"""
        <div class="metric-card">
          <p class="card-label">Retention</p>
          <p class="card-value">{retention:.1f}%</p>
          <p class="card-sub">Stay probability</p>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
          <p class="card-label">Tenure</p>
          <p class="card-value">{tenure} months</p>
          <p class="card-sub">{'Loyal' if tenure > 36 else 'New' if tenure < 12 else 'Growing'}</p>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
          <p class="card-label">LTV Risk</p>
          <p class="card-value">{ltv_risk}</p>
          <p class="card-sub">Revenue risk</p>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # Key risk drivers
    st.markdown('<p class="section-label">Risk Drivers</p>', unsafe_allow_html=True)
    factors = [
        ("Contract type",     0.9 if contract == "Month-to-month" else 0.3 if contract == "One year" else 0.1, "#ef4444" if contract == "Month-to-month" else "#22c55e"),
        ("Tenure",            max(0.05, 1 - tenure/72), "#ef4444" if tenure < 12 else "#eab308" if tenure < 36 else "#22c55e"),
        ("Monthly charges",   min(1.0, charges/120),    "#ef4444" if charges > 80 else "#eab308" if charges > 50 else "#22c55e"),
        ("Tech support",      0.1 if tech_support == "Yes" else 0.7, "#22c55e" if tech_support == "Yes" else "#ef4444"),
        ("Internet service",  0.85 if internet == "Fiber optic" else 0.4 if internet == "DSL" else 0.1, "#ef4444" if internet == "Fiber optic" else "#22c55e"),
    ]
    for name, val, color in factors:
        bar_w = round(val * 100)
        st.markdown(f"""
        <div class="factor-row">
          <span class="factor-name">{name}</span>
          <div class="factor-bar-bg">
            <div class="factor-bar-fill" style="width:{bar_w}%;background:{color}"></div>
          </div>
          <span class="factor-val">{bar_w}%</span>
        </div>""", unsafe_allow_html=True)
 
# ── Action insights (shown after predict) ───────────────
if True:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Recommended Actions</p>', unsafe_allow_html=True)
 
    insights = []
    if contract == "Month-to-month":
        insights.append(("warn", "💡", "Offer 1-year contract discount to lock in customer"))
    if charges > 80:
        insights.append(("danger", "⚠️", "High monthly charges — consider loyalty pricing"))
    if tenure < 12:
        insights.append(("warn", "📅", "Early-stage customer — onboarding support recommended"))
    if tech_support == "No":
        insights.append(("danger", "🛠", "No tech support — offer free trial of support plan"))
    if internet == "Fiber optic" and charges > 70:
        insights.append(("danger", "🌐", "Fiber optic + high charges = highest churn combo"))
    if tenure > 36 and contract != "Month-to-month":
        insights.append(("safe", "✅", "Long tenure + stable contract — low intervention needed"))
    if prob < 0.3:
        insights.append(("safe", "✅", "Customer is healthy — focus on upsell opportunities"))
 
    if not insights:
        insights.append(("warn", "📊", "Monitor this customer in next billing cycle"))
 
    chips_html = '<div class="insight-row">'
    for cls, icon, text in insights:
        chips_html += f'<div class="insight-chip {cls}">{icon} {text}</div>'
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)
 
# ── Footer ───────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<hr class="divider">
<p style="font-size:11px;color:#334155;text-align:center;margin:0">
  Built with XGBoost · SMOTE · StandardScaler · Streamlit &nbsp;·&nbsp; Telco Customer Churn Dataset
</p>
""", unsafe_allow_html=True)
