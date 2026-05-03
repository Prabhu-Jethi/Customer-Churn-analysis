import sys
sys.path.append('D:/Python/Churn_predictor/src')
from preprocessing import load_and_preprocess

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

X_res, X_test, y_res, y_test, scaler = load_and_preprocess(
    r'D:\Python\Churn_predictor\data\WA_Fn-UseC_-Telco-Customer-Churn.csv')

models = {
  'Logistic Regression': LogisticRegression(max_iter=1000),
  'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
  'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for name, model in models.items():
    model.fit(X_res, y_res)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    print(f"\n{name}")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")

# Save model and scaler
joblib.dump(model, r'D:\Python\Churn_predictor\src\xgb_model.pkl')
joblib.dump(scaler, r'D:\Python\Churn_predictor\src\scaler.pkl')

print("Model and scaler saved successfully!")