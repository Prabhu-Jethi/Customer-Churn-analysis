import sys
import joblib
sys.path.append(r"D:\Python\Churn_predictor\src")
from preprocessing import preprocess_customer

## Load saved artifacts 
model = joblib.load(r"D:\Python\Churn_predictor\src\xgb_model.pkl")
scaler = joblib.load(r"D:\Python\Churn_predictor\src\scaler.pkl")
feature_columns = joblib.load(r"D:\Python\Churn_predictor\src\feature_columns.pkl")

customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "PaperlessBilling": "Yes",
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "InternetService": "Fiber optic"
}

X = preprocess_customer(
    customer,
    feature_columns,
    scaler
)

print("\nProcessed Shape:", X.shape)
print("\nExpected Features:", len(feature_columns))

print("\nFirst 10 processed values:")
print(X[0][:10])

prediction = int(model.predict(X)[0])
probability = model.predict_proba(X)[0][1]

print("\nChurn Prediction:", prediction)
print("\nChurn Probability:", probability)
print("\nChurn Percentage:", round(probability * 100, 2))
