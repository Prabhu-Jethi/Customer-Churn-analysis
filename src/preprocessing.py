import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    
    # Fix TotalCharges
    df['TotalCharges'] = df['TotalCharges'].replace(' ', float('nan'))
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Encode target
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    # Encode binary columns
    bin_cols = ['gender','Partner','Dependents','PhoneService',
                'MultipleLines','OnlineSecurity','OnlineBackup',
                'DeviceProtection','TechSupport','StreamingTV',
                'StreamingMovies','PaperlessBilling']
    le = LabelEncoder()
    for col in bin_cols:
        df[col] = le.fit_transform(df[col])
    
    # One-hot encode
    df = pd.get_dummies(df, columns=['Contract','PaymentMethod','InternetService'])
    
    X = df.drop(['customerID','Churn'], axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # SMOTE on train only
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    # Scale
    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_res)
    X_test = scaler.transform(X_test)
    
    return X_res, X_test, y_res, y_test, scaler, X.columns.to_list()


def preprocess_customer(customer_data, feature_columns, scaler):

    '''Preprocess single customer data for model inference'''

    df = pd.DataFrame([customer_data])

    # Encode binary columns
    bin_cols = [
        'gender',
        'Partner',
        'Dependents',
        'PhoneService',
        'MultipleLines',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'PaperlessBilling'
    ]

    for col in bin_cols:
        if col in df.columns:
            le = LabelEncoder()
            le.fit(['No', 'Yes'])

            # gender uses Female/Male instead
            if col == 'gender':
                le.fit(['Female', 'Male'])

            df[col] = le.transform(df[col])

    # One-hot encode categorical columns
    df = pd.get_dummies(
        df,
        columns=[
            'Contract',
            'PaymentMethod',
            'InternetService'
        ]
    )

    # Remove columns that the model does not use
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Make sure every training feature exists
    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Apply the scaler fitted during training
    X = scaler.transform(df)

    return X