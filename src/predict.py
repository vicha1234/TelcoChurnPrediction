import joblib
import numpy as np
import pandas as pd


def predict_churn(customer_data: dict):
    model = joblib.load("models/decision_tree.pkl")
    scaler = joblib.load("models/scaler.pkl")

    df = pd.DataFrame([customer_data])

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 3),
        "message": "Customer will churn" if prediction == 1 else "Customer will stay"
    }


if __name__ == "__main__":
    sample_customer = {
        'gender': 1, 'SeniorCitizen': 0, 'Partner': 1, 'Dependents': 0,
        'tenure': 12, 'PhoneService': 1, 'MultipleLines': 0,
        'OnlineSecurity': 0, 'OnlineBackup': 1, 'DeviceProtection': 0,
        'TechSupport': 0, 'StreamingTV': 1, 'StreamingMovies': 1,
        'PaperlessBilling': 1, 'MonthlyCharges': 70.5, 'TotalCharges': 845.0,
        'InternetService_Fiber optic': 1, 'InternetService_No': 0,
        'Contract_One year': 0, 'Contract_Two year': 0,
        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 1,
        'PaymentMethod_Mailed check': 0,
    }

    result = predict_churn(sample_customer)
    print(result)