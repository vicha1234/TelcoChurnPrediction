import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.logger import logger

app = FastAPI(title="Telco Churn Prediction API")

# Load model and scaler when server starts
model = joblib.load("models/decision_tree.pkl")
scaler = joblib.load("models/scaler.pkl")

# Define what input data looks like
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    PaperlessBilling: int
    MonthlyCharges: float
    TotalCharges: float
    InternetService_Fiber_optic: int
    InternetService_No: int
    Contract_One_year: int
    Contract_Two_year: int
    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int


@app.get("/")
def home():
    return {"message": "Telco Churn Prediction API is running!"}


@app.get("/health")
def health_check():
    logger.info("Health check called")
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerData):
    # Convert input to dataframe
    data = {
        'gender': customer.gender,
        'SeniorCitizen': customer.SeniorCitizen,
        'Partner': customer.Partner,
        'Dependents': customer.Dependents,
        'tenure': customer.tenure,
        'PhoneService': customer.PhoneService,
        'MultipleLines': customer.MultipleLines,
        'OnlineSecurity': customer.OnlineSecurity,
        'OnlineBackup': customer.OnlineBackup,
        'DeviceProtection': customer.DeviceProtection,
        'TechSupport': customer.TechSupport,
        'StreamingTV': customer.StreamingTV,
        'StreamingMovies': customer.StreamingMovies,
        'PaperlessBilling': customer.PaperlessBilling,
        'MonthlyCharges': customer.MonthlyCharges,
        'TotalCharges': customer.TotalCharges,
        'InternetService_Fiber optic': customer.InternetService_Fiber_optic,
        'InternetService_No': customer.InternetService_No,
        'Contract_One year': customer.Contract_One_year,
        'Contract_Two year': customer.Contract_Two_year,
        'PaymentMethod_Credit card (automatic)': customer.PaymentMethod_Credit_card_automatic,
        'PaymentMethod_Electronic check': customer.PaymentMethod_Electronic_check,
        'PaymentMethod_Mailed check': customer.PaymentMethod_Mailed_check,
    }

    df = pd.DataFrame([data])

    # Scale numerical columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    logger.info(f"Prediction made - churn: {int(prediction)}, probability: {round(float(probability), 3)}")
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 3),
        "message": "Customer will churn" if prediction == 1 else "Customer will stay"
    }