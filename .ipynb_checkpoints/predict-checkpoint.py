import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# Load trained pipeline model
model = joblib.load("churn_model.pkl")

app = FastAPI(title="Customer Churn Prediction API")


# Request schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# Health check
@app.get("/")
def home():
    return {"status": "API running"}


# Prediction endpoint
@app.post("/predict")
def predict(data: CustomerData):
    input_df = pd.DataFrame([data.dict()])
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(round(probability, 4))
    }
