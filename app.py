from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Customer Churn Prediction API")

# Load trained pipeline
model = joblib.load("churn_model.pkl")

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: dict):
    values = np.array(list(features.values())).reshape(1, -1)
    prediction = model.predict(values)[0]
    probability = model.predict_proba(values)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(round(probability, 4))
    }
