# Customer Churn Prediction – ML Capstone Project

## Project Overview
This project builds a binary classification machine learning model to predict whether a customer will churn. 
The model uses customer demographic, service, and billing features to estimate churn probability and exposes predictions via a REST API.

This helps:
- Improve retention by identifying likely churners
- Prioritize customer support actions
- Support business decision-making

## Dataset
The dataset is from the Telco Customer Churn dataset, containing features like:

- Gender, SeniorCitizen, Partner, Dependents
- Tenure, Contract, PaymentMethod, PaperlessBilling
- Services subscribed (Phone, Internet, TV, etc.)
- Billing amounts (MonthlyCharges, TotalCharges)
- Target variable: Churn (Yes / No)

The dataset is stored in `data/churn.csv`. If missing, instructions in `notebook.ipynb` show how to download it.

## Contents of This Repository

| File | Description |
|------|-------------|
| `notebook.ipynb` | EDA, model training, tuning |
| `train.py` | Train and save final model |
| `app.py` | FastAPI web service exposing `/health` and `/predict` |
| `Dockerfile` | Containerization |
| `requirements.txt` | Dependencies |
| `data/churn.csv` | Dataset |
| `deployment/cloud_deploy.md` | Deployment instructions |

---

## Model Training & Tuning
- Extensive EDA performed
- Models trained: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Hyperparameter tuning via `GridSearchCV`
- Comparison table included
- Best models exported using `joblib` / `pickle`

---

## Running Locally

### 1. Create environment & install dependencies
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt

### Start API
``` bash
python app.py

### Health check
```bash
curl http://localhost:9696/health
# Expected response:
# {"status":"ok"}

### Task predictions
```bash
curl -X POST http://localhost:9696/predict \
-H "Content-Type: application/json" \
-d '{
  "gender":"Male",
  "SeniorCitizen":0,
  "Partner":"Yes",
  "Dependents":"No",
  "tenure":12,
  "PhoneService":"Yes",
  "MultipleLines":"No",
  "InternetService":"DSL",
  "OnlineSecurity":"Yes",
  "OnlineBackup":"No",
  "DeviceProtection":"Yes",
  "TechSupport":"No",
  "StreamingTV":"No",
  "StreamingMovies":"No",
  "Contract":"Month-to-month",
  "PaperlessBilling":"Yes",
  "PaymentMethod":"Electronic check",
  "MonthlyCharges":70.35,
  "TotalCharges":845.5
}'

## Dockerization

### Build image
```bash
docker build -t churn-predictor .

### Run container
```bash
docker run -p 9696:9696 churn-predictor

### Test endpoints
```bash
curl http://localhost:9696/health
curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d '{...}'

## Evidence
<img width="654" height="961" alt="Screenshot 2026-01-20 042223" src="https://github.com/user-attachments/assets/d67edcf8-00cf-4c5a-a524-bc988b947f90" />

<img width="669" height="237" alt="Screenshot 2026-01-20 042236" src="https://github.com/user-attachments/assets/7f63f8c4-2ef3-4ca9-8e87-1f1d0be96ff0" />

<img width="690" height="43" alt="Screenshot 2026-01-20 051454" src="https://github.com/user-attachments/assets/9234156b-21cc-4394-b28e-c9d9d87c62e6" />

<img width="769" height="365" alt="Screenshot 2026-01-20 055344" src="https://github.com/user-attachments/assets/da709661-09ef-41f4-85ce-822b074b906f" />

<img width="763" height="121" alt="Screenshot 2026-01-20 055427" src="https://github.com/user-attachments/assets/7898ae82-32c2-4dad-a92f-6fc7dfa2f6a4" />

<img width="749" height="89" alt="Screenshot 2026-01-20 055540" src="https://github.com/user-attachments/assets/a57605d8-acb1-4328-a530-9985ed908d71" />

<img width="1907" height="990" alt="Screenshot 2026-01-20 064904" src="https://github.com/user-attachments/assets/5b0fa74e-d232-4a49-9bae-6b9e36bd2fa8" />

<img width="974" height="178" alt="Screenshot 2026-01-20 124905" src="https://github.com/user-attachments/assets/22a321cf-e689-4786-82ef-3476a35191c0" />

## URL
[Render link] (https://customer-churn-project-07in.onrender.com)
