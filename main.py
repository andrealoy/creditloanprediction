from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'models', 'credit_risk_model_rf.pkl')
model = joblib.load(model_path)

app = FastAPI(title="Credit Score API", description="Predicts the risk of default for a client")

# 2. Define the input data model
class ClientData(BaseModel):
    credit_lines_outstanding: int
    loan_amt_outstanding: float
    total_debt_outstanding: float
    income: float
    years_employed: int
    fico_score: int

@app.get("/")
def home():
    return {"message": "Credit Scoring API is operational. Go to /docs to test."}

@app.post("/predict")
def predict(client: ClientData):
    # a. Convert the received JSON into a DataFrame
    input_df = pd.DataFrame([client.model_dump()])
    
    # b. Apply log1p transformation (crucial for the model to recognize the data)
    input_df['loan_amt_outstanding'] = np.log1p(input_df['loan_amt_outstanding'])
    input_df['total_debt_outstanding'] = np.log1p(input_df['total_debt_outstanding'])
    
    # c. Prediction
    probability = model.predict_proba(input_df)[0][1]
    prediction = int(model.predict(input_df)[0])
    
    # d. Return the result
    return {
        "status": "success",
        "default_probability": round(float(probability), 4),
        "prediction": "Default" if prediction == 1 else "Healthy",
        "risk_level": "High" if probability > 0.5 else "Low"
    }
