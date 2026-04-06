from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import mlflow

app = FastAPI(title="Credit Score API", description="Predicts the risk of default for a client")

# 1. Loading model from mlflow
model = mlflow.sklearn.load_model("models:/best_credit_loan_model/Production")
preprocessor = mlflow.sklearn.load_model("models:/credit_loan_preprocessor/latest")

# 2. Define the input data model
class ClientData(BaseModel):
    #credit_lines_outstanding: int - info non connue au moment de l'évaluation
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
    
    # b. Apply the preprocessor
    X_processed = preprocessor.transform(input_df)

    # c. Prediction
    probability = model.predict_proba(X_processed)[0][1]
    prediction = int(model.predict(X_processed)[0])

    dti = X_processed["dti"].values[0]
    #lti = X_processed["lti"].values[0]
    
    # d. Return the result
    return {
        "status": "success",
        "default_probability": round(float(probability), 4),
        "prediction": "Default" if prediction == 1 else "Healthy",
        "risk_level": "High" if probability > 0.5 else "Low",
        "dti": round(float(dti), 6),
        #"lti": round(float(lti), 6)
    }
