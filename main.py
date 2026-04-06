from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import mlflow
from functools import lru_cache


app = FastAPI(
    title="Credit Score API",
    description="Predicts the risk of default for a client"
)


@lru_cache
def get_model():
    return mlflow.sklearn.load_model("models:/best_credit_loan_model/1")


@lru_cache
def get_preprocessor():
    return mlflow.sklearn.load_model("models:/credit_loan_preprocessor/1")


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
    input_df = pd.DataFrame([client.model_dump()])

    preprocessor = get_preprocessor()
    model = get_model()

    X_processed = preprocessor.transform(input_df)

    probability = model.predict_proba(X_processed)[0][1]
    prediction = int(model.predict(X_processed)[0])

    return {
        "status": "success",
        "default_probability": round(float(probability), 4),
        "prediction": "Default" if prediction == 1 else "Healthy",
        "risk_level": "High" if probability > 0.5 else "Low"
    }