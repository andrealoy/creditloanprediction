import mlflow
import mlflow.sklearn
import joblib

# 1. Charger les fichiers locaux
model = joblib.load("models/credit_risk_model_rf.pkl")
preprocessor = joblib.load("models/scaler_credit.pkl")

# 2. Créer une expérience MLflow (optionnel mais propre)
mlflow.set_experiment("credit-loan")

# 3. Log + register le modèle
with mlflow.start_run():

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="best_credit_loan_model"
    )

    mlflow.sklearn.log_model(
        preprocessor,
        artifact_path="preprocessor",
        registered_model_name="credit_loan_preprocessor"
    )

print("✅ Models registered in MLflow")