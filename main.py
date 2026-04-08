import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None
    MlflowClient = None


app = FastAPI(title="Credit Score API", description="Predicts the risk of default for a client")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "credit_risk_model_xgb.pkl"
DEFAULT_TRACKING_URI = f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"
DEFAULT_REGISTRY_URI = DEFAULT_TRACKING_URI
DEFAULT_MODEL_NAME = "best_credit_loan_model"
DEFAULT_PREPROCESSOR_NAME = "credit_loan_preprocessor"
DEFAULT_MODEL_STAGE = "Production"
LOCAL_FALLBACK_CREDIT_LINES_OUTSTANDING = 0


class ClientData(BaseModel):
    loan_amt_outstanding: float = Field(gt=0)
    total_debt_outstanding: float = Field(gt=0)
    income: float = Field(gt=0)
    years_employed: int = Field(ge=0)
    fico_score: int = Field(ge=300, le=850)


class CompatDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return CompatDataFrame

    @property
    def np(self):
        return np


class Predictor(Protocol):
    def predict(self, client: ClientData) -> dict:
        ...


class LocalPredictor:
    def __init__(self, model_path: Path) -> None:
        self.model = joblib.load(model_path)

    @staticmethod
    def _build_engineered_features(client: ClientData) -> tuple[pd.DataFrame, float]:
        input_df = pd.DataFrame([client.model_dump()])
        dti = np.log(input_df["total_debt_outstanding"]) / input_df["income"]
        features = pd.DataFrame(
            {
                "income": input_df["income"],
                "years_employed": input_df["years_employed"],
                "fico_score": input_df["fico_score"],
                "dti": dti,
            }
        )
        return features, float(dti.iloc[0])

    @staticmethod
    def _build_local_xgb_input(client: ClientData) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "credit_lines_outstanding": LOCAL_FALLBACK_CREDIT_LINES_OUTSTANDING,
                    "loan_amt_outstanding": client.loan_amt_outstanding,
                    "total_debt_outstanding": client.total_debt_outstanding,
                    "income": client.income,
                    "years_employed": client.years_employed,
                    "fico_score": client.fico_score,
                }
            ]
        )

    def predict(self, client: ClientData) -> dict:
        _, dti = self._build_engineered_features(client)
        model_input = self._build_local_xgb_input(client)
        probability = self.model.predict_proba(model_input)[0][1]
        prediction = int(self.model.predict(model_input)[0])

        return {
            "status": "success",
            "default_probability": round(float(probability), 4),
            "prediction": "Default" if prediction == 1 else "Healthy",
            "risk_level": "High" if probability > 0.5 else "Low",
            "dti": round(dti, 6),
        }


class MlflowPredictor:
    def __init__(self, model_uri: str, preprocessor_uri: str | None = None) -> None:
        if mlflow is None:
            raise RuntimeError("MLflow support is not available in this environment.")
        self.model = mlflow.sklearn.load_model(model_uri)
        self.preprocessor = mlflow.sklearn.load_model(preprocessor_uri) if preprocessor_uri else None

    def predict(self, client: ClientData) -> dict:
        features, dti = LocalPredictor._build_engineered_features(client)
        expected_features = list(getattr(self.model, "feature_names_in_", features.columns.tolist()))
        model_input = features[expected_features]

        if self.preprocessor is not None:
            raw_input = CompatDataFrame([client.model_dump()])
            model_input = self.preprocessor.transform(raw_input)

        probability = self.model.predict_proba(model_input)[0][1]
        prediction = int(self.model.predict(model_input)[0])

        return {
            "status": "success",
            "default_probability": round(float(probability), 4),
            "prediction": "Default" if prediction == 1 else "Healthy",
            "risk_level": "High" if probability > 0.5 else "Low",
            "dti": round(dti, 6),
        }


def configure_mlflow() -> None:
    if mlflow is None:
        raise RuntimeError("MLflow support is not available in this environment.")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)


def resolve_local_model_path(model_source: str) -> str:
    if model_source.startswith("models:/m-"):
        model_id = model_source.removeprefix("models:/")
        candidates = sorted(BASE_DIR.glob(f"mlruns/*/models/{model_id}/artifacts"))
        if candidates:
            return str(candidates[-1])

    return model_source


def resolve_registered_model_uri(model_name: str, stage: str | None = None, version: str | None = None) -> str:
    if MlflowClient is None:
        raise RuntimeError("MLflow client support is not available in this environment.")

    client = MlflowClient()
    if version:
        model_version = client.get_model_version(name=model_name, version=version)
        return resolve_local_model_path(model_version.source)

    versions = list(client.search_model_versions(f"name = '{model_name}'"))
    if not versions:
        raise RuntimeError(f"No registered versions found for MLflow model '{model_name}'.")

    if stage:
        versions = [item for item in versions if item.current_stage == stage]
        if not versions:
            raise RuntimeError(f"No registered versions found for MLflow model '{model_name}' in stage '{stage}'.")

    latest = max(versions, key=lambda item: int(item.version))
    return resolve_local_model_path(latest.source)


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    source = os.getenv("MODEL_SOURCE", "auto").lower()
    if source in {"mlflow", "auto"}:
        try:
            configure_mlflow()
            model_uri = os.getenv("MLFLOW_MODEL_URI") or resolve_registered_model_uri(
                model_name=os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME),
                stage=os.getenv("MLFLOW_MODEL_STAGE", DEFAULT_MODEL_STAGE),
                version=os.getenv("MLFLOW_MODEL_VERSION"),
            )
            preprocessor_uri = None
            if os.getenv("MLFLOW_USE_PREPROCESSOR", "false").lower() == "true":
                preprocessor_uri = os.getenv("MLFLOW_PREPROCESSOR_URI") or resolve_registered_model_uri(
                    model_name=os.getenv("MLFLOW_PREPROCESSOR_NAME", DEFAULT_PREPROCESSOR_NAME),
                    version=os.getenv("MLFLOW_PREPROCESSOR_VERSION"),
                )
            return MlflowPredictor(model_uri=model_uri, preprocessor_uri=preprocessor_uri)
        except Exception:
            if source == "mlflow":
                raise
            logger.warning("MLflow registry unavailable, falling back to local model artifacts.", exc_info=True)

    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    return LocalPredictor(model_path=model_path)


@app.get("/")
def home() -> dict:
    return {"message": "Credit Scoring API is operational. Go to /docs to test."}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(client: ClientData) -> dict:
    predictor = get_predictor()
    return predictor.predict(client)
