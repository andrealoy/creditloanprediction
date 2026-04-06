from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

from main import app

client = TestClient(app)


@patch("main.get_model")
def test_predict(mock_get_model):
    # Mock du modèle
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    mock_model.predict.return_value = np.array([1])

    mock_get_model.return_value = mock_model

    # Données de test
    payload = {
        "credit_lines_outstanding": 2,
        "loan_amt_outstanding": 5000,
        "total_debt_outstanding": 10000,
        "income": 40000,
        "years_employed": 5,
        "fico_score": 650
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "default_probability" in data
    assert "prediction" in data
    assert "risk_level" in data