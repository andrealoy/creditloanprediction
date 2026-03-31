import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np

# On importe l'instance app depuis ton fichier principal (supposé nommé main.py)
from main import app

client = TestClient(app)

## 1. Tests de la route racine
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Scoring API is operational. Go to /docs to test."}

## 2. Tests de la route /predict
def test_predict_success():
    """
    Test d'une prédiction réussie avec des données valides.
    On utilise un mock pour éviter de charger le vrai modèle lourd en mémoire 
    ou si le fichier .pkl est absent durant les tests CI.
    """
    payload = {
        "credit_lines_outstanding": 2,
        "loan_amt_outstanding": 5000.0,
        "total_debt_outstanding": 15000.0,
        "income": 50000.0,
        "years_employed": 5,
        "fico_score": 700
    }
    
    # On mocke la méthode predict et predict_proba du modèle chargé dans main.py
    with patch('main.model') as mock_model:
        # Simule un retour : [probabilité_0, probabilité_1]
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        # Simule une prédiction (0 pour Healthy)
        mock_model.predict.return_value = np.array([0])
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["prediction"] == "Healthy"
        assert data["risk_level"] == "Low"
        assert data["default_probability"] == 0.2

def test_predict_invalid_data():
    """Vérifie que l'API renvoie une erreur 422 si les données sont incorrectes."""
    payload = {
        "credit_lines_outstanding": "beaucoup", # Devrait être un int
        "fico_score": 700
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_high_risk():
    """Test le cas où le modèle prédit un risque élevé."""
    payload = {
        "credit_lines_outstanding": 10,
        "loan_amt_outstanding": 100000.0,
        "total_debt_outstanding": 200000.0,
        "income": 20000.0,
        "years_employed": 1,
        "fico_score": 400
    }
    
    with patch('main.model') as mock_model:
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
        mock_model.predict.return_value = np.array([1])
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "Default"
        assert data["risk_level"] == "High"
        assert data["default_probability"] == 0.9