from fastapi.testclient import TestClient
import pytest

from main import app, get_predictor


class DummyPredictor:
    def predict(self, client):
        return {
            "status": "success",
            "default_probability": 0.7,
            "prediction": "Default",
            "risk_level": "High",
            "dti": 0.230259,
            "lti": 0.213146,
        }


client = TestClient(app)


def test_predict_returns_expected_payload():
    app.dependency_overrides[get_predictor] = lambda: DummyPredictor()

    payload = {
        "loan_amt_outstanding": 5000,
        "total_debt_outstanding": 10000,
        "income": 40000,
        "years_employed": 5,
        "fico_score": 650,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "default_probability": 0.7,
        "prediction": "Default",
        "risk_level": "High",
        "dti": 0.230259,
        "lti": 0.213146,
    }


def test_predict_rejects_missing_required_fields():
    response = client.post(
        "/predict",
        json={
            "loan_amt_outstanding": 5000,
            "income": 40000,
            "years_employed": 5,
            "fico_score": 650,
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loan_amt_outstanding", 0),
        ("total_debt_outstanding", 0),
        ("income", 0),
        ("fico_score", 250),
    ],
)
def test_predict_rejects_invalid_numeric_ranges(field, value):
    payload = {
        "loan_amt_outstanding": 5000,
        "total_debt_outstanding": 10000,
        "income": 40000,
        "years_employed": 5,
        "fico_score": 650,
    }
    payload[field] = value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predictor_override_isolation():
    response = client.get("/")
    assert response.status_code == 200