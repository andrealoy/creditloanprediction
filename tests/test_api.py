from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Scoring API is operational. Go to /docs to test."}

def test_prediction():
    # Testing with a dummy client profile
    payload = {
        "credit_lines_outstanding": 1,
        "loan_amt_outstanding": 1000,
        "total_debt_outstanding": 5000,
        "income": 50000,
        "years_employed": 5,
        "fico_score": 700
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "default_probability" in response.json()