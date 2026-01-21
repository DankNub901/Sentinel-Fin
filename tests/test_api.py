import pytest
from fastapi.testclient import TestClient
from src.api.main import app

# We remove the global 'client = TestClient(app)' because we want 
# to trigger the lifespan for the ML-dependent tests.

@pytest.fixture
def client():
    # This 'with' block handles the startup/shutdown for every test that uses it
    with TestClient(app) as c:
        yield c

def test_health_check():
    """Ensure the API is alive"""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Sentinel-Fin" in response.json()["system"]

def test_fraud_prediction_schema():
    """Ensure the API returns the correct JSON structure and ML logic loads"""
    with TestClient(app) as client:
        payload = {
            "amount": 1000.0,
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 0.0,
            "type_encoded": 4 # TRANSFER
        }
        response = client.post("/predict", json=payload)
        data = response.json()
        
        # This will now be 200 because lifespan loaded the model
        assert response.status_code == 200
        assert "is_fraud" in data
        assert "reasoning" in data
        assert isinstance(data["reasoning"], list)

def test_invalid_data_handling():
    """Ensure the API rejects bad data (Industry-ready apps must be strict)"""
    with TestClient(app) as client:
        payload = {"amount": "ONE MILLION DOLLARS"} # String instead of float
        response = client.post("/predict", json=payload)
        assert response.status_code == 422 # Unprocessable Entity