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

def test_health_check(client):
    """Ensure the API is alive"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Sentinel-Fin" in response.json()["system"]

# --- INTEGRATED SCENARIOS VIA PARAMETERIZATION ---
@pytest.mark.parametrize(
    "payload_type, payload, expected_verdict",
    [
        (
            "reasonable_amount",
            {
                "step": 11,
                "type": "TRANSFER",
                "amount": 1000.0,
                "nameOrig": "C123456",
                "nameDest": "M654321"
            },
            None  # None means we let the ML model decide freely without forcing an answer key
        ),
        (
            "large_drain_amount",
            {
                "step": 11,
                "type": "TRANSFER",
                "amount": 950000.0,         # Triggers HEURISTIC_AMOUNT_LIMIT
                "nameOrig": "C123456",
                "nameDest": "M654321",
                "oldbalanceOrg": 1000000.0, # Triggers HEURISTIC_DRAIN_RATIO (95% drain)
                "newbalanceOrig": 50000.0
            },
            "FLAGGED"  # Our answer key explicitly expects the heuristic safety net to catch this
        )
    ]
)

def test_fraud_prediction_schema(client, payload_type, payload, expected_verdict):
    """Ensure the API returns the correct JSON structure and ML logic loads"""
    response = client.post("/predict", json=payload)
    data = response.json()
        
    # This will now be 200 because lifespan loaded the model
    assert response.status_code == 200
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "verdict" in data
    assert "reasoning" in data
    assert isinstance(data["reasoning"], list)

    if expected_verdict is not None:
        assert data["verdict"] == expected_verdict
        # Verify our custom AML reasoning warnings were appended to the audit trail
        assert any("Heuristic Alert" in r or "AML Warning" in r for r in data["reasoning"])

def test_invalid_data_handling(client):
    """Ensure the API rejects bad data (Industry-ready apps must be strict)"""
    payload = {
        "step": "NotAnInteger",
        "type": "TRANSFER",
        "amount": "ONE MILLION DOLLARS",
        "nameOrig": "C111",
        "nameDest": "M222"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422