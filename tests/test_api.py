
import math

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Sample patient data
patient = {
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001,
}


# Test 1: Check health endpoint
def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "v0.1",
    }


# Test 2: Check successful prediction
def test_prediction():
    response = client.post("/predict", json=patient)

    assert response.status_code == 200

    prediction = response.json()["prediction"]

    assert isinstance(prediction, (int, float))
    assert math.isfinite(prediction)


# Test 3: Check missing feature
def test_missing_feature():
    invalid_patient = patient.copy()
    del invalid_patient["s6"]

    response = client.post(
        "/predict", json=invalid_patient
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["loc"][-1] == "s6"
        for error in errors
    )
