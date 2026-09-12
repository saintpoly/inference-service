from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_check():
    """The root route should confirm the service is running."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_customer():
    """A complete, valid customer body should return a probability and label."""
    payload = {
        "age": 34,
        "annual_income": 65000,
        "browsing_time_minutes": 12.5,
        "previous_purchases": 3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["purchase_probability"] <= 1.0
    assert body["prediction"] in ("likely to purchase", "unlikely to purchase")


def test_predict_high_engagement_customer():
    """A highly engaged customer should be predicted as likely to purchase."""
    payload = {
        "age": 28,
        "annual_income": 95000,
        "browsing_time_minutes": 45.0,
        "previous_purchases": 12,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == "likely to purchase"


def test_predict_missing_fields_returns_422():
    """Missing required fields should be rejected with a 422 validation error."""
    response = client.post("/predict", json={"annual_income": 65000})
    assert response.status_code == 422
    missing_fields = [err["loc"][-1] for err in response.json()["detail"]]
    assert "age" in missing_fields


def test_predict_negative_age_rejected():
    """Field constraints should reject an impossible negative age."""
    payload = {
        "age": -5,
        "annual_income": 65000,
        "browsing_time_minutes": 12.5,
        "previous_purchases": 3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
