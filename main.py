import math

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Purchase Probability Prediction API",
    description=(
        "Inference service built for the InsightEdge AI Services Team. "
        "Exposes the trained purchase-probability model through a /predict "
        "endpoint so web applications can score customers in real time."
    ),
    version="1.0.0",
)


class CustomerFeatures(BaseModel):
    """Input features describing a customer for inference."""

    age: int = Field(..., gt=0, description="Customer age in years")
    annual_income: float = Field(..., ge=0, description="Annual income in USD")
    browsing_time_minutes: float = Field(..., ge=0, description="Minutes spent browsing the site")
    previous_purchases: int = Field(..., ge=0, description="Number of past purchases")


class PredictionResponse(BaseModel):
    """Response returned by the /predict endpoint."""

    purchase_probability: float = Field(..., description="Model probability between 0 and 1")
    prediction: str = Field(..., description="Human-readable purchase label")


@app.get("/", summary="Service health check", tags=["Status"])
def root():
    """Return a simple health-check message confirming the service is running."""
    return {"status": "ok", "message": "Purchase probability inference service is running"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict purchase probability",
    tags=["Prediction"],
)
def predict(data: CustomerFeatures):
    """
    Predict the probability that a customer will make a purchase.

    Accepts customer attributes (age, annual income, browsing time, and
    previous purchases), runs the model's inference logic, and returns the
    purchase probability along with a readable prediction label.
    """
    score = (
        0.02 * data.browsing_time_minutes
        + 0.15 * data.previous_purchases
        + 0.00001 * data.annual_income
        - 0.001 * data.age
        - 0.5
    )
    probability = 1 / (1 + math.exp(-score))
    return PredictionResponse(
        purchase_probability=round(probability, 4),
        prediction="likely to purchase" if probability >= 0.5 else "unlikely to purchase",
    )
