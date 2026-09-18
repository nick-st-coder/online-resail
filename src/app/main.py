"""FastAPI app serving the customer-segmentation model.

Owns the REST API (``/health``, ``/predict``) and mounts the Gradio UI at
``/ui`` in the same process. All prediction logic lives in
:mod:`src.app.inference` — route handlers contain no business logic.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.app.inference import ModelLoadError, get_model, predict
from src.app.ui import build_ui

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Online Retail Customer Segmentation", version="0.1.0")


class PredictRequest(BaseModel):
    """Feature vector for a single customer, matching the training schema."""

    avg_order_value: float = Field(..., description="Average revenue per order")
    avg_items_per_order: float = Field(..., description="Average items per order")
    num_orders: int = Field(..., ge=0, description="Total number of orders")
    num_products: int = Field(..., ge=0, description="Distinct products bought")
    total_items: int = Field(..., ge=0, description="Total items purchased")
    avg_unit_price: float = Field(..., ge=0, description="Average unit price")
    total_revenue: float = Field(..., ge=0, description="Total revenue")
    sum_cancelled_orders: int = Field(..., ge=0, description="Cancelled orders")
    customer_lifetime_days: int = Field(
        ..., ge=0, description="Days between first and last purchase"
    )
    recency_days: int = Field(..., ge=0, description="Days since last purchase")
    purchase_frequency: float = Field(..., ge=0, description="Orders per month")


class PredictResponse(BaseModel):
    """Prediction result: the assigned cluster and its human-readable name."""

    cluster: int
    segment: str
    features: dict[str, float]


@app.on_event("startup")
def _load_model_on_startup() -> None:
    """Load the model at startup so failures surface immediately, not on first request."""
    try:
        get_model()
        logger.info("Model loaded successfully")
    except ModelLoadError as exc:
        logger.error("Model load failed at startup: %s", exc)
        raise


@app.get("/health")
def health() -> dict[str, str]:
    """Return 200 once the model is loaded; used by the Dockerfile HEALTHCHECK."""
    get_model()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    """Predict the customer segment for a feature vector."""
    try:
        result = predict(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    return PredictResponse(**result)


# Mount the Gradio UI in the same process/port.
build_ui(app)
