"""Smoke-test the registered MLflow model for CI.

Loads the model from the registry (``MODEL_URI`` or the default staging model)
and runs a small, realistic prediction. Fails if loading errors, the
prediction shape is wrong, or MLflow tracking creds/URI are missing.

Usage:
    uv run python scripts/validate_model.py
"""

from __future__ import annotations

import logging
import os
import sys

import mlflow

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS: list[str] = [
    "avg_order_value",
    "avg_items_per_order",
    "num_orders",
    "num_products",
    "total_items",
    "avg_unit_price",
    "total_revenue",
    "sum_cancelled_orders",
    "customer_lifetime_days",
    "recency_days",
    "purchase_frequency",
]

# A small, realistic sample payload (a mid-range customer).
SAMPLE_PAYLOAD: dict[str, float] = {
    "avg_order_value": 120.0,
    "avg_items_per_order": 8.0,
    "num_orders": 5,
    "num_products": 12,
    "total_items": 60,
    "avg_unit_price": 4.5,
    "total_revenue": 1_500.0,
    "sum_cancelled_orders": 1,
    "customer_lifetime_days": 180,
    "recency_days": 90,
    "purchase_frequency": 0.8,
}


def main() -> None:
    """Load the model and run a smoke prediction; exit non-zero on failure."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.error("MLFLOW_TRACKING_URI is not set")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    uri = os.environ.get("MODEL_URI", "models:/online-retail-segmentation/staging")

    logger.info("Loading model from %s", uri)
    try:
        model = mlflow.pyfunc.load_model(uri)
    except Exception as exc:  # noqa: BLE001 - surface load failures clearly
        logger.error("Failed to load model: %s", exc)
        sys.exit(1)

    import pandas as pd

    payload = pd.DataFrame([SAMPLE_PAYLOAD])
    try:
        prediction = model.predict(payload)
    except Exception as exc:  # noqa: BLE001 - surface prediction failures clearly
        logger.error("Prediction failed: %s", exc)
        sys.exit(1)

    if prediction.shape != (1,):
        logger.error("Unexpected prediction shape: %s", prediction.shape)
        sys.exit(1)

    logger.info("Smoke test passed. Cluster: %s", prediction[0])


if __name__ == "__main__":
    main()
