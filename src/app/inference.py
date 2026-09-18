"""Model inference for the serving app.

Loads the MLflow model once at startup and exposes a single ``predict``
function used by both the FastAPI route and the Gradio UI — there is
exactly one code path making predictions. The logged model is a
:class:`~src.models.segmentation_model.SegmentationModel` pyfunc wrapper
that applies log1p + scaling internally, so raw customer features are sent
directly.
"""

from __future__ import annotations

import logging
import os

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)

# Feature columns the model expects, in the order the scaler was fit on.
# Kept in sync with src/features/build_features.py and the feature_schema.json
# artifact logged with each training run.
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

# Human-readable segment names, matching the modeling notebook's summary.
SEGMENT_NAMES: dict[int, str] = {
    0: "Occasional bulk buyers",
    1: "Everyday value shoppers",
    2: "High-volume resellers",
}


class ModelLoadError(RuntimeError):
    """Raised when the MLflow model cannot be loaded at startup."""


def _model_uri() -> str:
    """Return the model URI from ``MODEL_URI`` or the default staging model."""
    return os.environ.get("MODEL_URI", "models:/online-retail-segmentation/staging")


def load_model() -> mlflow.pyfunc.PyFuncModel:
    """Load the MLflow model, failing fast with a clear error on failure.

    Raises:
        ModelLoadError: If ``MLFLOW_TRACKING_URI`` is unset or the model
            cannot be loaded.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ModelLoadError(
            "MLFLOW_TRACKING_URI is not set. Set it to your MLflow tracking "
            "server before starting the app."
        )
    mlflow.set_tracking_uri(tracking_uri)

    uri = _model_uri()
    logger.info("Loading model from %s", uri)
    try:
        return mlflow.pyfunc.load_model(uri)
    except Exception as exc:
        raise ModelLoadError(f"Failed to load model from {uri}: {exc}") from exc


def predict(features: dict[str, float]) -> dict:
    """Predict the segment for a single customer's feature vector.

    Args:
        features: Dict of feature name -> value, one entry per feature in
            :data:`FEATURE_COLS`.

    Returns:
        Dict with ``cluster`` (0-based int), ``segment`` (human-readable
        name), and ``features`` (the input echoed back).
    """
    missing = [c for c in FEATURE_COLS if c not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    row = pd.DataFrame([{c: features[c] for c in FEATURE_COLS}])
    prediction = get_model().predict(row)
    cluster = int(prediction[0])
    return {
        "cluster": cluster,
        "segment": SEGMENT_NAMES.get(cluster, f"Cluster {cluster + 1}"),
        "features": features,
    }


_model: mlflow.pyfunc.PyFuncModel | None = None


def get_model() -> mlflow.pyfunc.PyFuncModel:
    """Return the lazily-loaded model, loading it on first call."""
    global _model
    if _model is None:
        _model = load_model()
    return _model
