"""Model inference for the serving app.

Loads the segmentation model once at startup and exposes a single ``predict``
function used by both the FastAPI route and the Gradio UI — there is
exactly one code path making predictions. The model is a
:class:`~src.models.segmentation_model.SegmentationModel` wrapper that
applies log1p + scaling internally, so raw customer features are sent
directly.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import pandas as pd

from src.models.segmentation_model import SegmentationModel

logger = logging.getLogger(__name__)

# Feature columns the model expects, in the order the scaler was fit on.
# Kept in sync with src/features/build_features.py.
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
    """Raised when the segmentation model cannot be loaded at startup."""


def _model_path() -> Path:
    """Return the model path from ``MODEL_PATH`` or the default location."""
    return Path(os.environ.get("MODEL_PATH", "artifacts/segmentation_model.pkl"))


def load_model() -> SegmentationModel:
    """Load the segmentation model, failing fast with a clear error on failure.

    Raises:
        ModelLoadError: If the model file is missing or cannot be loaded.
    """
    path = _model_path()
    logger.info("Loading model from %s", path)
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except FileNotFoundError as exc:
        raise ModelLoadError(
            f"Model file not found at {path}. Train it first with "
            "`uv run python scripts/run_clustering.py --save-model`."
        ) from exc
    except Exception as exc:
        raise ModelLoadError(f"Failed to load model from {path}: {exc}") from exc


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


_model: SegmentationModel | None = None


def get_model() -> SegmentationModel:
    """Return the lazily-loaded model, loading it on first call."""
    global _model
    if _model is None:
        _model = load_model()
    return _model
