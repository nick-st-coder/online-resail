"""Tests for src/models/segmentation_model.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.models.segmentation_model import SegmentationModel

FEATURES = [
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


@pytest.fixture
def fitted_wrapper() -> SegmentationModel:
    """A fitted wrapper on tiny synthetic data."""
    rng = np.random.default_rng(42)
    X = rng.uniform(1, 100, size=(50, len(FEATURES)))
    df = pd.DataFrame(X, columns=FEATURES)
    scaler = StandardScaler().fit(np.log1p(df))
    X_scaled = scaler.transform(np.log1p(df))
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
    return SegmentationModel(scaler=scaler, kmeans=kmeans)


def test_predict_returns_labels_for_dataframe(
    fitted_wrapper: SegmentationModel,
) -> None:
    """Predict on a DataFrame returns one label per row."""
    row = pd.DataFrame([{f: 10.0 for f in FEATURES}])
    labels = fitted_wrapper.predict(None, row)
    assert labels.shape == (1,)
    assert labels.dtype.kind == "i"


def test_predict_accepts_ndarray(fitted_wrapper: SegmentationModel) -> None:
    """Predict on an ndarray works too."""
    arr = np.full((2, len(FEATURES)), 10.0)
    labels = fitted_wrapper.predict(None, arr)
    assert labels.shape == (2,)


def test_predict_reorders_columns_to_scaler_order(
    fitted_wrapper: SegmentationModel,
) -> None:
    """Column order in the input doesn't matter; the scaler's order wins."""
    shuffled = list(reversed(FEATURES))
    row = pd.DataFrame([{f: 10.0 for f in shuffled}], columns=shuffled)
    labels = fitted_wrapper.predict(None, row)
    assert labels.shape == (1,)
