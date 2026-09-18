"""Tests for src/app/inference.py.

Mocks the MLflow model — no real server, no network.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from src.app.inference import ModelLoadError, load_model, predict


def _valid_features() -> dict[str, float]:
    """A complete, realistic feature vector."""
    return {
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


def test_load_model_raises_without_tracking_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails fast at startup when MLFLOW_TRACKING_URI is unset."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(ModelLoadError, match="MLFLOW_TRACKING_URI"):
        load_model()


def test_load_model_raises_on_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surfaces load failures as ModelLoadError."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking:5000")
    with (
        mock.patch(
            "src.app.inference.mlflow.pyfunc.load_model",
            side_effect=RuntimeError("boom"),
        ),
        pytest.raises(ModelLoadError, match="boom"),
    ):
        load_model()


def test_predict_returns_cluster_and_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Predict returns the cluster, segment name, and echoed features."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking:5000")
    fake_model = mock.Mock()
    fake_model.predict.return_value = np.array([2])
    with (
        mock.patch("src.app.inference.load_model", return_value=fake_model),
        mock.patch("src.app.inference._model", fake_model),
    ):
        result = predict(_valid_features())

    assert result["cluster"] == 2
    assert result["segment"] == "High-volume resellers"
    assert result["features"] == _valid_features()


def test_predict_raises_on_missing_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing features raise ValueError listing them."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking:5000")
    features = _valid_features()
    del features["recency_days"]
    with pytest.raises(ValueError, match="recency_days"):
        predict(features)
