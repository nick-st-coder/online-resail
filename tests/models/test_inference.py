"""Tests for src/app/inference.py.

Mocks the model file loading — no real model, no filesystem dependency.
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


def test_load_model_raises_when_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails fast when the model file does not exist."""
    monkeypatch.setenv("MODEL_PATH", "does/not/exist.pkl")
    with pytest.raises(ModelLoadError, match="Model file not found"):
        load_model()


def test_load_model_raises_on_load_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Surfaces load failures as ModelLoadError."""
    model_file = tmp_path / "model.pkl"
    model_file.write_bytes(b"not a pickle")
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    with pytest.raises(ModelLoadError, match="Failed to load model"):
        load_model()


def test_predict_returns_cluster_and_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Predict returns the cluster, segment name, and echoed features."""
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


def test_predict_raises_on_missing_feature() -> None:
    """Missing features raise ValueError listing them."""
    features = _valid_features()
    del features["recency_days"]
    with pytest.raises(ValueError, match="recency_days"):
        predict(features)
