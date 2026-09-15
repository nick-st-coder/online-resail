"""Tests for src/features/build_features.py."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    CUSTOMER_COLS,
    build_customer_features,
    log1p_transform,
    scale_features,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small synthetic transaction frame with two customers."""
    return pd.DataFrame(
        {
            "Customer ID": [1, 1, 2],
            "avg_order_value": [10.0, 10.0, 20.0],
            "avg_items_per_order": [2.0, 2.0, 5.0],
            "num_orders": [3, 3, 1],
            "num_products": [4, 4, 2],
            "total_items": [6, 6, 5],
            "avg_unit_price": [5.0, 5.0, 4.0],
            "total_revenue": [30.0, 30.0, 20.0],
            "sum_cancelled_orders": [1, 2, 0],
            "customer_lifetime_days": [100, 200, 50],
            "recency_days": [10, 20, 5],
            "purchase_frequency": [0.5, 0.5, 0.2],
        }
    )


def test_build_customer_features_one_row_per_customer(sample_df: pd.DataFrame) -> None:
    result = build_customer_features(sample_df)
    assert len(result) == 2
    assert set(result["Customer ID"]) == {1, 2}


def test_build_customer_features_sums_cancelled_orders(sample_df: pd.DataFrame) -> None:
    result = build_customer_features(sample_df)
    cust1 = result[result["Customer ID"] == 1].iloc[0]
    assert cust1["sum_cancelled_orders"] == 3  # 1 + 2


def test_build_customer_features_maxes_recency(sample_df: pd.DataFrame) -> None:
    result = build_customer_features(sample_df)
    cust1 = result[result["Customer ID"] == 1].iloc[0]
    assert cust1["recency_days"] == 20  # max(10, 20)


def test_log1p_transform_handles_zeros() -> None:
    df = pd.DataFrame({"Customer ID": [1], "total_revenue": [0.0], "num_orders": [0]})
    out = log1p_transform(df)
    assert out.loc[0, "total_revenue"] == 0.0  # log1p(0) == 0
    assert out.loc[0, "num_orders"] == 0.0


def test_log1p_transform_matches_numpy(sample_df: pd.DataFrame) -> None:
    out = log1p_transform(sample_df)
    assert np.allclose(out["total_revenue"], np.log1p(sample_df["total_revenue"]))


def test_scale_features_returns_scaled_matrix(sample_df: pd.DataFrame) -> None:
    cust = build_customer_features(sample_df)
    cust_log = log1p_transform(cust)
    X, scaler = scale_features(cust_log)
    assert X.shape == (2, len(CUSTOMER_COLS))
    assert hasattr(scaler, "mean_")
    # Standardized columns have ~zero mean and ~unit variance
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(X.std(axis=0), 1.0, atol=1e-10)
