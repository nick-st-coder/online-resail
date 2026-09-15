"""Tests for src/models/cluster.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from src.models.cluster import (
    cluster_profile,
    fit_kmeans,
    gmm_bic_aic,
    k_selection_sweep,
    stability_check,
)


@pytest.fixture
def blobs() -> tuple[np.ndarray, np.ndarray]:
    """Well-separated 3-cluster data for deterministic tests."""
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    return X, y


def test_fit_kmeans_returns_fitted_model(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = blobs
    model = fit_kmeans(X, k=3)
    assert model.n_clusters == 3
    assert model.labels_.shape == (len(X),)


def test_k_selection_sweep_shape(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = blobs
    sweep = k_selection_sweep(X, k_range=range(2, 5))
    assert list(sweep.index) == [2, 3, 4]
    assert {"inertia", "silhouette", "davies_bouldin"} <= set(sweep.columns)


def test_k_selection_sweep_inertia_decreases(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = blobs
    sweep = k_selection_sweep(X, k_range=range(2, 5))
    assert sweep["inertia"].is_monotonic_decreasing


def test_stability_check_returns_one_row_per_seed(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = blobs
    result = stability_check(X, k=3, seeds=[0, 1, 2])
    assert len(result) == 3
    assert list(result["seed"]) == [0, 1, 2]
    assert {"silhouette", "davies_bouldin", "inertia"} <= set(result.columns)


def test_gmm_bic_aic_shape(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = blobs
    result = gmm_bic_aic(X, k_range=range(2, 5))
    assert list(result.index) == [2, 3, 4]
    assert {"bic", "aic"} <= set(result.columns)


def test_cluster_profile_counts_customers() -> None:
    cust = pd.DataFrame(
        {
            "Customer ID": [1, 2, 3],
            "total_revenue": [100.0, 200.0, 300.0],
            "num_orders": [1, 2, 3],
        }
    )
    labels = np.array([0, 0, 1])
    profile = cluster_profile(cust, labels, feature_cols=["total_revenue", "num_orders"])
    assert profile.loc[0, "n_customers"] == 2
    assert profile.loc[1, "n_customers"] == 1
    assert profile.loc[0, "avg_total_revenue"] == 150.0
