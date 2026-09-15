"""Clustering models and evaluation for customer segmentation.

Provides K-Means fitting with k-selection (elbow / silhouette /
Davies-Bouldin), seed-stability checks, and a Gaussian Mixture comparison
with BIC/AIC for choosing the number of clusters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture


def fit_kmeans(X: np.ndarray, k: int, random_state: int = 78, n_init: int = 10) -> KMeans:
    """Fit K-Means with a fixed seed and return the fitted estimator.

    Args:
        X: Scaled feature matrix.
        k: Number of clusters.
        random_state: Seed for reproducibility.
        n_init: Number of K-Means restarts.

    Returns:
        Fitted ``KMeans`` estimator.
    """
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    model.fit(X)
    return model


def k_selection_sweep(
    X: np.ndarray,
    k_range: range = range(2, 6),
    random_state: int = 78,
    n_init: int = 10,
) -> pd.DataFrame:
    """Sweep k and return inertia, silhouette, and Davies-Bouldin per k.

    Args:
        X: Scaled feature matrix.
        k_range: Values of k to evaluate.
        random_state: Seed for reproducibility.
        n_init: Number of K-Means restarts per k.

    Returns:
        DataFrame indexed by k with columns ``inertia``, ``silhouette``,
        ``davies_bouldin``.
    """
    results = []
    for k in k_range:
        model = fit_kmeans(X, k, random_state=random_state, n_init=n_init)
        labels = model.labels_
        results.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
            }
        )
    return pd.DataFrame(results).set_index("k")


def stability_check(
    X: np.ndarray,
    k: int,
    seeds: list[int] | None = None,
    n_init: int = 10,
) -> pd.DataFrame:
    """Fit K-Means across multiple seeds and report the silhouette spread.

    K-Means is initialization-sensitive; a large spread across seeds means
    the partition is not stable.

    Args:
        X: Scaled feature matrix.
        k: Number of clusters.
        seeds: Random seeds to try. Defaults to ``range(10)``.
        n_init: Number of K-Means restarts per seed.

    Returns:
        DataFrame with one row per seed: ``seed``, ``silhouette``,
        ``davies_bouldin``, ``inertia``.
    """
    if seeds is None:
        seeds = list(range(10))

    rows = []
    for seed in seeds:
        model = fit_kmeans(X, k, random_state=seed, n_init=n_init)
        labels = model.labels_
        rows.append(
            {
                "seed": seed,
                "silhouette": silhouette_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
                "inertia": model.inertia_,
            }
        )
    return pd.DataFrame(rows)


def gmm_bic_aic(X: np.ndarray, k_range: range = range(2, 6), random_state: int = 78) -> pd.DataFrame:
    """Fit Gaussian Mixture Models and return BIC/AIC per k.

    BIC/AIC are principled model-selection criteria for the number of
    components; lower is better.

    Args:
        X: Scaled feature matrix.
        k_range: Number of components to evaluate.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame indexed by k with columns ``bic`` and ``aic``.
    """
    results = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=random_state)
        gmm.fit(X)
        results.append({"k": k, "bic": gmm.bic(X), "aic": gmm.aic(X)})
    return pd.DataFrame(results).set_index("k")


def cluster_profile(cust: pd.DataFrame, labels: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    """Profile each cluster on the original (untransformed) scale.

    Args:
        cust: Customer-level DataFrame (untransformed).
        labels: Cluster labels aligned with ``cust`` rows.
        feature_cols: Feature columns to average per cluster.

    Returns:
        DataFrame indexed by cluster with ``n_customers`` and the mean of
        each feature column, rounded to 2 decimals.
    """
    summary = cust.assign(cluster=labels).groupby("cluster").agg(
        n_customers=("Customer ID", "count"),
        **{f"avg_{col}": (col, "mean") for col in feature_cols},
    )
    return summary.round(2)
