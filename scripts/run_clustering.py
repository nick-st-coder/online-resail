"""End-to-end customer segmentation pipeline.

Loads the processed transactions, builds customer-level features, runs the
K-Means k-selection sweep, a seed-stability check, and a GMM BIC/AIC
comparison, then prints the chosen model's cluster profile. Optionally
saves the fitted model (scaler + K-Means) for the serving app.

Usage:
    uv run python scripts/run_clustering.py [--k 3] [--data data/processed/online_retail.csv] [--save-model]
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from src.features.build_features import (
    build_customer_features,
    log1p_transform,
    scale_features,
)
from src.models.cluster import (
    cluster_profile,
    fit_kmeans,
    gmm_bic_aic,
    k_selection_sweep,
    stability_check,
)
from src.models.segmentation_model import SegmentationModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CUSTOMER_COLS = [
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run customer segmentation pipeline.")
    parser.add_argument(
        "--k", type=int, default=3, help="Number of clusters for the final model."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/online_retail.csv",
        help="Path to processed transactions.",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        nargs="?",
        const="artifacts/segmentation_model.pkl",
        default=None,
        help="Save the fitted model (scaler + K-Means) to this path.",
    )
    args = parser.parse_args()

    logger.info("Loading %s", args.data)
    df = pd.read_csv(args.data)

    logger.info("Building customer-level features")
    cust = build_customer_features(df)
    logger.info("Customers: %d (was %d rows)", len(cust), len(df))

    cust_log = log1p_transform(cust)
    X_scaled, scaler = scale_features(cust_log)

    logger.info("K-Means k-selection sweep (k=2..5)")
    sweep = k_selection_sweep(X_scaled)
    print(sweep.round(4))

    logger.info("K-Means stability check (k=%d, 10 seeds)", args.k)
    stability = stability_check(X_scaled, k=args.k)
    print(stability.round(4))
    logger.info(
        "Silhouette across seeds: mean=%.4f std=%.4f min=%.4f max=%.4f",
        stability["silhouette"].mean(),
        stability["silhouette"].std(),
        stability["silhouette"].min(),
        stability["silhouette"].max(),
    )

    logger.info("GMM BIC/AIC comparison (k=2..5)")
    bic_aic = gmm_bic_aic(X_scaled)
    print(bic_aic.round(2))

    logger.info("Fitting final K-Means with k=%d", args.k)
    model = fit_kmeans(X_scaled, k=args.k)
    profile = cluster_profile(cust, model.labels_, CUSTOMER_COLS)
    print(profile)

    if args.save_model:
        path = Path(args.save_model)
        path.parent.mkdir(parents=True, exist_ok=True)
        wrapper = SegmentationModel(scaler=scaler, kmeans=model)
        with path.open("wb") as fh:
            pickle.dump(wrapper, fh)
        logger.info("Saved model to %s", path)


if __name__ == "__main__":
    main()
