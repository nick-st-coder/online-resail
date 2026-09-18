"""Train the K-Means segmentation model and log it to MLflow.

Loads the processed transactions, builds customer-level features, fits the
final K-Means model, and logs the run (params, metrics, artifacts, model)
to the MLflow server configured via ``MLFLOW_TRACKING_URI``. Optionally
registers the model in the registry.

Usage:
    uv run python scripts/train_and_register.py [--k 3] [--stage dev] [--register]
"""

from __future__ import annotations

import argparse
import logging
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from src.features.build_features import (
    build_customer_features,
    log1p_transform,
    scale_features,
)
from src.models.cluster import fit_kmeans
from src.tracking import FEATURE_COLS, log_clustering_run

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and log K-Means to MLflow.")
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
        "--stage",
        type=str,
        default="dev",
        choices=["dev", "staging", "prod"],
        help="Registry stage tag for the run.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register the model in the MLflow registry.",
    )
    parser.add_argument("--seed", type=int, default=78, help="Random seed for K-Means.")
    parser.add_argument(
        "--n-init", type=int, default=10, help="Number of K-Means restarts."
    )
    args = parser.parse_args()

    logger.info("Loading %s", args.data)
    df = pd.read_csv(args.data)

    logger.info("Building customer-level features")
    cust = build_customer_features(df)
    logger.info("Customers: %d (was %d rows)", len(cust), len(df))

    cust_log = log1p_transform(cust)
    X_scaled, scaler = scale_features(cust_log)

    logger.info("Fitting K-Means with k=%d, seed=%d", args.k, args.seed)
    model = fit_kmeans(X_scaled, k=args.k, random_state=args.seed, n_init=args.n_init)

    run_id = log_clustering_run(
        model=model,
        X_scaled=X_scaled,
        scaler=scaler,
        cust=cust,
        feature_cols=FEATURE_COLS,
        k=args.k,
        random_state=args.seed,
        n_init=args.n_init,
        dataset_path=args.data,
        stage=args.stage,
        register=args.register,
    )
    logger.info("Done. Run ID: %s", run_id)


if __name__ == "__main__":
    main()
