"""MLflow tracking for the customer-segmentation training pipeline.

Implements the conventions in ``.github/instructions/mlflow.instructions.md``:

- ``MLFLOW_TRACKING_URI`` is read from the environment and required — the
  module fails fast rather than silently writing to a local ``mlruns/``.
- One experiment per project/task: ``online-retail-customer-segmentation``.
- Run names are ``<model_type>_<YYYYMMDD_HHMMSS>``.
- Every run sets ``stage``, ``git_sha``, and ``dataset_version`` tags.
- The model is logged with an inferred signature and input example; the
  feature schema is logged as a JSON artifact for downstream serving code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.models.segmentation_model import SegmentationModel

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "online-retail-customer-segmentation"
REGISTERED_MODEL_NAME = "online-retail-segmentation"

# Feature columns the clustering model was trained on. Kept in sync with
# ``src/features/build_features.py``; the serving app validates against the
# schema artifact logged with each run.
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


def get_tracking_uri() -> str:
    """Return ``MLFLOW_TRACKING_URI`` or raise if unset.

    Raises:
        RuntimeError: If ``MLFLOW_TRACKING_URI`` is not set in the
            environment. Tracking must be explicit, never a silent local
            fallback.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. Set it to your MLflow tracking "
            "server (e.g. http://localhost:5000) before running training."
        )
    return uri


def _git_sha() -> str:
    """Return the short commit SHA, or ``dirty`` if it can't be determined."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "dirty"


def _dataset_version(path: str | Path) -> str:
    """Return a short content hash of the dataset file as its version id."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _run_name(model_type: str) -> str:
    """Generate a run name like ``kmeans_20260918_143000``."""
    now = datetime.now(UTC)
    return f"{model_type}_{now:%Y%m%d_%H%M%S}"


def log_clustering_run(
    *,
    model: KMeans,
    X_scaled: np.ndarray,
    scaler: StandardScaler,
    cust: pd.DataFrame,
    feature_cols: list[str],
    k: int,
    random_state: int,
    n_init: int,
    dataset_path: str | Path,
    stage: str = "dev",
    register: bool = False,
) -> str:
    """Train-log a K-Means run to MLflow and optionally register the model.

    Logs params (k, seed, n_init), metrics (silhouette, Davies-Bouldin,
    inertia), the feature schema as JSON, the cluster profile, and the model
    itself (a :class:`SegmentationModel` pyfunc wrapper) with an inferred
    signature and input example. Returns the run ID.

    Args:
        model: Fitted K-Means model.
        X_scaled: Scaled feature matrix the model was fit on.
        scaler: Fitted ``StandardScaler`` used to scale the features.
        cust: Customer-level DataFrame (untransformed) for the profile.
        feature_cols: Feature columns used for clustering.
        k: Number of clusters.
        random_state: Seed used for fitting.
        n_init: Number of K-Means restarts.
        dataset_path: Path to the dataset used, for the ``dataset_version`` tag.
        stage: Registry stage for the run tag (``dev``/``staging``/``prod``).
        register: Whether to register the model in the registry.

    Returns:
        The MLflow run ID.
    """
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    labels = model.labels_
    metrics = {
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
        "inertia": float(model.inertia_),
    }

    with mlflow.start_run(run_name=_run_name("kmeans")) as run:
        # Tags
        mlflow.set_tags(
            {
                "stage": stage,
                "git_sha": _git_sha(),
                "dataset_version": _dataset_version(dataset_path),
            }
        )

        # Params
        mlflow.log_params(
            {
                "k": k,
                "random_state": random_state,
                "n_init": n_init,
                "feature_set": "rfm-behavioral-v1",
                "feature_cols": json.dumps(feature_cols),
            }
        )

        # Metrics
        mlflow.log_metrics(metrics)

        # Feature schema artifact (serving app validates against this)
        schema = {
            "feature_cols": feature_cols,
            "n_features": len(feature_cols),
            "model_type": "kmeans",
            "k": k,
        }
        mlflow.log_dict(schema, "feature_schema.json")

        # Cluster profile artifact
        profile = (
            cust.assign(cluster=labels)
            .groupby("cluster")
            .agg(
                n_customers=("Customer ID", "count"),
                **{f"avg_{col}": (col, "mean") for col in feature_cols},
            )
            .round(2)
        )
        mlflow.log_dict(profile.to_dict(), "cluster_profile.json")

        # Model with signature + input example. The wrapper takes raw
        # features and applies log1p + scaling internally, so the signature
        # reflects what serving code sends.
        wrapper = SegmentationModel(scaler=scaler, kmeans=model)
        signature = infer_signature(cust[feature_cols].head(5), labels[:5])
        input_example = cust[feature_cols].head(5)
        mlflow.pyfunc.log_model(
            python_model=wrapper,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME if register else None,
        )

        run_id = run.info.run_id
        logger.info("Logged run %s (silhouette=%.4f)", run_id, metrics["silhouette"])
        if register:
            client = mlflow.tracking.MlflowClient()
            latest = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])
            if not latest:
                raise RuntimeError(
                    f"Model {REGISTERED_MODEL_NAME} was not registered by log_model"
                )
            version = latest[0].version
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME,
                version=version,
                stage=stage.upper(),
                archive_existing_versions=True,
            )
            logger.info(
                "Registered model %s version %s at stage %s",
                REGISTERED_MODEL_NAME,
                version,
                stage,
            )
        return run_id
