"""MLflow tracking helpers for the customer-segmentation project.

Conventions live in ``.github/instructions/mlflow.instructions.md``:
experiment naming, run naming, required tags/params/metrics, artifact
requirements, and registry stage transitions.
"""

from src.tracking.tracking import (
    EXPERIMENT_NAME,
    FEATURE_COLS,
    REGISTERED_MODEL_NAME,
    get_tracking_uri,
    log_clustering_run,
)

__all__ = [
    "EXPERIMENT_NAME",
    "FEATURE_COLS",
    "REGISTERED_MODEL_NAME",
    "get_tracking_uri",
    "log_clustering_run",
]
