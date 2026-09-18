"""Tests for src/tracking/tracking.py.

Mocks MLflow — no real tracking server, no network, per the testing
conventions.
"""

from __future__ import annotations

import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.tracking.tracking import (
    REGISTERED_MODEL_NAME,
    get_tracking_uri,
    log_clustering_run,
)

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


def test_get_tracking_uri_raises_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails fast when MLFLOW_TRACKING_URI is missing."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        get_tracking_uri()


def test_get_tracking_uri_returns_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns the env value when set."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking:5000")
    assert get_tracking_uri() == "http://tracking:5000"


def test_log_clustering_run_logs_expected_artifacts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run logs params, metrics, schema, profile, and the model."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking:5000")

    rng = np.random.default_rng(42)
    X = rng.uniform(1, 100, size=(30, len(FEATURES)))
    df = pd.DataFrame(X, columns=FEATURES).assign(**{"Customer ID": range(30)})
    scaler = StandardScaler().fit(np.log1p(df[FEATURES]))
    X_scaled = scaler.transform(np.log1p(df[FEATURES]))
    model = KMeans(n_clusters=3, random_state=78, n_init=10).fit(X_scaled)

    dataset = tmp_path / "data.csv"
    dataset.write_text("x\n1\n")

    logged: dict[str, list] = {"dicts": []}

    def fake_log_dict(d, artifact_file):
        logged["dicts"].append((artifact_file, json.loads(json.dumps(d))))

    with (
        mock.patch("src.tracking.tracking.mlflow.set_tracking_uri"),
        mock.patch("src.tracking.tracking.mlflow.set_experiment"),
        mock.patch("src.tracking.tracking.mlflow.start_run") as start_run,
        mock.patch("src.tracking.tracking.mlflow.set_tags") as set_tags,
        mock.patch("src.tracking.tracking.mlflow.log_params") as log_params,
        mock.patch("src.tracking.tracking.mlflow.log_metrics") as log_metrics,
        mock.patch("src.tracking.tracking.mlflow.log_dict", side_effect=fake_log_dict),
        mock.patch("src.tracking.tracking.mlflow.pyfunc.log_model") as log_model,
        mock.patch("src.tracking.tracking.mlflow.tracking.MlflowClient"),
    ):
        start_run.return_value.__enter__.return_value.info.run_id = "run-123"
        run_id = log_clustering_run(
            model=model,
            X_scaled=X_scaled,
            scaler=scaler,
            cust=df,
            feature_cols=FEATURES,
            k=3,
            random_state=78,
            n_init=10,
            dataset_path=dataset,
            stage="staging",
            register=True,
        )

    assert run_id == "run-123"
    assert log_params.call_args.args[0]["k"] == 3
    assert log_params.call_args.args[0]["random_state"] == 78
    assert set(log_metrics.call_args.args[0]) == {
        "silhouette",
        "davies_bouldin",
        "inertia",
    }
    assert set(set_tags.call_args.args[0]) == {"stage", "git_sha", "dataset_version"}
    assert set_tags.call_args.args[0]["stage"] == "staging"
    assert log_model.call_args.kwargs["registered_model_name"] == REGISTERED_MODEL_NAME
    assert log_model.call_args.kwargs["artifact_path"] == "model"
    schema_files = [name for name, _ in logged["dicts"]]
    assert "feature_schema.json" in schema_files
    assert "cluster_profile.json" in schema_files
