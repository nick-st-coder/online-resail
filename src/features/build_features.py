"""Customer-level feature engineering for the Online Retail II dataset.

Turns the row-level transaction data into one row per customer with
RFM-style features, then applies the transforms the clustering pipeline
needs (log1p for skew, StandardScaler for scale).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Columns that are already customer-level aggregates in the processed data.
# They are constant within a customer, so "first" is safe.
CUSTOMER_COLS: list[str] = [
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

# Columns that need a semantically meaningful aggregation rather than "first".
_AGG_OVERRIDES: dict[str, str] = {
    "sum_cancelled_orders": "sum",
    "customer_lifetime_days": "max",
    "recency_days": "max",
}


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level transactions into one row per customer.

    Args:
        df: Row-level transaction data with a ``Customer ID`` column and the
            columns in :data:`CUSTOMER_COLS`.

    Returns:
        DataFrame with one row per customer (index reset), containing
        ``Customer ID`` plus the aggregated feature columns.
    """
    agg_map = {col: "first" for col in CUSTOMER_COLS}
    agg_map.update(_AGG_OVERRIDES)

    return df.groupby("Customer ID")[CUSTOMER_COLS].agg(agg_map).reset_index()


def log1p_transform(
    cust: pd.DataFrame, feature_cols: list[str] | None = None
) -> pd.DataFrame:
    """Apply ``log1p`` to the feature columns to tame right skew.

    Args:
        cust: Customer-level DataFrame (from :func:`build_customer_features`).
        feature_cols: Columns to transform. Defaults to all columns except
            ``Customer ID``.

    Returns:
        A new DataFrame with the same columns, feature columns log-transformed.
    """
    if feature_cols is None:
        feature_cols = [c for c in cust.columns if c != "Customer ID"]

    out = cust.copy()
    out[feature_cols] = np.log1p(out[feature_cols])
    return out


def scale_features(
    cust_log: pd.DataFrame, feature_cols: list[str] | None = None
) -> tuple[np.ndarray, StandardScaler]:
    """Standardize the log-transformed features.

    Args:
        cust_log: Customer-level DataFrame with log-transformed features.
        feature_cols: Columns to scale. Defaults to all columns except
            ``Customer ID``.

    Returns:
        Tuple of ``(X_scaled, scaler)`` where ``X_scaled`` is the standardized
        feature matrix and ``scaler`` is the fitted ``StandardScaler``.
    """
    if feature_cols is None:
        feature_cols = [c for c in cust_log.columns if c != "Customer ID"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cust_log[feature_cols])
    return X_scaled, scaler
