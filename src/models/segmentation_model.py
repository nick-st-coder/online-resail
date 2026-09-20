"""Segmentation model wrapping the full transform chain.

The K-Means model alone expects log1p + standardized features. This wrapper
encapsulates the whole transform chain (log1p → StandardScaler → KMeans.predict)
so serving code can send raw customer features and get a cluster label back,
without re-implementing preprocessing in the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class SegmentationModel:
    """Wrapper around ``StandardScaler`` + ``KMeans``."""

    def __init__(self, scaler: StandardScaler, kmeans: KMeans) -> None:
        """Store the fitted scaler and K-Means model.

        Args:
            scaler: Fitted ``StandardScaler`` from training.
            kmeans: Fitted ``KMeans`` model.
        """
        self.scaler = scaler
        self.kmeans = kmeans

    def predict(self, model_input) -> np.ndarray:
        """Transform raw features and return cluster labels.

        Args:
            model_input: Raw customer features (one row per customer), in
                the same column order the scaler was fit on.

        Returns:
            Array of cluster labels (0-based ints).
        """
        if isinstance(model_input, np.ndarray):
            df = pd.DataFrame(model_input, columns=self.scaler.feature_names_in_)
        else:
            df = pd.DataFrame(model_input)
        df = df[self.scaler.feature_names_in_]
        X_log = np.log1p(df)
        X_scaled = self.scaler.transform(X_log)
        return self.kmeans.predict(X_scaled)
