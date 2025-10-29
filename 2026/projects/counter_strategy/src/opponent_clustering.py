"""
Opponent clustering for power-play counter strategy.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_csv  # noqa: E402


FEATURE_COLUMNS = [
    "avg_score_gain",
    "three_plus_rate",
    "blank_rate",
    "steal_rate",
]


def build_feature_table(min_usage: int = 8) -> pd.DataFrame:
    """Aggregate per-team power-play metrics for clustering."""

    ends = load_csv("ends")
    ends = ends[ends["PowerPlay"].fillna(0) > 0].copy()
    ends["Result"] = ends["Result"].fillna(0).astype(int)

    feature_frame = (
        ends.groupby("TeamID")
        .agg(
            avg_score_gain=("Result", "mean"),
            usage_count=("EndID", "count"),
            three_plus_rate=("Result", lambda x: np.mean(np.array(x) >= 3)),
            blank_rate=("Result", lambda x: np.mean(np.array(x) == 0)),
            steal_rate=("Result", lambda x: np.mean(np.array(x) < 0)),
        )
        .reset_index()
    )

    feature_frame = feature_frame[feature_frame["usage_count"] >= min_usage].reset_index(drop=True)
    return feature_frame


def fit_opponent_clusters(feature_frame: pd.DataFrame, n_clusters: int = 3) -> GaussianMixture:
    """Fit Gaussian mixture to derive opponent archetypes."""

    model = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42)
    model.fit(feature_frame[FEATURE_COLUMNS])
    return model


def assign_clusters(feature_frame: pd.DataFrame, model: GaussianMixture) -> pd.DataFrame:
    """Attach cluster labels and responsibilities to each opponent."""

    responsibilities = model.predict_proba(feature_frame[FEATURE_COLUMNS])
    labels = responsibilities.argmax(axis=1)
    feature_frame = feature_frame.copy()
    feature_frame["cluster"] = labels
    for idx in range(responsibilities.shape[1]):
        feature_frame[f"cluster_{idx}_prob"] = responsibilities[:, idx]
    return feature_frame

