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
    ends = load_csv("ends", usecols=["CompetitionID", "SessionID", "GameID", "EndID", "TeamID", "PowerPlay", "Result"])
    
    processed_ends = []
    for _, group in ends.groupby(["CompetitionID", "SessionID", "GameID", "EndID"]):
        if len(group) != 2:
            continue
        
        group["PowerPlay"] = group["PowerPlay"].fillna(0)
        hammer_rows = group[group["PowerPlay"] > 0]
        defensive_rows = group[group["PowerPlay"] == 0]

        if hammer_rows.empty or defensive_rows.empty:
            continue

        hammer_team_id = hammer_rows["TeamID"].iloc[0]
        hammer_result = hammer_rows["Result"].iloc[0]
        defensive_result = defensive_rows["Result"].iloc[0]
        
        signed_result = 0
        if hammer_result > 0:
            signed_result = hammer_result
        elif defensive_result > 0:
            signed_result = -defensive_result
            
        processed_ends.append({
            "TeamID": hammer_team_id,
            "SignedResult": signed_result
        })

    if not processed_ends:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["TeamID", "usage_count"])

    hammer_results = pd.DataFrame(processed_ends)
    
    feature_frame = (
        hammer_results.groupby("TeamID")
        .agg(
            avg_score_gain=("SignedResult", "mean"),
            usage_count=("SignedResult", "count"),
            three_plus_rate=("SignedResult", lambda x: np.mean(np.array(x) >= 3)),
            blank_rate=("SignedResult", lambda x: np.mean(np.array(x) == 0)),
            steal_rate=("SignedResult", lambda x: np.mean(np.array(x) < 0)),
        )
        .reset_index()
    )

    return feature_frame[feature_frame["usage_count"] >= min_usage].reset_index(drop=True)


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

