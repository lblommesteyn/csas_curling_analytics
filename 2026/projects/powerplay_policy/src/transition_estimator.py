"""
Transition model estimation for the power play MDP.

Fits conditional probability models of end outcomes given state/action pairs.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_csv  # noqa: E402


class TransitionEstimator:
    """Multinomial logistic regression estimator for score outcomes."""

    def __init__(self) -> None:
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.model = LogisticRegression(multi_class="multinomial", max_iter=1000)
        self.outcome_values: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> None:
        """Fit the estimator on engineered state-action data."""

        features = frame.drop(columns=["score_delta"])
        targets = frame["score_delta"]

        self.outcome_values = np.sort(targets.unique())
        encoded = self.encoder.fit_transform(features)
        self.model.fit(encoded, targets)

    def predict_transition(self, state_action: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict outcome probabilities for given state-action combos.

        Returns
        -------
        outcome_values, outcome_probs
        """

        encoded = self.encoder.transform(state_action)
        probs = self.model.predict_proba(encoded)
        return self.outcome_values, probs


def prepare_training_frame() -> pd.DataFrame:
    """Construct training data for the transition estimator."""

    ends = load_csv("ends")
    ends = ends.copy()
    ends["score_delta"] = ends["Result"].fillna(0).astype(int)

    features = ends[
        [
            "CompetitionID",
            "SessionID",
            "GameID",
            "EndID",
            "TeamID",
        ]
    ]

    features["power_play_used"] = ends["PowerPlay"].fillna(0).astype(int) > 0

    # TODO: merge engineered state features when available.
    features["score_delta"] = ends["score_delta"]
    return features
