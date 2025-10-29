"""
Shot value estimation model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_csv  # noqa: E402


@dataclass
class ModelArtifacts:
    pipeline: Pipeline
    feature_columns: list[str]


SHOT_FEATURE_COLS = [
    "EndID",
    "ShotID",
    "Task",
    "stone_count",
    "mean_radius",
    "min_radius",
    "max_radius",
    "spread",
    "mean_abs_x",
]


def assemble_features(layout_features: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    if "Result" in layout_features.columns:
        merged = layout_features.copy()
    else:
        ends = load_csv("ends")[
            ["CompetitionID", "SessionID", "GameID", "EndID", "Result"]
        ]
        merged = layout_features.merge(
            ends,
            on=["CompetitionID", "SessionID", "GameID", "EndID"],
            how="left",
        )
    merged["Result"] = merged["Result"].fillna(0).astype(int)
    return merged, merged["Result"].to_numpy()


def build_regression_pipeline() -> Pipeline:
    numeric_features = [col for col in SHOT_FEATURE_COLS if col not in {"Task"}]
    categorical_features = ["Task"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = GradientBoostingRegressor(random_state=42)
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def fit_value_model(layout_features: pd.DataFrame) -> ModelArtifacts:
    features, target = assemble_features(layout_features)
    pipeline = build_regression_pipeline()
    pipeline.fit(features[SHOT_FEATURE_COLS], target)
    return ModelArtifacts(pipeline=pipeline, feature_columns=SHOT_FEATURE_COLS)

