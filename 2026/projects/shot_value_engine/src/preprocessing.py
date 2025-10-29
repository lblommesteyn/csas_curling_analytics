"""
Shot layout preprocessing utilities.

Tasks:
    - parse Stones.csv into per-shot layouts
    - normalize coordinates via group-theoretic symmetries
    - tag shot intent categories
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Literal, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_csv  # noqa: E402


Symmetry = Literal["identity", "mirror_x", "mirror_y", "rotate_180"]


def _clean_coordinates(stones: np.ndarray) -> np.ndarray:
    """Remove placeholder coordinates and normalise scale to house units."""

    valid = ~np.isnan(stones).any(axis=1)
    valid &= ~(np.all(stones == 0, axis=1))
    valid &= ~(np.all(stones == 4095, axis=1))

    filtered = stones[valid]
    if filtered.size == 0:
        return filtered

    # Normalise: convert 0..4095 grid to approximately -1..1 with center at 0.
    center = 2047.5
    scale = 2047.5
    return (filtered - center) / scale


def canonicalize_layout(stones: np.ndarray) -> tuple[np.ndarray, Symmetry]:
    """
    Map a set of stone coordinates to a canonical configuration.

    Parameters
    ----------
    stones:
        Array of shape (n_stones, 2).

    Returns
    -------
    canonical stones, applied symmetry operation
    """

    stones = _clean_coordinates(stones)
    if stones.size == 0:
        return stones, "identity"

    operations = {
        "identity": stones,
        "mirror_x": stones * np.array([-1, 1]),
        "mirror_y": stones * np.array([1, -1]),
        "rotate_180": stones * -1,
    }

    determinants = {}
    for name, coords in operations.items():
        determinants[name] = np.sum(coords[:, 0] ** 2 + coords[:, 1] ** 2)

    best_symmetry = min(determinants, key=determinants.get)
    return operations[best_symmetry], best_symmetry


def load_shot_layouts() -> pd.DataFrame:
    """Return a tidy frame with per-shot stone layouts."""

    stones = load_csv("stones")
    coord_cols = [col for col in stones.columns if col.startswith("stone_")]

    melted = stones.melt(
        id_vars=["CompetitionID", "SessionID", "GameID", "EndID", "ShotID", "TeamID"],
        value_vars=coord_cols,
        var_name="stone_index",
        value_name="coordinate",
    )
    melted[["stone", "axis"]] = melted["stone_index"].str.extract(r"stone_(\d+)_(x|y)")
    layout = melted.pivot_table(
        index=["CompetitionID", "SessionID", "GameID", "EndID", "ShotID", "TeamID", "stone"],
        columns="axis",
        values="coordinate",
        aggfunc="first",
    ).reset_index()
    layout = layout.rename(columns={"x": "x_coord", "y": "y_coord"})
    return layout


def layout_moments(stones: np.ndarray) -> dict[str, float]:
    """Compute summary moments for a canonicalized layout."""

    if stones.size == 0:
        return {
            "stone_count": 0,
            "mean_radius": np.nan,
            "min_radius": np.nan,
            "max_radius": np.nan,
            "spread": np.nan,
            "mean_abs_x": np.nan,
        }

    radii = np.linalg.norm(stones, axis=1)
    return {
        "stone_count": stones.shape[0],
        "mean_radius": radii.mean(),
        "min_radius": radii.min(),
        "max_radius": radii.max(),
        "spread": radii.max() - radii.min(),
        "mean_abs_x": np.abs(stones[:, 0]).mean(),
    }
