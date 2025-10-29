"""
Encode stone layouts into machine-learning friendly representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class LayoutEncoding:
    embedding: np.ndarray
    metadata: Dict[str, float]


def radial_basis_features(stones: np.ndarray, centers: np.ndarray, gamma: float) -> np.ndarray:
    """Compute radial basis function features."""

    if stones.size == 0:
        return np.zeros(len(centers))

    diff = stones[:, None, :] - centers[None, :, :]
    distance_sq = np.sum(diff**2, axis=2)
    return np.exp(-gamma * distance_sq).mean(axis=0)


def encode_layout(stones: np.ndarray, centers: np.ndarray, gamma: float) -> LayoutEncoding:
    """Return embedding and summary stats for a layout."""

    embedding = radial_basis_features(stones, centers, gamma)
    radii = np.linalg.norm(stones, axis=1) if stones.size > 0 else np.array([])

    metadata = {
        "stone_count": stones.shape[0],
        "mean_radius": float(radii.mean()) if radii.size else float("nan"),
        "max_radius": float(radii.max()) if radii.size else float("nan"),
    }
    return LayoutEncoding(embedding=embedding, metadata=metadata)
