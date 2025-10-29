"""
Opponent shot sequence simulator for power-play scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


@dataclass
class ShotOutcome:
    score_delta: int
    probability: float


def simulate_sequence(outcomes: List[ShotOutcome], trials: int = 10_000) -> np.ndarray:
    """Monte Carlo sampling of opponent shot sequences."""

    probabilities = np.array([shot.probability for shot in outcomes])
    probabilities = probabilities / probabilities.sum()
    deltas = np.array([shot.score_delta for shot in outcomes])
    samples = np.random.choice(deltas, size=trials, p=probabilities)
    return samples
